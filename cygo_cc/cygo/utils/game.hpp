#ifndef MIGO_GAME_H
#define MIGO_GAME_H
#include <array>
#include <vector>
#include <deque>
#include <optional>

#include "inference.hpp"

namespace cygo {
struct SimpleRecord {
    int board_size = 9;
    float komi = 7.0;
    bool super_ko = false;
    // results
    float score = 0;
    std::vector<int16_t> moves = {};
    cygo::Color winner = cygo::Color::EMPTY;
    // misc
    std::string zone_b = "", zone_w = "";
    // maybe added later: territory, zone_score
    bool ends_with_two_passes() const;
};

struct PlayerModel {
    std::shared_ptr<ModelQueue> queue;
    bool greedy = false;
    int root_width = 1, depth1_width = 0;
    std::string delimiter="delimiter";
    int opening_zone_limit = 0;
    float aux_weight = 0.0;
    zone_vector_t primary_zone;
    std::string primary_zone_name = "";

    int board_size() const {
        return queue ? queue->model_config().board_size : 0;
    }
};

/** Object holding state of ongoing game
 *
 * an actual search strategy is given as a parameter player for each method.
 * Data for search
 * - state: root,
 * - next_move: the result after search completion,
 * - score_moves: score for root moves,
 * - depth1_score_moves: score for best opponent's reply for some of root moves.
 */
struct Game {
    SimpleRecord game_config, game;
    int history_n;
    int next_move;
    std::shared_ptr<cygo::State> state;
    std::vector<std::pair<float, int>> score_moves, depth1_score_moves; // logit | value, move_id
    zone_vector_t null_zone;

    Game(const SimpleRecord &config, int history_limit);
    ~Game() = default;
    int board_size() const { return game_config.board_size; }
    int pass_id() const { return board_size() * board_size(); }
    cygo::Move to_move(int id) const {
        return (id == pass_id()) ? cygo::Move::PASS
                                 : cygo::Move::from_raw(id, board_size());
    }
    void reset();
    /** pick the first legal move in moves */
    cygo::Move sample() const;
    std::optional<SimpleRecord> advance();
    /** methods called by GameManager */
    void make_policy_request(const PlayerModel &player) {
        if (player.primary_zone_name == "")
            player.queue->push(*state);
        else {
            int nmoves = game.moves.size();
            auto zone = (nmoves >= player.opening_zone_limit)
                            ? player.primary_zone
                            : null_zone;
            auto weight = (nmoves >= player.opening_zone_limit) ? player.aux_weight : 0.f;
            player.queue->push(*state, std::make_optional(zone), weight);
        }        
    }
    void recv_logits(const PlayerModel &player);
    void make_value_request(const PlayerModel &player);
    void recv_values(const PlayerModel &player);
    void make_depth1_request(const PlayerModel &player);
    void recv_depth1(const PlayerModel &player);
};

/** manage a sequence of games being played in parallel */
class GameManager {
   public:
    static std::atomic<int> global_completions;

    GameManager(SimpleRecord game_config, const PlayerModel &player,
                int parallel);
    void step();
    int num_completed() const { return completed.size(); }
    void save_games(std::deque<SimpleRecord> &out) {
        std::swap(out, completed);
    }
    int steps_elapsed() const { return total_steps; }
    const SimpleRecord &game_of(int id) const { return completed.at(id); }
    void set_white_player(const PlayerModel &white) { players[1] = white; }

   private:
    SimpleRecord game_config;
    std::array<PlayerModel, 2> players;
    std::vector<Game> on_going;
    int total_steps = 0;
    std::deque<SimpleRecord> completed;
    std::vector<int> restart_waiting;
};

std::shared_ptr<cygo::TorchInferenceModel> load_model(
    const std::string &model_path, std::string device);

PlayerModel make_player(const std::string &model_path, std::string device,
                        int opening_zone_limit, float aux_weight,
                        std::string zone, int root_width, int reply_width,
                        bool greedy);

/** save games in csv, with conversion of pass from -2 to -1 */
std::tuple<int, int, int> save_games(const std::deque<SimpleRecord> &completed,
                                     const std::string &path);

class ProgressBar {
   public:
    explicit ProgressBar(const char *name, bool enable, int bar_width = 0);
    ~ProgressBar();
    /** value \in [0, 100] */
    void set_progress(float value);

   private:
    class Data;
    std::shared_ptr<Data> state;
};

/** convert 1:1 win ratio into elo (difference) */
double p2elo(double p, double eps = 2e-4);

}  // namespace cygo
#endif
// MIGO_GAME_H
