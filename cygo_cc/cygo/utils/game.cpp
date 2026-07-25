#include "game.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <nlohmann/json.hpp>

#include "cygo/color.hpp"
#include "cygo/utils/inference.hpp"
#include "term.hpp"

bool cygo::SimpleRecord::ends_with_two_passes() const {
    auto pass = cygo::Move::PASS.raw();
    return moves.size() >= 2 && moves.back() == pass
           && *(moves.rbegin() + 1) == pass;
}

cygo::Game::Game(const SimpleRecord &config, int history_limit)
    : game_config(config), history_n(history_limit) {
    reset();
    if (!config.moves.empty()) throw std::invalid_argument("config for Game");
    null_zone = make_zone_vector("null", config.board_size);
}

void cygo::Game::reset() {
    state.reset(
        new cygo::State(board_size(), game_config.komi, game_config.super_ko, history_n));
    game = game_config;  // copy members
}

cygo::Move cygo::Game::sample() const {
    for (const auto &[_, id] : score_moves) {
        auto move = to_move(id);
        if (state->is_legal(move)) return move;
    }
    return cygo::Move::PASS;
}

std::optional<cygo::SimpleRecord> cygo::Game::advance() {
    auto move = sample();
#ifndef NDEBUG
    if (!state->is_legal(move)) {
        std::cerr << "found invalid move" << move << "\n" << state->to_string();
        move = cygo::Move::PASS;
    }
#endif
    state->make_move(move);
    game.moves.push_back(move.raw());
    const size_t moves_limit = board_size() * board_size() * 2;
    if (!game.ends_with_two_passes() && game.moves.size() < moves_limit)
        return std::nullopt;

    game.score = state->tromp_taylor_score(cygo::Color::BLACK);
    game.winner = (game.score != 0) ? ((game.score > 0) ? cygo::Color::BLACK
                                                        : cygo::Color::WHITE)
                                    : cygo::Color::EMPTY;
    SimpleRecord ret;
    std::swap(game, ret);
    reset();
    return ret;
}

void cygo::Game::recv_logits(const PlayerModel &player) {
    int width = std::max(4, player.root_width);
    auto [logit_start, logit_end, value] = player.queue->pop();
    score_moves.clear();

    for (int i = 0; i < logit_end - logit_start; ++i) {
        score_moves.emplace_back(logit_start[i], i);
    }
    std::partial_sort(
        score_moves.begin(),
        score_moves.begin()
            + std::min((width + 1),  // 1 for spare in case illegal
                       (int)score_moves.size()),
        score_moves.end(), [](auto l, auto r) { return l.first > r.first; });
    if ((int)score_moves.size() < player.root_width)
        throw std::logic_error("moves less than player.root_width");
}

template <typename T> T sgn(T value) {
    return (T(0) < value) - (value < T(0));
}

void cygo::Game::make_value_request(const PlayerModel &player) {
    int nmoves = game.moves.size();
    auto zone = (nmoves >= player.opening_zone_limit) ? player.primary_zone
                                                      : null_zone;
    auto weight
        = (nmoves >= player.opening_zone_limit) ? player.aux_weight : 0.f;

    for (int i = 0; i < player.root_width; i++) {
        cygo::State child(*state);
        auto &move = score_moves[i];
        bool is_pass = move.second == pass_id();
        if (is_pass || !child.is_legal(to_move(move.second))) {
#ifdef TO_BE_INVESTIGATED_LATER
            bool pass2 = state->last_move().is_pass() && is_pass;
#endif
            child.make_move(cygo::Move::PASS);
            if (!is_pass)  // illegal move
                move.first = -penalty_scale;
#ifdef TO_BE_INVESTIGATED_LATER
            else if (pass2) {
                // consecutive passes terminates a game
                auto terminal_score
                    = child.tromp_taylor_score(state->current_player);
                if (terminal_score > 0) {
                    terminal_score = 1.0;
                    transformQ(terminal_score);
                    move.first += terminal_score;
                }
            }
#endif
        } else
            child.make_move(to_move(move.second));

        if (player.primary_zone_name == "")
            player.queue->push(std::move(child));
        else {
            player.queue->push(std::move(child), std::make_optional(zone), weight);
        }
    }
}

void cygo::Game::recv_values(const PlayerModel &player) {
    std::vector<std::tuple<int,int,float>> child_reply(player.root_width); // root-move, reply, value
    // (1) choose top-{player.root_width} moves in score_move
    for (int i = 0; i < player.root_width; ++i) {
        auto [logit_start, logit_end, value] = player.queue->pop();
        score_moves[i].first += value;  // transformed with sign in advance
        if (player.depth1_width > 0) {
            auto best_reply
                = std::max_element(logit_start, logit_end) - logit_start;
            assert(0 <= best_reply && best_reply <= board_size()*board_size());
            child_reply[i] = std::make_tuple(score_moves[i].second, best_reply, value);
        }
    }
    std::stable_sort(score_moves.begin(),
                     score_moves.begin() + std::max(4, player.root_width),
                     [](auto l, auto r) { return l.first > r.first; });
    // (2) prepare best reply for each top move in (1)
    depth1_score_moves.clear();
    for (int i = 0; i < player.depth1_width; ++i) {
        auto child = std::find_if(
            child_reply.begin(), child_reply.end(),
            [&](auto e) { return std::get<0>(e) == score_moves[i].second; });
        assert(child != child_reply.end());
        auto [_, best_reply, value] = *child;
        depth1_score_moves.emplace_back(value, best_reply);
    }
}

void cygo::Game::make_depth1_request(const PlayerModel &player) {
    int nmoves = game.moves.size();
    auto zone = (nmoves >= player.opening_zone_limit) ? player.primary_zone
                                                      : null_zone;
    auto weight
        = (nmoves >= player.opening_zone_limit) ? player.aux_weight : 0.f;

    for (int i = 0; i < player.depth1_width; i++) {
        cygo::State child(*state);
        auto move = score_moves[i];
        bool is_pass = move.second == pass_id();
        if (is_pass || !child.is_legal(to_move(move.second)))
            child.make_move(cygo::Move::PASS);
        else
            child.make_move(to_move(move.second));

        auto reply = depth1_score_moves[i].second;
        is_pass = reply == pass_id();
        assert(0 <= reply && reply <= board_size()*board_size());
        if (is_pass || !child.is_legal(to_move(reply)))
            child.make_move(cygo::Move::PASS);
        else
            child.make_move(to_move(reply));

        if (player.primary_zone_name == "")
            player.queue->push(std::move(child));
        else {
            player.queue->push(std::move(child), std::make_optional(zone), weight);
        }
    }
}

void cygo::Game::recv_depth1(const PlayerModel &player) {
    for (int i = 0; i < player.depth1_width; ++i) {
        auto [_logit_start, _logit_end, value] = player.queue->pop();
        auto value1 = depth1_score_moves[i].first;
        auto value2 = -value;     // adjust sign for root
        score_moves[i].first += (-value1 + value2) / 2; // = logit + (value1 + value2) / 2
    }
    std::sort(score_moves.begin(), score_moves.begin() + player.depth1_width,
              [](auto l, auto r) { return l.first > r.first; });
}


std::atomic<int> cygo::GameManager::global_completions = 0;  // to be refactored

cygo::GameManager::GameManager(SimpleRecord game_config,
                               const PlayerModel &player, int parallel)
    : game_config(game_config), players({player, player}) {
    for (int i = 0; i < parallel; ++i)
        on_going.emplace_back(game_config,
                              player.queue->model_config().history_n);
}
void cygo::GameManager::step() {
    auto &player = players[total_steps % 2];
    // adjust black plays when total_steps % 2 == 0
    if (total_steps % 2 == 0) {
        // now, black to play
        for (auto id : restart_waiting) on_going[id].reset();
        restart_waiting.clear();
    }
    // root policy
    if (!player.queue->accepting_push())
        throw std::logic_error("policy_request not ready");
    for (auto &game : on_going) {
        game.make_policy_request(player);
    }
    player.queue->infer(!player.greedy);  // usually true, enabling gumbel noise
    for (auto &game : on_going) {
        game.recv_logits(player);
    }
    // child values
    if (player.root_width > 0) {
        if (!player.queue->accepting_push())
            throw std::logic_error("value_request not ready");
        for (auto &game : on_going) {
            game.make_value_request(player);
        }
        player.queue->infer(false);  // no policy noise in this phase
        for (auto &game : on_going) {
            game.recv_values(player);
        }
    }
    // grand child
    if (player.depth1_width > 0) {
        if (!player.queue->accepting_push())
            throw std::logic_error("depth1_request not ready");
        for (auto &game : on_going) {
            game.make_depth1_request(player);
        }
        player.queue->infer(false);  // no policy noise in this phase

        for (auto &game : on_going) {
            game.recv_depth1(player);
        }
    }
    // play
    for (size_t i = 0; i < on_going.size(); ++i) {
        auto &game = on_going[i];
        auto ret = game.advance();
        if (ret) {
            ++global_completions;
            completed.push_back(*ret);
            if (total_steps % 2 == 0)  // the next player is white
                restart_waiting.push_back(i);
        }
    }
    total_steps += 1;
}

std::tuple<int, int, int> cygo::save_games(
    const std::deque<SimpleRecord> &completed, const std::string &path) {
    auto pass = cygo::Move::PASS.raw();
    int black_win = 0, draw = 0, white_win = 0;

    std::ofstream os(path);
    os << "\"boardsize\",\"komi\",\"winner\",\"score\",\"zone_b\",\"zone_w\",\"moves\"\n";
    for (const auto &record : completed) {
        // save one game
        os << record.board_size << "," << record.komi << ","
           << (int)record.winner << "," << record.score << "," << record.zone_b
           << "," << record.zone_w;
        for (auto move : record.moves) {
            os << "," << (move == pass ? -1 : move);
#ifdef CYGO_USE_GTP_FOR_CSV
            auto cmove = (move >= 0)
                             ? cygo::Move::from_raw(move, record.board_size)
                             : cygo::Move::PASS;
            os << "," << cmove.to_string();
#endif
        }
        os << "\n";
        // verify
#ifndef NDEBUG
        cygo::State state(record.board_size, record.komi, false, 0);
        for (auto move : record.moves) {
            auto cmove = (move >= 0)
                             ? cygo::Move::from_raw(move, record.board_size)
                             : cygo::Move::PASS;
            if (!state.is_legal(cmove)) {
                std::cerr << "found invalid move" << move << "\n"
                          << state.to_string();
            }
            state.make_move(cmove);
        }
#endif
        // count
        switch (record.winner) {
            case cygo::Color::BLACK:
                black_win += 1;
                break;
            case cygo::Color::WHITE:
                white_win += 1;
                break;
            default:
                draw += 1;
        }
    }
    return {black_win, draw, white_win};
}

namespace {
auto change_extension(std::string path, std::string ext) {
    auto new_path = std::filesystem::path(path);
    new_path.replace_extension(std::filesystem::path(ext));
    return new_path;
}
}  // anonymous namespace

static std::atomic<int> extended_network_warned = 1; // set to 0 to be verbose
std::shared_ptr<cygo::TorchInferenceModel> cygo::load_model(
    const std::string &model_path, std::string device) {
    auto json_path = change_extension(model_path, "json");

    if (!std::filesystem::exists(model_path)
        || !std::filesystem::exists(json_path)) {
        std::cerr << "model not found " << model_path << ' ' << json_path
                  << '\n';
        exit(1);
    }

    nlohmann::json config;
    {
        std::ifstream is(json_path);
        config = nlohmann::json::parse(is);
    }

    int board_size = config.at("board_size");
    int in_channels = config.at("in_channels");
    std::string network_type = config["network_class"];
    bool has_aux_channel = false;
    int aux_policy_channels = 0;
    if (network_type == "ExtendedNetwork") {
        if (extended_network_warned.fetch_add(1) == 0) {
            std::cerr << "using ExtendedNetwork (alpha)\n";
        }
        has_aux_channel = true;
        aux_policy_channels = config.at("aux_policy_channels");
    }
    else if (network_type != "PVNetwork") {
        std::cerr << "unknown network_type " << network_type << '\n';
        exit(1);
    }
    int history_n = (in_channels - /*color*/ 1 - has_aux_channel) / 2 - 1;
    return std::make_shared<TorchInferenceModel>(
        model_path,
        ModelConfig{board_size, history_n, in_channels, aux_policy_channels, device});
}

cygo::PlayerModel cygo::make_player(const std::string &model_path,
                                    std::string device, int opening_zone_limit,
                                    float aux_weight,
                                    std::string zone, int root_width,
                                    int reply_width, bool greedy) {
    std::shared_ptr<ModelQueue> queue(
        new ModelQueue(load_model(model_path, device)));
    auto cfg = queue->model_config();
    return {
        queue, greedy, root_width, reply_width, "---", opening_zone_limit, aux_weight,
            make_zone_vector(zone, cfg.board_size),
            zone,
    };
}

class cygo::ProgressBar::Data : public indicators::BlockProgressBar {
   public:
    // forward all arguments
    using indicators::BlockProgressBar::BlockProgressBar;
};

cygo::ProgressBar::ProgressBar(const char *name, bool enable, int bar_width) {
    if (!enable) return;
    if (bar_width <= 0) bar_width = term_width(80) - strlen(name) - 32;

    indicators::show_console_cursor(false);
    state.reset(new Data{
        indicators::option::BarWidth{bar_width},
        indicators::option::Start{"["},
        indicators::option::End{"]"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::PrefixText{name},
    });
}
cygo::ProgressBar::~ProgressBar() {
    if (!state) return;

    state->set_progress(100);
    // state->mark_as_completed(); // involves newline
    state.reset();
    indicators::show_console_cursor(true);
    std::cout << "\033[K" << termcolor::reset << std::flush;
}

void cygo::ProgressBar::set_progress(float value) {
    if (state) state->set_progress(std::clamp(value, 0.f, 100.f));
}

double cygo::p2elo(double p, double eps) {
    return -400 * std::log10((1 + eps) / (std::abs(p) + eps / 2) - 1);
}
