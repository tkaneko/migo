#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <deque>
#include <iostream>
#include <thread>

#include "game.hpp"

using cygo::GameManager;
using cygo::PlayerModel;
using cygo::SimpleRecord;

void play_games(PlayerModel &player, float komi, bool super_ko,
                int num_games, int parallel,
                std::deque<SimpleRecord> &out) {
    SimpleRecord config{player.board_size(), komi, super_ko};
    config.zone_b = player.primary_zone_name;
    config.zone_w = player.primary_zone_name;
    GameManager game_manager(config, player, parallel);

    while (game_manager.num_completed() < num_games) {
        game_manager.step();
    }
    game_manager.save_games(out);
}

int main(int argc, const char *argv[]) {
    using namespace std::string_literals;
    argparse::ArgumentParser args("cygo-play");

    args.add_argument("model").required().help("path to the model");
    args.add_argument("--device")
        .default_value("cuda:0"s)
        .help("device for model, e.g., cuda:0");
    args.add_argument("--output")
        .default_value("out.csv"s)
        .help("output filename where games are stored");
    args.add_argument("--width").default_value(8).scan<'i', int>().help(
        "root width, 0 for policy only");
    args.add_argument("--reply-width").default_value(0).scan<'i', int>().help(
        "root width to consider reply for root move");
    args.add_argument("--parallel")
        .default_value(64)
        .scan<'i', int>()
        .help("maximum ongoing games in parallel");
    args.add_argument("--games").default_value(1000).scan<'i', int>().help(
        "#games to play");
    args.add_argument("--n-procs")
        .default_value(8)
        .scan<'i', int>()
        .help("number of gpu streams");
    args.add_argument("--quiet")
        .help("do not show progress bar")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--verbose")
        .help("show extra messages")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--komi")
        .default_value(7.0f)
        .scan<'g', float>()
        .help("komi for black");
    args.add_argument("--super-ko")
        .help("use super ko rule")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--greedy")
        .help("disable gumbel noise")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--zone")
        .default_value(""s)
        .help("zone name (full, null, center, edge) or model w/o zone for empty string");
    args.add_argument("--aux-weight")
        .default_value(0.f)
        .scan<'g', float>()
        .help("weight for zone value when zone is enabled");
    args.add_argument("--enable-zone-after")
        .default_value(0)
        .scan<'i', int>()
        .help("provided for compatibility, just ignored");
    args.add_argument("--tqdm-position")
        .default_value(0)
        .scan<'i', int>()
        .help("vertical position of progressbar");

    try {
        args.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << '\n' << args;
        return 1;
    }
    cygo::TorchInferenceModel::manual_seed();

    const auto model_path = args.get("model"), device = args.get("--device"),
        zone = args.get("zone");

    bool super_ko = args["super-ko"] == true;
    bool quiet = args["quiet"] == true;
    bool verbose = args["verbose"] == true;
    bool greedy = args["greedy"] == true;
    int root_width = args.get<int>("width");
    int reply_width = args.get<int>("reply-width");
    int parallel = args.get<int>("parallel");
    int games = args.get<int>("games");
    int n_procs = args.get<int>("n-procs");
    const float komi = args.get<float>("komi");
    int tqdm_position = args.get<int>("tqdm-position");
    float aux_weight = args.get<float>("aux-weight");
    int enable_zone_after = args.get<int>("enable-zone-after");
    
    // dry run to ensure existence of files
    auto player0
        = cygo::make_player(model_path, device, enable_zone_after, aux_weight,
                            zone, root_width, reply_width, greedy);    

    std::vector<std::deque<SimpleRecord>> result(n_procs);
    auto task = [&](int id) {
        auto player = cygo::make_player(model_path, device, enable_zone_after,
                                        aux_weight, zone, root_width,
                                        reply_width, greedy);        
        play_games(player, komi, super_ko, games / n_procs, parallel, result[id]);
    };
    std::vector<std::thread> tasks;
    for (int i = 0; i < n_procs; ++i) tasks.emplace_back(task, i);
    if (!quiet) {
        for (int i = 0; i < tqdm_position; ++i) std::cout << '\n' << std::flush;
        cygo::ProgressBar bar("cygo-play", !quiet);
        {
            while (true) {
                int games_done = GameManager::global_completions.load(
                    std::memory_order_relaxed);
                if (games_done / n_procs >= games / n_procs) break;
                using namespace std::chrono_literals;
                auto progress = 100.0 * games_done / games;
                bar.set_progress(std::min(100.0, progress));
                std::this_thread::sleep_for(50ms);
            }
        }
        for (int i = 0; i < tqdm_position; ++i) std::cout << "\x1b[A";
        std::cout << std::flush;
    }

    for (int i = 0; i < n_procs; ++i) {
        tasks[i].join();
        if (i > 0) {
            result[0].insert(result[0].end(), result[i].begin(),
                             result[i].end());
            result[i].clear();
        }
    }
    auto [black_win, draw, white_win]
        = save_games(result[0], args.get("--output"));
    if (verbose)
        std::cout << black_win << " - " << draw << " - " << white_win << "\n";
    if (verbose && games == 1) {
        auto model = cygo::load_model(model_path, device);

        const auto& game = result[0][0];
        cygo::State state(game.board_size, game.komi, false, 0);
        for (auto move : game.moves) {
            auto cmove = (move >= 0)
                             ? cygo::Move::from_raw(move, game.board_size)
                             : cygo::Move::PASS;
            auto pass2 = cmove.is_pass() && state.last_move().is_pass();
            if (pass2)
                model->inspect(state, {player0.primary_zone}, player0.aux_weight);
                
            state.make_move(cmove);

            if (pass2)
                model->inspect(state, {player0.primary_zone}, player0.aux_weight);
        }
    }        
}
