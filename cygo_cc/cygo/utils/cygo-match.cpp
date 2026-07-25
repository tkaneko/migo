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

void match_games(PlayerModel &player_a, PlayerModel &player_b,
                 float komi, bool super_ko,
                 int num_games, int parallel, std::deque<SimpleRecord> &out) {
    SimpleRecord config{player_a.board_size(), komi, super_ko};
    config.zone_b = player_a.primary_zone_name;
    config.zone_w = player_b.primary_zone_name;
    GameManager game_manager(config, player_a, parallel);
    game_manager.set_white_player(player_b);

    while (game_manager.num_completed() < num_games) {
        game_manager.step();
    }
    game_manager.save_games(out);
}

int main(int argc, const char *argv[]) {
    using namespace std::string_literals;
    argparse::ArgumentParser args("cygo-play");

    args.add_argument("model").required().help("path to the model");
    args.add_argument("model-b").default_value("").help(
        "path to the model of the second player");
    args.add_argument("--device")
        .default_value("cuda:0"s)
        .help("device for model, e.g., cuda:0");
    args.add_argument("--device-b")
        .default_value("cuda:0"s)
        .help("device for model-b, e.g., cuda:0");
    args.add_argument("--output")
        .default_value("out.csv"s)
        .help("output filename where games are stored");
    args.add_argument("--width").default_value(8).scan<'i', int>().help(
        "root width, 0 for policy only");
    args.add_argument("--width-b")
        .default_value(8)
        .scan<'i', int>()
        .help("root width for player_b, 0 for policy only");
    args.add_argument("--reply-width").default_value(0).scan<'i', int>().help(
        "root width to consider reply for root move");
    args.add_argument("--reply-width-b").default_value(0).scan<'i', int>().help(
        "root width to consider reply for root move");
    args.add_argument("--greedy")
        .help("disable gumbel noise")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--greedy-b")
        .help("disable gumbel noise")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--zone")
        .default_value(""s)
        .help("zone name (full, null, center, edge) or model w/o zone for empty string");
    args.add_argument("--zone-b")
        .default_value(""s)
        .help("zone name (full, null, center, edge) or model w/o zone for empty string");
    args.add_argument("--aux-weight")
        .default_value(0.f)
        .scan<'g', float>()
        .help("weight for zone value when zone is enabled");
    args.add_argument("--aux-weight-b")
        .default_value(0.f)
        .scan<'g', float>()
        .help("weight for zone value when zone is enabled");
    args.add_argument("--enable-zone-after")
        .default_value(0)
        .scan<'i', int>()
        .help("provided for compatibility, just ignored");
    args.add_argument("--enable-zone-after-b")
        .default_value(0)
        .scan<'i', int>()
        .help("provided for compatibility, just ignored");

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
    args.add_argument("--komi").default_value(7.0f).scan<'g', float>().help(
        "komi for black");
    args.add_argument("--quiet")
        .help("do not show progress bar")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--super-ko")
        .help("use super ko rule")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--alternate")
        .help("play additional match with alternated colors")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("--enable-zone-after")
        .default_value(0)
        .scan<'i', int>()
        .help("provided for compatibility, just ignored");
    args.add_argument("--tqdm-position")
        .default_value(0)
        .scan<'i', int>()
        .help("vertical position of progressbar");
    args.add_argument("--verbose")
        .help("show extra messages")
        .default_value(false)
        .implicit_value(true);

    try {
        args.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << '\n' << args;
        return 1;
    }
    bool super_ko = args["super-ko"] == true;
    bool quiet = args["quiet"] == true;
    bool verbose = args["verbose"] == true;

    // first player
    auto model_path = args.get("model"), device = args.get("device"),
        zone = args.get("zone");
    bool greedy = args["greedy"] == true;
    int root_width = args.get<int>("width");
    int reply_width = args.get<int>("reply-width");
    int enable_zone_after = args.get<int>("enable-zone-after");
    float aux_weight = args.get<float>("aux-weight");

    // second player
    auto model_path_b = args.get("model-b"), device_b = args.get("device-b"),
        zone_b = args.get("zone-b");
    bool greedy_b = args["greedy-b"] == true;
    if (model_path_b == ""s) model_path_b = model_path;
    int root_width_b = args.get<int>("width-b");
    int reply_width_b = args.get<int>("reply-width-b");
    int enable_zone_after_b = args.get<int>("enable-zone-after-b");
    float aux_weight_b = args.get<float>("aux-weight-b");

    const int parallel = args.get<int>("parallel");
    const int games = args.get<int>("games");
    const int n_procs = args.get<int>("n-procs");
    const double komi = args.get<float>("komi");
    const bool both_side = args.get<bool>("alternate");
    int tqdm_position = args.get<int>("tqdm-position");

    // dry run to ensure existence of files
    cygo::make_player(model_path, device, enable_zone_after, aux_weight, zone,
                      root_width, reply_width, greedy);
    cygo::make_player(model_path_b, device_b, enable_zone_after_b, aux_weight_b,
                      zone_b, root_width_b, reply_width_b, greedy_b);

    int counts[3] = {0};
    for (int z = 0; z <= both_side; ++z) {  // runs one or two times
        std::vector<std::deque<SimpleRecord>> result(n_procs);
        auto task = [&](int id) {
            auto player = cygo::make_player(model_path, device,
                                            enable_zone_after, aux_weight, zone,
                                            root_width, reply_width, greedy);
            auto player_b = cygo::make_player(
                model_path_b, device_b, enable_zone_after_b, aux_weight_b,
                zone_b, root_width_b, reply_width_b, greedy_b);
            match_games(player, player_b, komi, super_ko, games / n_procs, parallel,
                        result[id]);
        };
        std::vector<std::thread> tasks;
        for (int i = 0; i < n_procs; ++i) tasks.emplace_back(task, i);

        if (!quiet) {
            for (int i = 0; i < tqdm_position; ++i) std::cout << '\n' << std::flush;
            cygo::ProgressBar bar("cygo-match", !quiet);
            while (true) {
                int games_done = GameManager::global_completions.load(
                    std::memory_order_relaxed);
                if (games_done / n_procs >= games / n_procs) break;
                using namespace std::chrono_literals;
                auto progress = 100.0 * games_done / games;
                bar.set_progress(std::min(100.0, progress));
                std::this_thread::sleep_for(50ms);
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
        counts[0 + z * 2] += black_win;
        counts[1] += draw;
        counts[2 - z * 2] += white_win;

        // alternate colors
        std::swap(model_path, model_path_b);
        std::swap(device, device_b);
        std::swap(aux_weight, aux_weight_b);
        std::swap(zone, zone_b);
        std::swap(root_width, root_width_b);
        std::swap(reply_width, reply_width_b);
        std::swap(greedy, greedy_b);
        // umm
        GameManager::global_completions = 0;
    }
    auto ratio
        = (counts[0] + counts[1] / 2.) / (counts[0] + counts[1] + counts[2]);
    if (verbose)
        std::cout << ratio << ' ' << cygo::p2elo(ratio) << '\n';
}
