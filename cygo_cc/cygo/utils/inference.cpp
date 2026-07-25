#include "inference.hpp"

#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/types.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torch_tensorrt/core/runtime/runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "cygo/features.hpp"

namespace {
bool env_set(const char* env) {
    if (!env || !*env || *env == '0') return false;
    return true;
}

/** make random number generator with seeds */
cygo::TorchInferenceModel::rng_t make_rng() {
    typedef cygo::TorchInferenceModel::rng_t rng_t;
    static const auto env = std::getenv("CYGO_DETERMINISTIC");
    static rng_t::result_type cnt = 0;
    static std::random_device rdev;
    return rng_t(env_set(env) ? (cnt++) : rdev());
}

    std::map<int,int> zone_center = {{9, 2}, {13, 2}, {19, 3}}; // board_size -> corner

}  // namespace

std::vector<int8_t> cygo::make_zone_vector(std::string zone_type, int board_size) {
    if (zone_type == "") return {};
    std::vector<int8_t> ret(board_size * board_size, 0);
    if (zone_type == "full") std::fill(ret.begin(), ret.end(), 1);
    else if (zone_type != "null") {
        if (zone_type != "center" && zone_type != "edge")
            throw std::invalid_argument("zone not supported " + zone_type);

        auto corner = zone_center.at(board_size);
        for (auto r = 0; r < board_size; ++r) {
            bool in_center_r = (corner <= r) && (r + corner < board_size);
            for (auto c = 0; c < board_size; ++c) {
                bool in_center = in_center_r
                    && (corner <= c) && (c + corner < board_size);
                if (zone_type == "center") ret[r * board_size + c] = in_center;
                if (zone_type == "edge") ret[r * board_size + c] = !in_center;
            }
        }
    }
    return ret;
}

void cygo::TorchInferenceModel::manual_seed(uint64_t seed) {
    static const auto env = std::getenv("CYGO_DETERMINISTIC");
    if (env_set(env))
        seed = 0;
    else if (seed == 0) {
        auto rng = make_rng();
        std::uniform_int_distribution<rng_t::result_type> dist(
            0, std::numeric_limits<rng_t::result_type>::max());
        seed = dist(rng);
    }
    torch::manual_seed(seed);
}

struct cygo::TorchInferenceModel::ModelHolder {
    cudaStream_t raw_stream;
    std::unique_ptr<torch::cuda::CUDAStream> torch_stream;
    std::unique_ptr<torch::jit::Module> runtime;

    ~ModelHolder() {
        if (torch_stream && runtime) {
            torch_stream->synchronize();
            runtime.reset();
        }
        torch_stream.reset();
        auto err = cudaStreamSynchronize(raw_stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to synchronize CUDA stream: "
                      << cudaGetErrorString(err) << std::endl;
        }
        err = cudaStreamDestroy(raw_stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to destroy CUDA stream: "
                      << cudaGetErrorString(err) << std::endl;
        }
    }
};

cygo::TorchInferenceModel::TorchInferenceModel(std::string filename,
                                               const ModelConfig& config)
    : config(config), rng(make_rng()) {
    torch_tensorrt::core::runtime::set_multi_device_safe_mode(true);
    if (config.device.find("cuda:") == 0) {
        try {
            this->config.device_id = std::stoi(config.device.substr(5));
        } catch (...) {
            this->config.device_id = 0;
        }
    }
    torch::Device device(torch::kCUDA, this->config.device_id);
    torch::DeviceGuard device_guard(device);
    model.reset(new ModelHolder);

    auto err = cudaStreamCreate(&model->raw_stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err)
                  << std::endl;
        throw std::invalid_argument("stream init error");
    }

    model->torch_stream.reset(
        new torch::cuda::CUDAStream(torch::cuda::getStreamFromExternal(
            model->raw_stream, this->config.device_id)));

    // torch::cuda::CUDAStreamGuard stream_guard(*model->torch_stream);
    model->runtime.reset(
        new torch::jit::Module(torch::jit::load(filename, device)));
    // model->runtime->to(device);
}

cygo::TorchInferenceModel::~TorchInferenceModel() {}

void cygo::TorchInferenceModel::infer(std::vector<int8_t>& nn_input,
                                      std::vector<float>& aux_weight,
                                      int batch_size,
                                      std::vector<int8_t>& legals_relaxed,
                                      std::vector<float>& policy_out,
                                      std::vector<float>& value_out,
                                      bool add_noise) {
    int n_channels = config.in_channels;
    torch::Device device(torch::kCUDA, this->config.device_id);
    torch::DeviceGuard device_guard(device);
    torch::cuda::CUDAStreamGuard stream_guard(*model->torch_stream);

    int channel_size = config.board_size * config.board_size;
    int logit_dim = (channel_size + 1);
    // transfer to gpu
    // int8 features
    torch::Tensor f = torch::from_blob(
        &nn_input[0],
        {batch_size, n_channels, config.board_size, config.board_size},
        torch::TensorOptions().dtype(torch::kInt8));
    torch::Tensor tt_aux_weight
        = torch::from_blob(&aux_weight[0], {(int)aux_weight.size()},
                           torch::TensorOptions().dtype(torch::kFloat));
    
    tt_aux_weight = tt_aux_weight.to(device, true);

    f = f.to(device, /*nonblocking*/ true);
    // after transferred, retrieve float16
    f = f.to(torch::kHalf, /*nonblocking*/ true);
    auto legals_adjust
        = torch::from_blob(&legals_relaxed[0], {batch_size, channel_size + 1},
                           torch::TensorOptions().dtype(torch::kInt8));
    legals_adjust = legals_adjust.to(device, /*nonblocking*/ true);

    // main inference
    auto out = model->runtime->forward({f});
    auto tt = out.toTuple()->elements();
    auto tt_logits = tt[0].toTensor();
    auto tt_value = tt[1].toTensor();

    // adjust values
    if (extended()) {
        // tt[2] for aux policy
        // std::cerr << tt_aux_weight.sizes() << ' ' << tt_aux_weight.device() << '\n';
        auto tt_aux_value = tt[3].toTensor();
        if (aux_weight.size() > 0)
            tt_value = (1 - tt_aux_weight) * tt_value + tt_aux_weight * tt_aux_value;
    }
    tt_value *= -1;  // view from parent
    transformQ(tt_value);

    // adjust logits
    tt_logits = tt_logits.to(torch::kFloat, /*nonblocking*/ false);
    legals_adjust = (1 - legals_adjust).to(torch::kFloat, /*nonblocking*/ false)
                    * penalty_scale;

    auto slice = tt_logits.reshape({batch_size, -1, logit_dim});
    auto primary_logits
        = slice.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    if (extended()) {
        tt_aux_weight = tt_aux_weight.unsqueeze(-1);
        // std::cerr << tt_aux_weight.sizes() << ' ' << tt_aux_weight.device() << '\n';
        
        auto aux_logits = slice.index({torch::indexing::Slice(), 1, torch::indexing::Slice()});
        if (aux_weight.size() > 0)
            primary_logits = (1 - tt_aux_weight) * primary_logits + tt_aux_weight * aux_logits;
        tt_logits = primary_logits.contiguous();
    }
    tt_logits -= legals_adjust;  // for primary logits


    if (add_noise) {
        torch::Tensor noise = torch::rand(
            tt_logits.sizes(), torch::TensorOptions().device(device));
        noise.clamp(0 + 1.1755e-38,
                    1 - 1.1921e-07);  // follow torch.distributions.Gumbel
                                      // U[finfo.tiny, 1-finfo.eps]
        noise = -torch::log(noise);
        noise = -torch::log(noise);

        tt_logits += noise;
    }

    // retrieve results
    auto moves_tensor = tt_logits.to(torch::kCPU, /*nonblocking*/ true);
    auto value_tensor = tt_value.to(torch::kCPU)
                            .to(torch::kFloat);  // blocking here is important
                                                 // at least for dynamo
    model->torch_stream->synchronize();

    policy_out.resize(batch_size * logit_dim);
    value_out.resize(batch_size);
    auto values = value_tensor.accessor<float, 2>();
    auto logits = moves_tensor.accessor<float, 2>();
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < logit_dim; ++i)
            policy_out[b * logit_dim + i] = logits[b][i];

    for (int b = 0; b < batch_size; ++b) value_out[b] = values[b][0];
}

void cygo::TorchInferenceModel::run_model(
    std::vector<cygo::State> const& state_list,
    std::vector<zone_vector_t> const& input_zones,
    std::vector<float> const& input_aux_weight,
    std::vector<float>& policy_out, std::vector<float>& value_out,
    bool add_noise) {
    // prepare features
    if (state_list.empty()) return;
    int batch_size = state_list.size(), history_n = config.history_n;
    int board_size = state_list[0].board_size();
    if (board_size != config.board_size)
        throw std::invalid_argument("board size");
    int n_channels = config.in_channels;
    int channel_size = board_size * board_size,
        size_per_state = n_channels * channel_size;
    std::vector<int8_t> nn_input(batch_size * size_per_state),
        legal_moves_relaxed(batch_size * (channel_size + 1));

    if (config.aux_policy_channels == 0) {
        if (input_zones.size() > 0 || input_aux_weight.size() > 0)
            throw std::invalid_argument("no channels but has associated zones/weights");
        cygo::feature_impl::batch_features_to_ptr(
            state_list, history_n, channel_size, size_per_state, &nn_input[0],
            &legal_moves_relaxed[0]);
    } else {
        if ((int)input_zones.size() != batch_size || (int)input_aux_weight.size() != batch_size)
            throw std::invalid_argument(
                "aux zone/weights mismatch " + std::to_string(batch_size)
                + " " + std::to_string(input_zones.size())
                + " " + std::to_string(input_aux_weight.size()));        
        cygo::feature_impl::batch_features_with_zone_to_ptr(
            state_list, history_n, [&](int id) { return input_zones[id]; },
            channel_size, size_per_state, &nn_input[0],
            &legal_moves_relaxed[0]);
    }
    // inference
    infer(nn_input, const_cast<std::vector<float>&>(input_aux_weight), batch_size, legal_moves_relaxed,
          policy_out, value_out, add_noise);
}

void cygo::TorchInferenceModel::inspect(const cygo::State& state, const zone_vector_t& zone, float zone_weight) {
    std::vector<float> policy_out, value_out;

    // inspect the state
    run_model({state}, {zone}, {zone_weight}, policy_out, value_out, false);

    auto legal_moves = state.legal_moves(state.current_player, true);

    auto state_value = value_out[0];
    std::vector<std::tuple<float, float, cygo::Move>> logit_moves;
    std::vector<cygo::State> children;
    int board_size = state.board_size();
    for (int i = 0; i <= board_size * board_size; ++i) {
        auto cmove = (i < board_size * board_size)
                         ? cygo::Move::from_raw(i, board_size)
                         : cygo::Move::PASS;
        if (cmove.is_pass() || state.is_legal(cmove)) {
            logit_moves.emplace_back(policy_out[i], policy_out[i], cmove);
            cygo::State child(state);
            child.make_move(cmove);
            children.push_back(std::move(child));
        }
    }

    // inspect children
    run_model({children}, {zone}, {zone_weight}, policy_out, value_out, false);
    for (size_t i = 0; i < children.size(); ++i)
        std::get<0>(logit_moves[i]) += value_out[i];

    std::sort(logit_moves.begin(), logit_moves.end(),
              [](auto l, auto r) { return std::get<0>(l) > std::get<0>(r); });

    std::cout << state.to_string() << '\n';
    std::cout << (state.current_player == cygo::Color::BLACK ? "B" : "W")
              << " to play" << '\n';
    for (auto [total, logit, move] : logit_moves)
        std::cout << std::setprecision(4) << std::setw(7) << total
                  << std::setprecision(4) << std::setw(7) << logit << " "
                  << move << "\n";
    std::cout << "value from parent " << state_value << '\n';
}

cygo::ModelQueue::ModelQueue(const std::shared_ptr<TorchInferenceModel>& model)
    : model(model) {
    //
    int expected_bs = 1024;
    input_state.reserve(expected_bs);
    input_zones.reserve(expected_bs);
}
void cygo::ModelQueue::push(const cygo::State& state,
                            std::optional<zone_vector_t> zone,
                            float aux_weight) {
    input_state.push_back(state);
    if (zone) {
        input_zones.push_back(*zone);
        input_aux_weight.push_back(aux_weight);
    }
}
void cygo::ModelQueue::push(cygo::State&& state,
                            std::optional<zone_vector_t> zone,
                            float aux_weight) {
    input_state.push_back(state);
    if (zone) {
        input_zones.push_back(*zone);
        input_aux_weight.push_back(aux_weight);
    }
}

void cygo::ModelQueue::infer(bool add_noise) {
    model->run_model(input_state, input_zones, input_aux_weight, policy_out, value_out, add_noise);
    pop_cur = 0;
    input_state.clear();
    input_zones.clear();
    input_aux_weight.clear();
}

std::tuple<float*, float*, float> cygo::ModelQueue::pop() {
    if (!accepting_pop()) throw std::logic_error("pop");
    const int board_size = model->config.board_size;
    const int logit_dim = board_size * board_size + 1;  // 1 for pass
    int id = pop_cur++;
    auto first = &policy_out[id * logit_dim];
    return {first, first + logit_dim, value_out[id]};
}
