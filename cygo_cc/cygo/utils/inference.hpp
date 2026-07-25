#ifndef MIGO_INFERENCE_H
#define MIGO_INFERENCE_H

#include <memory>
#include <optional>
#include <random>
#include <tuple>

#include "cygo/state.hpp"

namespace cygo {

typedef std::vector<int8_t> zone_vector_t;
/** make zone, should be consistent with migo.network.zone_plane */
zone_vector_t make_zone_vector(std::string zone_type, int board_size);

/** penalty in logit for invalid moves */
constexpr float penalty_scale = 100;
/** Transforms neural network value outputs for integration with logits. */
template <class T>
void transformQ(T& nnQ, int cvisit = 50, int maxnb = 1, float cscale = 1.0) {
    // [-1, 1] -> [0, 1]
    nnQ /= 2.0;
    nnQ += 0.5;
    // Gumbel MuZero's transformation
    nnQ *= (cvisit + maxnb) * cscale;
}

struct ModelConfig {
    int board_size = 9;
    int history_n = 7;
    int in_channels = 17;
    int aux_policy_channels = 0; // e.g., zone policy output
    std::string device = "cuda";
    int device_id = 0;           // -1 for default cuda stream, 0 or 1 for specific one
};


/** use a learnt model exported as modelname.ts */
struct TorchInferenceModel {
    struct ModelHolder;
    std::unique_ptr<ModelHolder> model;
    ModelConfig config;
    typedef std::default_random_engine rng_t;
    rng_t rng;
    const bool half_precision = true;

    TorchInferenceModel(std::string filename, const ModelConfig& config);
    ~TorchInferenceModel();

    /** run model.
        Argument state_lists must consist of states with the same board size.
        The output vectors will be automatically resized.
        Note: values in values_out are negated as parents' view.

        Params input_zones and input_aux_weights must have the same length as state_list, or empty (w/o zone)
    */
    void run_model(std::vector<cygo::State> const& state_list,
                   std::vector<zone_vector_t> const& input_zones,
                   std::vector<float> const & input_aux_weight,
                   std::vector<float>& policy_out,
                   std::vector<float>& value_out, bool add_noise = false);
    void infer(/* const */ std::vector<int8_t>& input_features,
               std::vector<float> /* const */ & input_aux_weight,
               int batch_size,
               /* const */ std::vector<int8_t>& legals_relaxed,
               std::vector<float>& policy_out, std::vector<float>& value_out,
               bool add_noise);

    /** default 0 for seeding by random device, still envvar CYGO_DETERMINISTIC
     * has superior priority */
    static void manual_seed(uint64_t seed = 0);
    /** show logits and values */
    void inspect(const cygo::State& state, const zone_vector_t& zone, float zone_weight);
    bool extended() const { return config.aux_policy_channels > 0; }
};

class ModelQueue {
   public:
    explicit ModelQueue(const std::shared_ptr<TorchInferenceModel>& model);
    ~ModelQueue() = default;
    /** add state as inference request */
    void push(const cygo::State& state, std::optional<zone_vector_t> zone=std::nullopt, float aux_weight=0.);
    void push(cygo::State&& state, std::optional<zone_vector_t> zone=std::nullopt, float aux_weight=0.);

    void infer(bool add_noise);
    /** receive logits and value for a state */
    std::tuple<float*, float*, float> pop();

    ModelConfig model_config() const { return model->config; }

    bool accepting_push() const { return pop_cur == value_out.size(); }
    bool accepting_pop() const { return pop_cur <= value_out.size(); }

   private:
    std::shared_ptr<TorchInferenceModel> model;
    std::vector<cygo::State> input_state;
    std::vector<zone_vector_t> input_zones;
    std::vector<float> input_aux_weight;
    std::vector<float> policy_out, value_out;
    size_t pop_cur = 0;
};
}  // namespace cygo
#endif

// MIGO_INFERENCE_H
