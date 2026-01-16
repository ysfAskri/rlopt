// SwiftSim PPO Agent
// LibTorch-based Proximal Policy Optimization for drone control
// Based on "The 37 Implementation Details of PPO" (Huang et al.)

#pragma once

#include <torch/torch.h>
#include <vector>
#include <random>
#include <memory>

namespace swiftsim {
namespace rl {

// ============================================================================
// ACTOR-CRITIC NETWORK
// ============================================================================
struct ActorCriticImpl : torch::nn::Module {
    // Shared layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    // Actor head (policy)
    torch::nn::Linear actor_mean{nullptr};
    torch::nn::Linear actor_logstd{nullptr};

    // Critic head (value)
    torch::nn::Linear critic{nullptr};

    int obs_dim;
    int act_dim;

    ActorCriticImpl(int obs_dim_, int act_dim_, int hidden_size = 256)
        : obs_dim(obs_dim_), act_dim(act_dim_) {

        // Shared feature extractor
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, hidden_size));

        // Actor (outputs mean of action distribution)
        actor_mean = register_module("actor_mean", torch::nn::Linear(hidden_size, act_dim));
        actor_logstd = register_module("actor_logstd", torch::nn::Linear(hidden_size, act_dim));

        // Critic (outputs state value)
        critic = register_module("critic", torch::nn::Linear(hidden_size, 1));

        // Initialize weights (orthogonal initialization)
        init_weights();
    }

    void init_weights() {
        for (auto& module : modules(false)) {
            if (auto* linear = module->as<torch::nn::Linear>()) {
                torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
                torch::nn::init::constant_(linear->bias, 0.0);
            }
        }
        // Smaller init for output layers
        torch::nn::init::orthogonal_(actor_mean->weight, 0.01);
        torch::nn::init::orthogonal_(critic->weight, 1.0);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward_actor(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));

        auto mean = actor_mean->forward(x);
        auto logstd = actor_logstd->forward(x);
        logstd = torch::clamp(logstd, -20.0, 2.0);  // Stability

        return {mean, logstd};
    }

    torch::Tensor forward_critic(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        return critic->forward(x);
    }

    // Sample action from policy
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    get_action(torch::Tensor obs, bool deterministic = false) {
        auto [mean, logstd] = forward_actor(obs);
        auto std = torch::exp(logstd);

        torch::Tensor action;
        if (deterministic) {
            action = mean;
        } else {
            auto noise = torch::randn_like(mean);
            action = mean + std * noise;
        }

        // Compute log probability
        auto log_prob = -0.5f * (
            torch::pow((action - mean) / std, 2) +
            2.0f * logstd +
            std::log(2.0f * M_PI)
        );
        log_prob = log_prob.sum(-1, true);

        // Clamp action to valid range [0, 1]
        action = torch::sigmoid(action);  // Squash to [0, 1]

        return {action, log_prob, mean};
    }

    // Evaluate action (for PPO update)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate_action(torch::Tensor obs, torch::Tensor action) {
        auto [mean, logstd] = forward_actor(obs);
        auto std = torch::exp(logstd);

        // Unsquash action for log prob calculation
        action = torch::clamp(action, 1e-6, 1.0 - 1e-6);
        auto action_unsquashed = torch::log(action / (1.0 - action));  // Inverse sigmoid

        auto log_prob = -0.5f * (
            torch::pow((action_unsquashed - mean) / std, 2) +
            2.0f * logstd +
            std::log(2.0f * M_PI)
        );
        log_prob = log_prob.sum(-1, true);

        // Entropy
        auto entropy = 0.5f * (1.0f + std::log(2.0f * M_PI)) + logstd;
        entropy = entropy.sum(-1, true);

        // Value
        auto value = forward_critic(obs);

        return {log_prob, entropy, value};
    }
};
TORCH_MODULE(ActorCritic);

// ============================================================================
// ROLLOUT BUFFER
// ============================================================================
class RolloutBuffer {
public:
    std::vector<torch::Tensor> observations;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> log_probs;
    std::vector<torch::Tensor> rewards;
    std::vector<torch::Tensor> dones;
    std::vector<torch::Tensor> values;

    torch::Tensor advantages;
    torch::Tensor returns;

    size_t buffer_size;
    size_t n_envs;
    size_t pos = 0;
    bool full = false;

    torch::Device device;

    RolloutBuffer(size_t buffer_size_, size_t n_envs_, torch::Device device_)
        : buffer_size(buffer_size_), n_envs(n_envs_), device(device_) {
        reset();
    }

    void reset() {
        observations.clear();
        actions.clear();
        log_probs.clear();
        rewards.clear();
        dones.clear();
        values.clear();
        pos = 0;
        full = false;
    }

    void add(torch::Tensor obs, torch::Tensor action, torch::Tensor log_prob,
             torch::Tensor reward, torch::Tensor done, torch::Tensor value) {
        observations.push_back(obs.to(device));
        actions.push_back(action.to(device));
        log_probs.push_back(log_prob.to(device));
        rewards.push_back(reward.to(device));
        dones.push_back(done.to(device));
        values.push_back(value.to(device));
        pos++;
        if (pos >= buffer_size) {
            full = true;
        }
    }

    void compute_returns_and_advantages(torch::Tensor last_value, float gamma = 0.99f,
                                        float gae_lambda = 0.95f) {
        size_t T = observations.size();
        advantages = torch::zeros({(int64_t)T, (int64_t)n_envs, 1}, device);
        returns = torch::zeros({(int64_t)T, (int64_t)n_envs, 1}, device);

        torch::Tensor last_gae = torch::zeros({(int64_t)n_envs, 1}, device);

        for (int64_t t = T - 1; t >= 0; t--) {
            torch::Tensor next_value;
            if (t == (int64_t)T - 1) {
                next_value = last_value;
            } else {
                next_value = values[t + 1];
            }

            auto delta = rewards[t] + gamma * next_value * (1.0f - dones[t]) - values[t];
            last_gae = delta + gamma * gae_lambda * (1.0f - dones[t]) * last_gae;

            advantages[t] = last_gae;
            returns[t] = advantages[t] + values[t];
        }

        // Normalize advantages
        auto adv_flat = advantages.view({-1});
        auto mean = adv_flat.mean();
        auto std = adv_flat.std() + 1e-8f;
        advantages = (advantages - mean) / std;
    }
};

// ============================================================================
// PPO AGENT
// ============================================================================
class PPOAgent {
public:
    ActorCritic network;
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::unique_ptr<RolloutBuffer> buffer;

    torch::Device device;

    // Hyperparameters
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_range = 0.2f;
    float clip_range_vf = 0.2f;
    float ent_coef = 0.01f;
    float vf_coef = 0.5f;
    float max_grad_norm = 0.5f;
    int n_epochs = 10;
    int batch_size = 64;

    size_t n_envs;
    size_t buffer_size;

    PPOAgent(int obs_dim, int act_dim, size_t n_envs_, size_t buffer_size_,
             torch::Device device_ = torch::kCUDA)
        : device(device_), n_envs(n_envs_), buffer_size(buffer_size_) {

        network = ActorCritic(obs_dim, act_dim);
        network->to(device);

        optimizer = std::make_unique<torch::optim::Adam>(
            network->parameters(),
            torch::optim::AdamOptions(learning_rate)
        );

        buffer = std::make_unique<RolloutBuffer>(buffer_size, n_envs, device);
    }

    torch::Tensor select_action(torch::Tensor obs, bool deterministic = false) {
        torch::NoGradGuard no_grad;
        obs = obs.to(device);
        auto [action, log_prob, mean] = network->get_action(obs, deterministic);
        return action;
    }

    void collect_rollout(torch::Tensor obs, torch::Tensor action, torch::Tensor reward,
                         torch::Tensor done) {
        torch::NoGradGuard no_grad;
        obs = obs.to(device);
        action = action.to(device);

        auto [log_prob, entropy, value] = network->evaluate_action(obs, action);

        buffer->add(obs, action, log_prob, reward.to(device), done.to(device), value);
    }

    std::tuple<float, float, float> update(torch::Tensor last_obs) {
        torch::NoGradGuard no_grad_guard;
        last_obs = last_obs.to(device);
        auto last_value = network->forward_critic(last_obs);

        buffer->compute_returns_and_advantages(last_value, gamma, gae_lambda);

        // Stack all data
        auto all_obs = torch::stack(buffer->observations);
        auto all_actions = torch::stack(buffer->actions);
        auto all_old_log_probs = torch::stack(buffer->log_probs);
        auto all_advantages = buffer->advantages;
        auto all_returns = buffer->returns;
        auto all_old_values = torch::stack(buffer->values);

        // Flatten for batching
        int64_t total = all_obs.size(0) * all_obs.size(1);
        all_obs = all_obs.view({total, -1});
        all_actions = all_actions.view({total, -1});
        all_old_log_probs = all_old_log_probs.view({total, -1});
        all_advantages = all_advantages.view({total, -1});
        all_returns = all_returns.view({total, -1});
        all_old_values = all_old_values.view({total, -1});

        float total_pg_loss = 0.0f;
        float total_vf_loss = 0.0f;
        float total_entropy = 0.0f;
        int n_updates = 0;

        // PPO epochs
        for (int epoch = 0; epoch < n_epochs; epoch++) {
            // Shuffle indices
            auto indices = torch::randperm(total, torch::kLong).to(device);

            for (int64_t start = 0; start < total; start += batch_size) {
                int64_t end = std::min(start + batch_size, total);
                auto batch_indices = indices.slice(0, start, end);

                auto obs_batch = all_obs.index_select(0, batch_indices);
                auto actions_batch = all_actions.index_select(0, batch_indices);
                auto old_log_probs_batch = all_old_log_probs.index_select(0, batch_indices);
                auto advantages_batch = all_advantages.index_select(0, batch_indices);
                auto returns_batch = all_returns.index_select(0, batch_indices);
                auto old_values_batch = all_old_values.index_select(0, batch_indices);

                // Evaluate actions with current policy
                auto [log_probs, entropy, values] =
                    network->evaluate_action(obs_batch, actions_batch);

                // Policy loss (PPO clipped objective)
                auto ratio = torch::exp(log_probs - old_log_probs_batch);
                auto pg_loss1 = -advantages_batch * ratio;
                auto pg_loss2 = -advantages_batch *
                    torch::clamp(ratio, 1.0f - clip_range, 1.0f + clip_range);
                auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

                // Value loss (clipped)
                auto values_clipped = old_values_batch +
                    torch::clamp(values - old_values_batch, -clip_range_vf, clip_range_vf);
                auto vf_loss1 = torch::pow(values - returns_batch, 2);
                auto vf_loss2 = torch::pow(values_clipped - returns_batch, 2);
                auto vf_loss = 0.5f * torch::max(vf_loss1, vf_loss2).mean();

                // Entropy loss
                auto entropy_loss = -entropy.mean();

                // Total loss
                auto loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss;

                // Optimize
                optimizer->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(network->parameters(), max_grad_norm);
                optimizer->step();

                total_pg_loss += pg_loss.item<float>();
                total_vf_loss += vf_loss.item<float>();
                total_entropy += (-entropy_loss).item<float>();
                n_updates++;
            }
        }

        buffer->reset();

        return {
            total_pg_loss / n_updates,
            total_vf_loss / n_updates,
            total_entropy / n_updates
        };
    }

    void save(const std::string& path) {
        torch::save(network, path);
    }

    void load(const std::string& path) {
        torch::load(network, path);
    }
};

} // namespace rl
} // namespace swiftsim
