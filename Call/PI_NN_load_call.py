import os

import numpy as np
import torch
import torch.nn as nn


def should_use_compensated_stopping(lambda_temp):
    return abs(float(lambda_temp) - 0.01) < 1e-12


def _ensure_state_tensor(x, d):
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.dim() == 1:
        if d == 1:
            return x.reshape(-1, 1)
        return x.unsqueeze(0)
    return x


def _terminal_call_payoff(x, strike, d):
    x = _ensure_state_tensor(x, d)
    if d == 1:
        return torch.relu(x - strike)

    ### geometric basket payoff for d = 10
    geometric_basket = torch.exp(torch.mean(torch.log(torch.clamp(x, min=1e-12)), dim=1, keepdim=True))
    return torch.relu(geometric_basket - strike)


class PE_call(nn.Module):
    def __init__(
        self,
        d,
        total_time,
        n_time_steps,
        K,
        r,
        dividend,
        sigma,
        strike,
        x_0,
        lambda_temp,
        epsilon,
        use_compensated_stopping=None,
        hidden_layers=3,
        hidden_dim=64,
        test_size=2**17,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.d = d
        self.total_time = total_time
        self.n_time_steps = n_time_steps
        self.dt = total_time / n_time_steps
        self.K = K
        self.r = r
        self.dividend = float(dividend)
        self.sigma = float(sigma)
        self.strike = strike
        self.x_0 = float(x_0)
        self.lambda_temp = lambda_temp
        self.epsilon = epsilon
        self.use_compensated_stopping = (
            should_use_compensated_stopping(lambda_temp)
            if use_compensated_stopping is None
            else bool(use_compensated_stopping)
        )
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.test_size = test_size
        self.device = device

        print(f"Using device: {self.device}")
        print(f"Dimension: {self.d}")
        print(f"Compensated stopping: {self.use_compensated_stopping}")

        self.y_networks = nn.ModuleList([self._build_network(d + 1, 1) for _ in range(n_time_steps)])
        self.to(self.device)

    def _build_network(self, input_dim, output_dim):
        layers = [
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        ]

        for _ in range(self.hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                ]
            )

        layers.append(nn.Linear(self.hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def g_terminal(self, x):
        return _terminal_call_payoff(x, self.strike, self.d)

    def y_NNs_forward(self, l, x):
        x = _ensure_state_tensor(x, self.d)
        payoff = self.g_terminal(x)

        if l == self.n_time_steps:
            return payoff

        inputs = torch.cat((x, payoff), dim=1)
        return self.y_networks[l].forward(inputs) + payoff

    def discounted_payoff(self, t, x):
        discount = torch.exp(-self.r * t)
        if discount.dim() == 0:
            discount = discount.reshape(1, 1)
        elif discount.dim() == 1:
            discount = discount.reshape(-1, 1)
        return discount * self.g_terminal(x)

    def _time_dependent_compensation(self, step_idx, dtype):
        if not self.use_compensated_stopping:
            return torch.zeros((), dtype=dtype, device=self.device)
        time_remaining = self.total_time - step_idx * self.dt
        return torch.as_tensor(
            self.lambda_temp * time_remaining * np.log(2.0),
            dtype=dtype,
            device=self.device,
        )

    def compute_optimal_stopping_times_batch(self, x_trajectories):
        with torch.no_grad():
            batch_size = x_trajectories.shape[0]
            n_steps = self.n_time_steps

            stopping_decisions = torch.zeros(batch_size, n_steps + 1, dtype=torch.bool, device=self.device)
            stopping_decisions[:, n_steps] = True

            stopping_times = torch.full((batch_size,), fill_value=n_steps, dtype=torch.long, device=self.device)
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            for l in range(n_steps):
                if not active_mask.any():
                    break

                x_l = x_trajectories[:, l, :]
                payoff_l = self.g_terminal(x_l).squeeze(1)
                continuation_values = self.y_NNs_forward(l, x_l).squeeze(1)
                compensation = self._time_dependent_compensation(l, x_trajectories.dtype)

                current_decisions = (
                    continuation_values[active_mask] - compensation
                ) <= payoff_l[active_mask]

                stopping_decisions[active_mask, l] = current_decisions

                stopping_indices = active_mask.clone()
                stopping_indices[active_mask] = current_decisions
                stopping_times[stopping_indices] = l
                active_mask[stopping_indices] = False

            return stopping_times, stopping_decisions

    def compute_payoffs_at_stopping_times(self, x_trajectories, stopping_times):
        with torch.no_grad():
            batch_size = x_trajectories.shape[0]
            batch_indices = torch.arange(batch_size, device=self.device)
            x_tau = x_trajectories[batch_indices, stopping_times, :]
            stopping_times_true = stopping_times.to(x_trajectories.dtype) * self.dt
            payoffs = self.discounted_payoff(stopping_times_true, x_tau)
            return payoffs.squeeze(1)

    def evaluate_expected_reward(self, x_test, batch_size=8192):
        with torch.no_grad():
            all_payoffs = []
            n_samples = x_test.shape[0]
            n_batches = (n_samples + batch_size - 1) // batch_size

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                x_trajectories = x_test[start_idx:end_idx, :, :]
                stopping_times, _ = self.compute_optimal_stopping_times_batch(x_trajectories)
                payoffs = self.compute_payoffs_at_stopping_times(x_trajectories, stopping_times)
                all_payoffs.append(payoffs)

            all_payoffs = torch.cat(all_payoffs)
            expected_reward = all_payoffs.mean().item()
            std_error = all_payoffs.std().item() / np.sqrt(n_samples)
            return expected_reward, std_error

    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.load_state_dict(state_dict)
            return True
        return False


PE = PE_call
