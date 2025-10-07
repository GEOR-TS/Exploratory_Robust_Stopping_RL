import torch
import torch.nn as nn
import numpy as np
import os


class PE_call(nn.Module):
    def __init__(self, d, total_time, n_time_steps, K, r, dividend, sigma, strike, x_0, lambda_temp, epsilon,
                 hidden_layers=3, hidden_dim=64, test_size=2**17, device=torch.device('cpu')):
        super(PE_call, self).__init__()
        self.d = d
        self.total_time = total_time
        self.n_time_steps = n_time_steps
        self.dt = total_time / n_time_steps
        self.K = K
        self.r = r
        self.dividend = dividend
        self.sigma = sigma
        self.strike = strike
        self.x_0 = x_0
        self.lambda_temp = lambda_temp
        self.epsilon = epsilon

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.device = device
        print(f"Using device: {self.device}")
        self.test_size = test_size

        self.y_networks = nn.ModuleList([
            self._build_network(d + 1, 1) for _ in range(n_time_steps)
        ])

        self.to(self.device)

    def _build_network(self, input_dim, output_dim):
        layers = []

        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def y_NNs_forward(self, l, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        payoff = self.g_terminal(x)
        if payoff.dim() == 1:
            payoff = payoff.unsqueeze(1)

        if l == self.n_time_steps:
            return payoff
        else:
            inputs = torch.cat((x, payoff), dim=1)
            return self.y_networks[l].forward(inputs) + payoff

    def g_terminal(self, x):
        return torch.relu(x - self.strike)

    def g_terminal_time_batch_misspecification(self, t, x):
        return torch.exp(-self.r * t) * torch.relu( x - self.strike )


    def compute_optimal_stopping_times_batch(self, x_trajectories):
        with torch.no_grad():
            batch_size = x_trajectories.shape[0]
            n_steps = self.n_time_steps

            stopping_decisions = torch.zeros(batch_size, n_steps + 1, dtype=torch.bool, device=self.device)
            # At terminal time T, we always stop
            stopping_decisions[:, n_steps] = True

            stopping_times = torch.ones(batch_size, device=self.device) * n_steps
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            for l in range(n_steps):
                # Skip if all trajectories have already stopped
                if not active_mask.any():
                    break

                x_l = x_trajectories[:, l]
                payoff_l = self.g_terminal(x_l)
                continuation_values = self.y_NNs_forward(l, x_l)
                continuation_values_active = continuation_values[active_mask]
                exercise_values = payoff_l[active_mask]

                # Check stopping condition
                current_decisions = continuation_values_active.squeeze() <= exercise_values.squeeze()

                # Update stopping decisions
                stopping_decisions[active_mask, l] = current_decisions

                # Update stopping times for trajectories that stop at this time step
                stopping_indices = active_mask.clone()
                stopping_indices[active_mask] = current_decisions
                stopping_times[stopping_indices] = l

                active_mask[stopping_indices] = False

            return stopping_times, stopping_decisions

    def compute_payoffs_at_stopping_times(self, x_trajectories, stopping_times):
        with torch.no_grad():
            batch_size = x_trajectories.shape[0]
            batch_indices = torch.arange(batch_size, device=self.device)

            x_tau = x_trajectories[batch_indices, stopping_times.long()].squeeze(-1)
            stopping_times_true = stopping_times * self.dt
            payoffs = self.g_terminal_time_batch_misspecification(stopping_times_true, x_tau)
            return payoffs

    def evaluate_expected_reward(self, x_test, batch_size=8192):
        with torch.no_grad():
            all_payoffs = []
            n_samples = self.test_size
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
            std_error_ER = all_payoffs.std().item() / np.sqrt(n_samples)
            return expected_reward, std_error_ER

    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.load_state_dict(state_dict)
            return True
        return False


