import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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


def _format_float_for_tag(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def make_experiment_tag(d, sigma, dividend, x_0):
    return "_".join(
        [
            "call_highdim",
            "model_gbm",
            f"d_{d}",
            "payoff_geometric",
            f"sigma_const_{_format_float_for_tag(sigma)}",
            f"div_const_{_format_float_for_tag(dividend)}",
            f"x0_const_{_format_float_for_tag(x_0)}",
            "corr_indep",
        ]
    )


def default_model_dir(d, lambda_temp):
    if d == 1:
        return "call_trained_models_robust_PI"
    if abs(float(lambda_temp) - 0.01) < 1e-12:
        return "call_trained_models_robust_PI_highdim_gbm-2"
    return "call_trained_models_robust_PI_highdim_gbm"


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
        hidden_layers=3,
        hidden_dim=64,
        lr=0.01,
        device=torch.device("cpu"),
    ):
        super().__init__()
        if d < 1:
            raise ValueError("d must be at least 1.")

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
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device

        print(f"Using device: {self.device}")
        print(f"Dimension: {self.d}")
        print("Payoff type: geometric basket call" if self.d > 1 else "Payoff type: 1D call")

        self.y_networks = nn.ModuleList([self._build_network(d + 1, 1) for _ in range(n_time_steps)])

        self._init_weights()
        self.to(self.device)

        self.optimizers = [
            optim.Adam(list(self.y_networks[t].parameters()), lr=lr)
            for t in range(n_time_steps)
        ]
        self.pi_function = None

    def _init_weights(self):
        for network in self.y_networks:
            for module in network.children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0.0)

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

    def compute_y_grad_y(self, l, x):
        x = _ensure_state_tensor(x, self.d)
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        y = self.y_NNs_forward(l, x)
        grad_y = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
        )[0]
        return y.detach(), grad_y.detach()

    def small_g(self, z):
        z_norm = torch.norm(z, dim=1)
        return (-1.0) * self.epsilon * z_norm

    def f_driver(self, t, x, y, z, pi):
        del t
        f_x = torch.zeros_like(y)
        discount_term = self.r * y

        F_x = self.g_terminal(x)
        control_term = self.K * (F_x - y) * pi
        z_reg_term = self.small_g(z).unsqueeze(1)

        eps = 1e-10
        pi_safe = torch.clamp(pi, min=eps)
        pi_1_safe = torch.clamp(1.0 - pi, min=eps)
        entropy = pi * torch.log(pi_safe) + (1.0 - pi) * torch.log(pi_1_safe)
        entropy_term = self.lambda_temp * entropy

        return f_x - discount_term + z_reg_term + control_term - entropy_term

    def simulate_forward_process(self, batch_size):
        x = torch.zeros(batch_size, self.n_time_steps + 1, self.d, device=self.device)
        x[:, 0, :] = self.x_0

        for t in range(self.n_time_steps):
            dw = np.sqrt(self.dt) * torch.randn(batch_size, self.d, device=self.device)
            x[:, t + 1, :] = x[:, t, :] * torch.exp(
                (self.r - self.dividend - 0.5 * self.sigma**2) * self.dt + self.sigma * dw
            )

        return x

    def train_step_backward(self, x, time_idx, optimizer):
        optimizer.zero_grad()

        x_t = x[:, time_idx, :]
        y_t = self.y_NNs_forward(time_idx, x_t)

        x_next = x[:, time_idx + 1, :]
        dx_t = x_next - x_t
        multiplier = torch.abs(dx_t) / (self.dt**0.5)

        y_next, grad_y_next = self.compute_y_grad_y(time_idx + 1, x_next)
        z_next = multiplier * grad_y_next
        pi_next = self.pi_function(time_idx + 1, x_next)

        f_value = self.f_driver(time_idx, x_next, y_next, z_next, pi_next)
        temp_diff = y_next - y_t + f_value * self.dt
        loss = torch.mean(temp_diff**2)

        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, n_iterations=20000, batch_size=64):
        losses = []
        y0_values = []

        for _ in tqdm(range(n_iterations)):
            x = self.simulate_forward_process(batch_size)
            iteration_loss = 0.0
            for t in range(self.n_time_steps - 1, -1, -1):
                iteration_loss += self.train_step_backward(x, t, self.optimizers[t])

            losses.append(iteration_loss)
            with torch.no_grad():
                y0 = self.y_NNs_forward(0, x[:, 0, :]).mean().item()
            y0_values.append(y0)

        return losses, y0_values


class Policy_Iteration:
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
        device=torch.device("cpu"),
        hidden_layers=3,
        hidden_dim=64,
        lr=0.01,
        model_save_flag=1,
        model_dir=None,
        model_tag=None,
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
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.model_save_flag = model_save_flag
        self.model_dir = model_dir or default_model_dir(d, lambda_temp)
        self.model_tag = model_tag

        self.policy_evaluation_NN_solver = PE_call(
            d=d,
            total_time=total_time,
            n_time_steps=n_time_steps,
            K=K,
            r=r,
            dividend=dividend,
            sigma=sigma,
            strike=strike,
            x_0=x_0,
            lambda_temp=lambda_temp,
            epsilon=epsilon,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            lr=lr,
            device=device,
        )

        if self.d > 1 and self.model_tag is None:
            self.model_tag = make_experiment_tag(d=d, sigma=sigma, dividend=dividend, x_0=x_0)

        self.pi_history = {
            "networks": [],
            "losses": [],
            "y0_values": [],
        }

    def freeze_network(self, network_list):
        frozen_list = nn.ModuleList()
        for net in network_list:
            frozen_net = copy.deepcopy(net)
            for param in frozen_net.parameters():
                param.requires_grad = False
            frozen_net.eval()
            frozen_list.append(frozen_net)
        return frozen_list

    def g_terminal(self, x):
        return self.policy_evaluation_NN_solver.g_terminal(x)

    def y_NNs_forward(self, y_networks, n, x):
        x = _ensure_state_tensor(x, self.d)
        payoff = self.g_terminal(x)

        if n == self.n_time_steps:
            return payoff

        inputs = torch.cat((x, payoff), dim=1)
        return y_networks[n].forward(inputs) + payoff

    def _model_name(self):
        if self.d == 1:
            return f"model_eps_{self.epsilon}_lamda_{self.lambda_temp}_penalty_{self.K}"
        return f"{self.model_tag}_eps_{self.epsilon}_lamda_{self.lambda_temp}_penalty_{self.K}"

    def PolicyIteration(self, PI_iteration=2, PE_iteration=1000, batch_size=2**10):
        print(f"Lambda: {self.lambda_temp}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Dimension: {self.d}")

        start_time = time.time()
        for p in range(PI_iteration):
            print(f"---------------------- PI Step {p + 1} -----------------------")

            y_networks = self.freeze_network(self.policy_evaluation_NN_solver.y_networks)
            self.pi_history["networks"].append(y_networks)

            def pi_function(n, x, _y_nets=y_networks):
                F_x = self.g_terminal(x)
                y_t = self.y_NNs_forward(_y_nets, n, x)
                return torch.sigmoid(-(self.K / self.lambda_temp) * (y_t - F_x))

            self.policy_evaluation_NN_solver.pi_function = pi_function

            losses, y0_values = self.policy_evaluation_NN_solver.train(
                n_iterations=PE_iteration,
                batch_size=batch_size,
            )

            self.pi_history["losses"].append(losses[-1])
            self.pi_history["y0_values"].append(y0_values[-1])
            print("Y0 :", y0_values[-1])

        total_elapsed = time.time() - start_time
        print(f"Total PI time: {total_elapsed:.2f}s")

        if self.model_save_flag == 1:
            model_dir = Path(self.model_dir)
            model_dir.mkdir(exist_ok=True)
            save_path = model_dir / f"{self._model_name()}.pt"
            torch.save(self.policy_evaluation_NN_solver.state_dict(), save_path)
            print(f"Model saved to {save_path}")
