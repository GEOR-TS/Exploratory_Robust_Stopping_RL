import time

import matplotlib.pyplot as plt
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


def _ensure_payoff_tensor(payoff):
    if payoff.dim() == 0:
        return payoff.reshape(1, 1)
    if payoff.dim() == 1:
        return payoff.reshape(-1, 1)
    return payoff


def _terminal_put_payoff(x, strike, d):
    x = _ensure_state_tensor(x, d)
    if d == 1:
        return torch.relu(strike - x)
    basket_average = torch.mean(x, dim=1, keepdim=True) ### basket put payoff
    return torch.relu(strike - basket_average)


class DeepBackward(nn.Module):
    def __init__(
        self,
        d,
        total_time,
        n_time_steps,
        K,
        r,
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
        super(DeepBackward, self).__init__()
        self.d = d
        self.total_time = total_time
        self.n_time_steps = n_time_steps
        self.dt = total_time / n_time_steps
        self.K = K
        self.r = r
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
        print(f"Sigma: {self.sigma}")

        self.y_networks = nn.ModuleList([self._build_network(d + 1, 1) for _ in range(n_time_steps)])
        self.z_networks = nn.ModuleList([self._build_network(d, d) for _ in range(n_time_steps)])

        self._init_weights()
        self.to(self.device)

        self.optimizers = [
            optim.Adam(
                list(self.y_networks[t].parameters()) + list(self.z_networks[t].parameters()),
                lr=lr,
            )
            for t in range(n_time_steps)
        ]

    def _init_weights(self):
        for network in self.z_networks:
            for module in network.children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
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
        return _terminal_put_payoff(x, self.strike, self.d)

    def y_NNs_forward(self, l, x, payoff):
        x = _ensure_state_tensor(x, self.d)
        payoff = _ensure_payoff_tensor(payoff)
        inputs = torch.cat((x, payoff), dim=1)
        return self.y_networks[l].forward(inputs) + payoff

    def small_g(self, z):
        z_norm = torch.norm(z, dim=1)
        return (-1.0) * self.epsilon * z_norm

    def f_driver(self, t, x, y, z):
        f_x = torch.zeros_like(y)
        discount_term = self.r * y

        F_x = self.g_terminal(x)
        penalty_term = self.K * (F_x - y)

        exponent = -(self.K / self.lambda_temp) * (F_x - y)
        exp_0 = torch.zeros_like(y)
        log_1_exp = torch.logaddexp(exp_0, exponent)
        non_linear_x_y = penalty_term + log_1_exp
        z_reg_term = self.small_g(z).unsqueeze(1)

        return f_x - discount_term + z_reg_term + non_linear_x_y

    def simulate_forward_process(self, batch_size):
        x = torch.zeros(batch_size, self.n_time_steps + 1, self.d, device=self.device)
        x[:, 0, :] = self.x_0
        dw = torch.zeros(batch_size, self.n_time_steps, self.d, device=self.device)

        for t in range(self.n_time_steps):
            dw[:, t, :] = np.sqrt(self.dt) * torch.randn(batch_size, self.d, device=self.device)
            x[:, t + 1, :] = x[:, t, :] * torch.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dw[:, t, :]
            )

        return x, dw

    def train_step_backward(self, x, dw, time_idx, optimizer):
        optimizer.zero_grad()

        x_t = x[:, time_idx, :]
        F_x = self.g_terminal(x_t)
        y_t = self.y_NNs_forward(time_idx, x_t, F_x)
        z_t = self.z_networks[time_idx](x_t)

        x_next = x[:, time_idx + 1, :]
        if time_idx < self.n_time_steps - 1:
            with torch.no_grad():
                F_x_next = self.g_terminal(x_next)
                y_next = self.y_NNs_forward(time_idx + 1, x_next, F_x_next)
        else:
            y_next = self.g_terminal(x_next)

        f_value = self.f_driver(time_idx, x_t, y_t, z_t)
        z_dw_term = torch.sum(z_t * dw[:, time_idx, :], dim=1, keepdim=True)
        temp_diff = y_next - y_t + f_value * self.dt - z_dw_term

        loss = torch.mean(temp_diff ** 2)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, n_iterations=20000, batch_size=64, print_every=1000, evaluate_every=500):
        losses = []
        y0_values = []

        start_time = time.time()
        last_y0 = None
        for it in tqdm(range(n_iterations)):
            x, dw = self.simulate_forward_process(batch_size)
            iteration_loss = 0.0
            for t in range(self.n_time_steps - 1, -1, -1):
                loss = self.train_step_backward(x, dw, t, self.optimizers[t])
                iteration_loss += loss

            losses.append(iteration_loss)

            if (it + 1) % evaluate_every == 0:
                with torch.no_grad():
                    g_0 = self.g_terminal(x[:, 0, :])
                    last_y0 = self.y_NNs_forward(0, x[:, 0, :], g_0).mean().item()
                y0_values.append(last_y0)

            if (it + 1) % print_every == 0:
                if last_y0 is None:
                    with torch.no_grad():
                        g_0 = self.g_terminal(x[:, 0, :])
                        last_y0 = self.y_NNs_forward(0, x[:, 0, :], g_0).mean().item()
                elapsed = time.time() - start_time
                print(
                    f"Iteration {it + 1}/{n_iterations}, Loss: {iteration_loss:.8f}, Y0: {last_y0:.8f}"
                )
                print(f"Elapsed time: {elapsed:.2f}s")

                if it > 0:
                    plt.figure(figsize=(15, 8))

                    plt.subplot(1, 2, 1)
                    plt.semilogy(range(len(losses)), losses)
                    plt.title("Loss")
                    plt.xlabel("Iterations")

                    plt.subplot(1, 2, 2)
                    plt.plot(range(len(y0_values)), y0_values)
                    plt.title("Y0 Value")
                    plt.xlabel("Iterations")

                    plt.tight_layout()
                    plt.show()

        return losses, y0_values


if __name__ == "__main__":
    d = 1
    total_time = 1.0
    n_time_steps = 50
    K = 10.0
    r = 0.06
    sigma = 0.4
    strike = 40.0
    x_0 = 40.0
    lambda_temp = 1
    epsilon = 0.2

    print(f"Lambda: {lambda_temp}")
    print(f"Epsilon: {epsilon}")
    torch.set_default_dtype(torch.float64)

    solver = DeepBackward(
        d=d,
        total_time=total_time,
        n_time_steps=n_time_steps,
        K=K,
        r=r,
        sigma=sigma,
        strike=strike,
        x_0=x_0,
        lambda_temp=lambda_temp,
        epsilon=epsilon,
        hidden_layers=2,
        hidden_dim=21 if d == 1 else 20 + d,
        lr=0.001,
    )

    losses, y0_values = solver.train(
        n_iterations=10000,
        batch_size=2**10,
        print_every=2000,
        evaluate_every=1000,
    )

    print(f"Lambda: {lambda_temp}")
    print(f"Epsilon: {epsilon}")

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(losses)), losses)
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")

    plt.subplot(1, 2, 2)
    plt.plot(range(len(y0_values)), y0_values)
    plt.title("Y0 Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Y0 Value")

    plt.tight_layout()

    if d == 1:
        filename = f"Backward_BSDE_put_eps_{epsilon}_lambda_{lambda_temp}_penalty_{K}.png"
    else:
        filename = (
            f"Backward_BSDE_put_highdim_d{d}_arithmetic_eps_{epsilon}_"
            f"lambda_{lambda_temp}_penalty_{K}.png"
        )

    plt.savefig(filename)
    plt.show()
