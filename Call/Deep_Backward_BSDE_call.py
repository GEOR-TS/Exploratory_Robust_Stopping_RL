import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class DeepBackward(nn.Module):
    def __init__(self, d, total_time, n_time_steps, K, r, dividend, sigma, strike, x_0, lambda_temp, epsilon,
                 hidden_layers=3, hidden_dim=64, lr=0.01, device=torch.device('cpu')):
        super(DeepBackward, self).__init__()
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

        self.y_networks = nn.ModuleList([
            self._build_network(d+1, 1) for _ in range(n_time_steps)
        ])
        self.z_networks = nn.ModuleList([
            self._build_network(d, d) for _ in range(n_time_steps)
        ])

        self._init_weights()
        self.to(self.device)
        self.optimizers = [
            optim.Adam(list(self.y_networks[t].parameters()) +
                       list(self.z_networks[t].parameters()), lr=lr)
            for t in range(n_time_steps)
        ]

    def _init_weights(self):
        for network in self.z_networks:
            for name, module in network.named_children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
        for network in self.y_networks:
            for name, module in network.named_children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0.0)

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

    def y_NNs_forward(self, l, x, payoff):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if payoff.dim() == 1:
            payoff = payoff.unsqueeze(1)

        inputs = torch.cat((x, payoff), dim=1)
        return self.y_networks[l].forward(inputs) + payoff

    def g_terminal(self, x):
        return torch.relu(x - self.strike)

    def small_g(self, z):
        epsilon = self.epsilon
        z_norm = torch.norm(z, dim=1)
        z_reg_term =  (-1) * epsilon * z_norm
        return z_reg_term

    def f_driver(self, t, x, y, z):
        f_x = torch.zeros_like(y)
        discount_term = self.r * y

        F_x = self.g_terminal(x)
        control_term = self.K * (F_x - y)

        exponent = -(self.K / self.lambda_temp) * (F_x - y)
        exp_0 = torch.zeros_like(y)
        log_1_exp = torch.logaddexp(exp_0, exponent)

        non_linear_x_y = control_term + log_1_exp
        ## small_g of z
        z_reg_term = self.small_g(z).unsqueeze(1)

        return f_x - discount_term + z_reg_term + non_linear_x_y

    def simulate_forward_process(self, batch_size):
        x = torch.zeros(batch_size, self.n_time_steps + 1, self.d, device=self.device)
        x[:, 0, :] = self.x_0
        dw = torch.zeros(batch_size, self.n_time_steps, self.d, device=self.device)
        for t in range(self.n_time_steps):
            dw[:, t, :] = np.sqrt(self.dt) * torch.randn(batch_size, self.d, device=self.device)
            x[:, t + 1, :] = x[:, t, :] * torch.exp(
                (self.r - self.dividend - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dw[:, t, :]
            )
        return x, dw

    def train_step_backward(self, x, dw, time_idx, optimizer):
        optimizer.zero_grad()

        # Current state
        x_t = x[:, time_idx, :]
        F_x = self.g_terminal(x_t)
        y_t = self.y_NNs_forward(time_idx, x_t, F_x)
        z_t = self.z_networks[time_idx](x_t)

        # Next state
        x_next = x[:, time_idx + 1, :]
        if time_idx < self.n_time_steps - 1:
            with torch.no_grad():
                F_x_next = self.g_terminal(x_next)
                y_next = self.y_NNs_forward(time_idx + 1, x_next, F_x_next)
        else:
            y_next = self.g_terminal(x_next)

        # Compute driver term
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
        for it in tqdm(range(n_iterations)):
            x, dw = self.simulate_forward_process(batch_size)
            iteration_loss = 0.0
            for t in range(self.n_time_steps - 1, -1, -1):  # Backward from N-1 to 0
                loss = self.train_step_backward(x, dw, t, self.optimizers[t])
                iteration_loss += loss

            losses.append(iteration_loss)
            if (it + 1) % evaluate_every == 0:
                # Compute Y0 value
                with torch.no_grad():
                    g_0 = self.g_terminal(x[:, 0, :])
                    y0 = self.y_NNs_forward(0, x[:, 0, :], g_0).mean().item()
                y0_values.append(y0)

            if (it + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Iteration {it + 1}/{n_iterations}, iteration Loss: {iteration_loss:.8f}, Y0: {y0:.8f}")
                print(f"Elapsed time: {elapsed:.2f}s")

                if it > 0:
                    plt.figure(figsize=(15, 8))

                    plt.subplot(1, 2, 1)
                    plt.semilogy(range(len(losses)), losses)
                    plt.title('Loss')
                    plt.xlabel('Iterations')

                    plt.subplot(1, 2, 2)
                    plt.plot(range(len(y0_values)), y0_values)
                    plt.title('Y0 Value')
                    plt.xlabel('Iterations')

                    plt.tight_layout()
                    plt.show()

        return losses, y0_values

if __name__ == "__main__":
    # Parameters
    d = 1
    total_time = 0.5
    n_time_steps = 100
    K = 10.0
    r = 0.05
    dividend = 0.05
    sigma = 0.4
    strike = 40.0
    x_0 = 40.0
    lambda_temp = 1
    epsilon = 0.4 ## can adjust to different ambiguity degree

    # Use double precision
    torch.set_default_dtype(torch.float64)

    solver = DeepBackward(
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
        hidden_layers=2,
        hidden_dim=21,
        lr=0.001,
    )

    # Train the model
    losses, y0_values = solver.train(
        n_iterations=10000,
        batch_size=2 ** 10,
        print_every=2000,
        evaluate_every=1000,
    )

    print('Lamda is :', lambda_temp)
    print('Epsilon is :', epsilon)

    # Plot final results
    plt.figure(figsize=(15, 8))

    # Plot loss history
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(losses)), losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')

    # Plot Y0 convergence
    plt.subplot(1, 2, 2)
    plt.plot(range(len(y0_values)), y0_values)
    plt.title('Y0 Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Y0 Value')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"Backward_BSDE_call_eps_{epsilon}_lambda_{lambda_temp}_penalty_{K}.png")
    plt.show()

























