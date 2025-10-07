import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import copy


class PE_put(nn.Module):
    def __init__(self, d, total_time, n_time_steps, K, r, sigma, strike, x_0, lambda_temp, epsilon,
                 hidden_layers=3, hidden_dim=64, lr=0.01, device=torch.device('cpu')):
        super(PE_put, self).__init__()
        self.d = d  # dimension
        self.total_time = total_time
        self.n_time_steps = n_time_steps # number of time grid points
        self.dt = total_time / n_time_steps # time mesh width
        self.K = K # penalty factor
        self.r = r # interest rate
        self.sigma = sigma # volatility
        self.strike = strike # strike price
        self.x_0 = x_0 # initial price
        self.lambda_temp = lambda_temp # lambda temperature
        self.epsilon = epsilon # ambiguity degree

        self.hidden_layers = hidden_layers # hidden layer numbers
        self.hidden_dim = hidden_dim # network width

        self.device = device
        print(f"Using device: {self.device}")

        self.y_networks = nn.ModuleList([
            self._build_network(d + 1, 1) for _ in range(n_time_steps)
        ])

        self._init_weights()
        self.to(self.device)

        self.optimizers = [
            optim.Adam(list(self.y_networks[t].parameters()), lr=lr)
            for t in range(n_time_steps)
        ]

        self.pi_function = None

    def _init_weights(self):
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

    def compute_y_grad_y(self, l, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_requires_grad = x.requires_grad
        if not x_requires_grad:
            x = x.clone().detach().requires_grad_(True)

        y = self.y_NNs_forward(l, x)
        # Compute gradient
        grad_y = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
        )[0]

        return y.detach(), grad_y.detach()

    def g_terminal(self, x):
        return torch.relu(self.strike - x)

    def small_g(self, z):
        epsilon = self.epsilon
        z_norm = torch.norm(z, dim=1)
        z_reg_term = (-1) * epsilon * z_norm
        return z_reg_term

    def f_driver(self, t, x, y, z, pi):
        f_x = torch.zeros_like(y)
        discount_term = self.r * y

        F_x = self.g_terminal(x)
        control_term = self.K * (F_x - y) * pi
        ## small_g of z
        z_reg_term = self.small_g(z).unsqueeze(1)

        # Entropy
        eps = 1e-10
        pi_safe = torch.clamp(pi, min=eps)
        pi_1_safe = torch.clamp(1.0 - pi, min=eps)
        entropy = pi * torch.log(pi_safe) + (1.0 - pi) * torch.log(pi_1_safe)
        entropy_term = self.lambda_temp * entropy

        H = control_term - entropy_term
        return f_x - discount_term + z_reg_term + H

    def simulate_forward_process(self, batch_size):
        x = torch.zeros(batch_size, self.n_time_steps + 1, self.d, device=self.device)
        x[:, 0, :] = self.x_0
        dw = torch.zeros(batch_size, self.n_time_steps, self.d, device=self.device)
        for t in range(self.n_time_steps):
            dw[:, t, :] = np.sqrt(self.dt) * torch.randn(batch_size, self.d, device=self.device)
            x[:, t + 1, :] = x[:, t, :] * torch.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dw[:, t, :]
            )
        return x

    def train_step_backward(self, x, time_idx, optimizer):
        optimizer.zero_grad()
        # Current state
        x_t = x[:, time_idx, :]
        y_t = self.y_NNs_forward(time_idx, x_t)

        # Next state
        x_next = x[:, time_idx + 1, :]

        dx_t = x_next - x_t
        quadratic_variation_x_abs = torch.abs(dx_t)
        multiplier = (1 / (self.dt ** (0.5))) * quadratic_variation_x_abs

        y_next, grad_y_next = self.compute_y_grad_y(time_idx + 1, x_next)
        z_next = multiplier * grad_y_next
        pi_next = self.pi_function(time_idx+1, x_next)

        # Compute driver term
        f_value = self.f_driver(time_idx, x_next, y_next, z_next , pi_next)

        temp_diff = y_next - y_t + f_value * self.dt
        loss = torch.mean(temp_diff ** 2)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, n_iterations=20000, batch_size=64):
        losses = []
        y0_values = []
        for it in tqdm(range(n_iterations)):
            x = self.simulate_forward_process(batch_size)
            iteration_loss = 0.0
            for t in range(self.n_time_steps - 1, -1, -1):  # Backward from I-1 to 0
                loss = self.train_step_backward(x, t, self.optimizers[t])
                iteration_loss += loss

            losses.append(iteration_loss)
            # Compute Y0 value
            with torch.no_grad():
                y0 = self.y_NNs_forward(0, x[:, 0, :]).mean().item()
            y0_values.append(y0)

        return losses, y0_values


class Policy_Iteration:
    def __init__(self, d, total_time, n_time_steps, K, r, sigma, strike, x_0, lambda_temp, epsilon, device=torch.device("cpu"),
                 hidden_layers=3, hidden_dim=64, lr=0.01):
        super(Policy_Iteration, self).__init__()
        self.d = d  # dimension
        self.total_time = total_time
        self.n_time_steps = n_time_steps  # number of time grid points
        self.dt = total_time / n_time_steps  # time mesh width
        self.K = K  # penalty factor
        self.r = r  # interest rate
        self.sigma = sigma  # volatility
        self.strike = strike  # strike price
        self.x_0 = x_0  # initial price
        self.lambda_temp = lambda_temp  # lambda temperature
        self.epsilon = epsilon  # ambiguity degree

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.policy_evaluation_NN_solver = PE_put(
                            d=d,
                            total_time=total_time,
                            n_time_steps=n_time_steps,
                            K=K,
                            r=r,
                            sigma=sigma,
                            strike=strike,
                            x_0=x_0,
                            lambda_temp=lambda_temp,
                            epsilon = epsilon,
                            hidden_layers=hidden_layers,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            device=device,
                        )

        # Storage for historical results
        self.pi_history = {
            'networks': [],
            'losses': [],
            'y0_values': [],
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


    def y_NNs_forward(self, y_networks, n, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        payoff = self.g_terminal(x)
        if payoff.dim() == 1:
            payoff = payoff.unsqueeze(1)

        if n == self.n_time_steps:
            return payoff
        else:
            inputs = torch.cat((x, payoff), dim=1)
            return y_networks[n].forward(inputs) + payoff

    def g_terminal(self, x):
        return torch.relu(self.strike - x)

    def PolicyIteration(self, PI_iteration=2, PE_iteration = 1000, batch_size=2**10):
        print('Lamda is :', self.lambda_temp)
        print('Epsilon is :', self.epsilon)

        self.PI_iteration = PI_iteration
        start_time = time.time()
        for p in range(PI_iteration):
            print(f"---------------------- PI Step {p + 1} -----------------------")
            ######################## Setup and fix the policy form at this PI step
            y_networks = self.freeze_network(self.policy_evaluation_NN_solver.y_networks)
            self.pi_history['networks'].append(y_networks)

            def pi_function(n, x):
                F_x = self.g_terminal(x)
                y_t = self.y_NNs_forward(y_networks, n, x)
                Y_minus_F = y_t - F_x
                return torch.sigmoid(-(self.K / self.lambda_temp) * Y_minus_F)

            self.policy_evaluation_NN_solver.pi_function = pi_function

            ######################## Train the model, Policy Evaluation
            losses, y0_values = self.policy_evaluation_NN_solver.train(
                n_iterations=PE_iteration,
                batch_size=batch_size,
            )

            ######################## Store results
            self.pi_history['losses'].append(losses[-1])
            self.pi_history['y0_values'].append(y0_values[-1])
            print('Y0 :', y0_values[-1])

        ############### Compute total time
        print_time = time.time() - start_time
        print('total time use of PI', print_time)




