import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class AmericanCallSolver:
    def __init__(self, S0, K, r, dividend, sigma, T, Nt=1000, Ny=4000):
        self.S0 = S0 # initial price
        self.K = K # strike price
        self.r = r # interest rate
        self.dividend = dividend # dividend rate
        self.sigma = sigma # volatility
        self.T = T # total time
        self.Nt = Nt # number of time grid points
        self.Ny = Ny # number of spatial grid points
        self.dt = T / Nt # width of time mesh
        self.penalty_K = 1e6 # penalty factor
        ##### spatial range
        y_center = np.log(K)
        y_range = 3.0
        y_min = y_center - y_range
        y_max = y_center + y_range
        self.y_grid = np.linspace(y_min, y_max, Ny)
        self.dy = self.y_grid[1] - self.y_grid[0]
        ##### PDE-FDM coefficients
        mu = (r - dividend) - 0.5 * sigma ** 2
        self.alpha = sigma ** 2 / (2 * self.dy ** 2) - mu / (2 * self.dy)
        self.beta = sigma ** 2 / (2 * self.dy ** 2) + mu / (2 * self.dy)
        ##### initialize solution
        self.u = np.zeros((Nt + 1, Ny))

    def payoff(self, y):
        S = np.exp(y) ##### we have used log-price (log spatial transformation) to simplify the equation
        return np.maximum(S - self.K, 0)

    def set_boundary(self, t_idx):
        tau = (self.Nt - t_idx) * self.dt
        ##### lower boundary
        self.u[t_idx, 0] = 0.0
        ##### upper boundary
        S_max = np.exp(self.y_grid[-1])
        if tau > 1e-10:
            self.u[t_idx, -1] = max(0.0, S_max * np.exp(-self.dividend * tau) - self.K * np.exp(-self.r * tau))
        else:
            self.u[t_idx, -1] = max(0, S_max - self.K)

    def solve(self):
        ##### terminal
        self.u[self.Nt, :] = self.payoff(self.y_grid)
        ##### starts solving
        for i in range(self.Nt - 1, -1, -1):
            self.set_boundary(i)
            u_old = self.u[i + 1, :].copy()
            ##### start fixed point
            for _ in range(50):
                payoff_vals = self.payoff(self.y_grid)
                indicator = (payoff_vals > u_old).astype(float)
                ##### linear equation coefficients
                gamma = 1.0 / self.dt + self.alpha + self.beta + self.r
                main_diag = gamma + self.penalty_K * indicator
                upper_diag = -self.beta * np.ones(self.Ny - 1)
                lower_diag = -self.alpha * np.ones(self.Ny - 1)
                ##### iteration-equation-rhs
                rhs = self.u[i + 1, :] / self.dt + self.penalty_K * payoff_vals * indicator
                rhs[1] += self.alpha * self.u[i, 0]
                rhs[-2] += self.beta * self.u[i, -1]
                ##### tridiagonal matrix
                A = diags(
                    [main_diag[1:-1], upper_diag[1:], lower_diag[:-1]],
                    [0, 1, -1],
                    format='csr'
                )
                ##### solve linear equation
                u_interior = spsolve(A, rhs[1:-1])
                ##### store solution
                u_new = np.zeros(self.Ny)
                u_new[0] = self.u[i, 0]
                u_new[1:-1] = u_interior
                u_new[-1] = self.u[i, -1]
                ##### check convergence
                if np.max(np.abs(u_new - u_old)) < 1e-6:
                    break
                u_old = u_new

            self.u[i, :] = u_old
        ##### interpolate
        y0 = np.log(self.S0)
        idx = np.searchsorted(self.y_grid, y0)
        if idx == 0:
            return self.u[0, 0]
        elif idx == self.Ny:
            return self.u[0, -1]
        else:
            y1, y2 = self.y_grid[idx - 1], self.y_grid[idx]
            u1, u2 = self.u[0, idx - 1], self.u[0, idx]
            weight = (y0 - y1) / (y2 - y1)
            return u1 * (1 - weight) + u2 * weight


if __name__ == "__main__":
    S0 = 40.0 # initial price
    K = 40.0 # strike price
    r = 0.05 # interest rate
    dividend = 0.05 # dividend rate
    sigma = 0.4 # volatility
    T = 0.5 # total time
    dividend_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    for dividend in dividend_list:
        solver = AmericanCallSolver(S0, K, r, dividend, sigma, T, Nt=1000, Ny=4000)
        price = solver.solve()
        print(f"Dividend Rate: {dividend:.2f}, Option Price: {price:.3f}")



