import math

from Implicit_FDM_call import AmericanCallSolver
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np


def compute_1d_call_reference_values(dividend_list, x_0, strike, r, sigma, total_time, Nt=1000, Ny=4000):
    rows = []
    for dividend_true in dividend_list:
        solver = AmericanCallSolver(
            S0=x_0,
            K=strike,
            r=r,
            dividend=dividend_true,
            sigma=sigma,
            T=total_time,
            Nt=Nt,
            Ny=Ny,
        )
        rows.append(
            {
                "dividend_true": dividend_true,
                "reference_value": solver.solve(),
            }
        )
    return rows


def effective_gbm_geometric_basket_params(d, dividend, sigma, x_0):
    sigma_eff = float(sigma) / math.sqrt(d)
    dividend_eff = float(dividend) + 0.5 * float(sigma) ** 2 * (1.0 - 1.0 / d)
    return {
        "x0_eff": float(x_0),
        "sigma_eff": sigma_eff,
        "dividend_eff": dividend_eff,
    }


class AmericanReducedGBMCallSolver:
    def __init__(
        self,
        S0,
        K,
        r,
        dividend,
        sigma,
        T,
        Nt=1000,
        Ny=4000,
        penalty_K=1e8,
        y_range=3.0,
        fp_iterations=50,
        tol=1e-6,
    ):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.dividend = float(dividend)
        self.sigma = float(sigma)
        self.T = float(T)
        self.Nt = int(Nt)
        self.Ny = int(Ny)
        self.dt = self.T / self.Nt
        self.penalty_K = float(penalty_K)
        self.fp_iterations = int(fp_iterations)
        self.tol = float(tol)

        y_center = math.log(max(self.K, self.S0, 1e-10))
        self.y_grid = np.linspace(y_center - y_range, y_center + y_range, self.Ny)
        self.dy = self.y_grid[1] - self.y_grid[0]

        mu = (self.r - self.dividend) - 0.5 * self.sigma**2
        self.alpha = self.sigma**2 / (2 * self.dy**2) - mu / (2 * self.dy)
        self.beta = self.sigma**2 / (2 * self.dy**2) + mu / (2 * self.dy)
        self.u = np.zeros((self.Nt + 1, self.Ny))

    def payoff(self, y):
        s = np.exp(y)
        return np.maximum(s - self.K, 0.0)

    def set_boundary(self, t_idx):
        tau = (self.Nt - t_idx) * self.dt
        self.u[t_idx, 0] = 0.0
        s_max = np.exp(self.y_grid[-1])
        if tau > 1e-10:
            hold_approx = s_max * np.exp(-self.dividend * tau) - self.K * np.exp(-self.r * tau)
            self.u[t_idx, -1] = max(0.0, hold_approx)
        else:
            self.u[t_idx, -1] = max(0.0, s_max - self.K)

    def solve(self):
        self.u[self.Nt, :] = self.payoff(self.y_grid)
        payoff_vals = self.payoff(self.y_grid)

        for i in range(self.Nt - 1, -1, -1):
            self.set_boundary(i)
            u_old = self.u[i + 1, :].copy()
            gamma = 1.0 / self.dt + self.alpha + self.beta + self.r

            for _ in range(self.fp_iterations):
                indicator = (payoff_vals > u_old).astype(float)
                main_diag = gamma + self.penalty_K * indicator
                upper_diag = -self.beta * np.ones(self.Ny - 1)
                lower_diag = -self.alpha * np.ones(self.Ny - 1)

                rhs = self.u[i + 1, :] / self.dt + self.penalty_K * payoff_vals * indicator
                rhs[1] += self.alpha * self.u[i, 0]
                rhs[-2] += self.beta * self.u[i, -1]

                A = diags(
                    [main_diag[1:-1], upper_diag[1:], lower_diag[:-1]],
                    [0, 1, -1],
                    format="csr",
                )
                u_interior = spsolve(A, rhs[1:-1])

                u_new = np.zeros(self.Ny)
                u_new[0] = self.u[i, 0]
                u_new[1:-1] = u_interior
                u_new[-1] = self.u[i, -1]

                if np.max(np.abs(u_new - u_old)) < self.tol:
                    u_old = u_new
                    break
                u_old = u_new

            self.u[i, :] = u_old

        y0 = math.log(self.S0)
        idx = np.searchsorted(self.y_grid, y0)
        if idx == 0:
            return self.u[0, 0]
        if idx == self.Ny:
            return self.u[0, -1]

        y1, y2 = self.y_grid[idx - 1], self.y_grid[idx]
        u1, u2 = self.u[0, idx - 1], self.u[0, idx]
        weight = (y0 - y1) / (y2 - y1)
        return u1 * (1 - weight) + u2 * weight


def benchmark_gbm_geometric_call_fdm(
    d,
    x_0,
    strike,
    r,
    dividend,
    sigma,
    total_time,
    Nt=1000,
    Ny=4000,
    penalty_K=1e8,
    y_range=3.0,
    fp_iterations=50,
    tol=1e-6,
):
    params = effective_gbm_geometric_basket_params(
        d=d,
        dividend=dividend,
        sigma=sigma,
        x_0=x_0,
    )
    solver = AmericanReducedGBMCallSolver(
        S0=params["x0_eff"],
        K=strike,
        r=r,
        dividend=params["dividend_eff"],
        sigma=params["sigma_eff"],
        T=total_time,
        Nt=Nt,
        Ny=Ny,
        penalty_K=penalty_K,
        y_range=y_range,
        fp_iterations=fp_iterations,
        tol=tol,
    )
    return solver.solve(), params
