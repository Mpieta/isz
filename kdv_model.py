import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve


class Solver:
    def __init__(self, L=40, N=200, T=10, dt=0.005, c=0.5, alpha=0.3, beta=0.01):
        """
        Parameters:
        L: Domain length
        N: Number of points
        T: Final time
        dt: Time step
        c: speed factor
        alpha: Nonlinear coefficient
        beta: Dispersion coefficient
        """
        self.L = L
        self.N = N
        self.T = T
        self.dt = dt
        self.c = c
        self.alpha = alpha
        self.beta = beta

        # Spatial grid
        self.x = np.linspace(0, L, N)
        self.dx = L / (N - 1)

        # Time grid
        self.t = np.arange(0, T, dt)
        self.Nt = len(self.t)

        # Initialize solution
        self.u = np.zeros((self.Nt, N))
        self.set_initial_condition()

        cfl = (abs(c) + np.max(np.abs(self.u[0, :])) * abs(alpha)) * dt / self.dx
        print(f"CFL number: {cfl:.4f} (should be < 1 for stability)")

    def set_initial_condition(self):
        x0 = self.L / 4
        width = 2.0
        amplitude = 1.0
        self.u[0, :] = amplitude / np.cosh((self.x - x0) / width) ** 2

    def compute_derivatives(self, u):
        dx = self.dx
        N = self.N

        du_dx = np.zeros(N)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        du_dx[0] = (u[1] - u[-1]) / (2 * dx)
        du_dx[-1] = (u[0] - u[-2]) / (2 * dx)

        d3u_dx3 = np.zeros(N)

        for i in range(2, N - 2):
            d3u_dx3[i] = (u[i + 2] - 2 * u[i + 1] + 2 * u[i - 1] - u[i - 2]) / (2 * dx ** 3)

        d3u_dx3[0] = (u[2] - 2 * u[1] + 2 * u[-1] - u[-2]) / (2 * dx ** 3)
        d3u_dx3[1] = (u[3] - 2 * u[2] + 2 * u[0] - u[-1]) / (2 * dx ** 3)
        d3u_dx3[-2] = (u[0] - 2 * u[-1] + 2 * u[-3] - u[-4]) / (2 * dx ** 3)
        d3u_dx3[-1] = (u[1] - 2 * u[0] + 2 * u[-2] - u[-3]) / (2 * dx ** 3)

        return du_dx, d3u_dx3

    def solve(self):
        for n in range(self.Nt - 1):
            u_n = self.u[n, :]

            if np.any(np.isnan(u_n)) or np.any(np.isinf(u_n)):
                self.u[n + 1:, :] = self.u[n, :]
                break

            du_dx, d3u_dx3 = self.compute_derivatives(u_n)

            # -c*du/dx - alpha*u*du/dx - beta*d³u/dx³
            rhs = -self.c * du_dx - self.alpha * u_n * du_dx - self.beta * d3u_dx3

            damping = 0.01
            self.u[n + 1, :] = u_n + self.dt * rhs - damping * self.dt * (u_n - np.mean(u_n))


        return self.u

    def animate(self, skip=5):
        u_min = np.nanmin(self.u)
        u_max = np.nanmax(self.u)

        if np.isnan(u_min) or np.isnan(u_max) or np.isinf(u_min) or np.isinf(u_max):
            u_min, u_max = -2, 2

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(self.x, self.u[0, :], 'b-', linewidth=2)

        ax.set_xlim(0, self.L)
        ax.set_ylim(u_min - 0.2, u_max + 0.2)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f'Solution: du/dt + {self.c}*du/dx + {self.alpha}*u*du/dx + {self.beta}*d³u/dx³ = 0', fontsize=12)
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        def update(frame):
            idx = frame * skip
            if idx >= self.Nt:
                idx = self.Nt - 1
            line.set_ydata(self.u[idx, :])
            time_text.set_text(f't = {self.t[idx]:.2f}')
            return line, time_text

        anim = FuncAnimation(fig, update, frames=self.Nt // skip,
                             interval=50, blit=True, repeat=True)
        plt.tight_layout()
        return fig, anim

    def plot_snapshots(self):
        u_min = np.nanmin(self.u)
        u_max = np.nanmax(self.u)

        if np.isnan(u_min) or np.isnan(u_max) or np.isinf(u_min) or np.isinf(u_max):
            u_min, u_max = -2, 2

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        times_idx = [0, self.Nt // 4, self.Nt // 2, self.Nt - 1]

        for ax, idx in zip(axes.flat, times_idx):
            u_plot = self.u[idx, :]
            u_plot = np.where(np.isfinite(u_plot), u_plot, 0)

            ax.plot(self.x, u_plot, 'b-', linewidth=2)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('u(x,t)', fontsize=11)
            ax.set_title(f't = {self.t[idx]:.2f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(u_min - 0.2, u_max + 0.2)

        plt.suptitle(f'PDE Solution (c={self.c}, α={self.alpha}, β={self.beta})',
                     fontsize=14, y=1.00)
        plt.tight_layout()
        return fig


if __name__ == "__main__":

    solver = Solver(
        L=80,
        N=300,
        T=15,
        dt=0.005,
        c=1.5,
        alpha=0.75,
        beta=0.01
    )

    u_solution = solver.solve()

    fig_anim, anim = solver.animate(skip=10)

    print("\nSimulation parameters:")
    print(f"Domain: [0, {solver.L}]")
    print(f"Grid points: {solver.N}")
    print(f"Spatial step dx: {solver.dx:.4f}")
    print(f"Time steps: {solver.Nt}")
    print(f"Time step dt: {solver.dt:.4f}")
    print(f"c = {solver.c}, alpha = {solver.alpha}, beta = {solver.beta}")

    plt.show()