import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict
import imageio.v2 as imageio
import io
import os
from pde import PDEBase, CartesianGrid, ScalarField, MemoryStorage


class KdVPDE(PDEBase):
    """Custom KdV-type PDE with damping term."""

    def __init__(self, c=5, alpha=1.5, beta=0.1, damping=0.01, bc='periodic'):
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.damping = damping
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """Calculate the right-hand side of the PDE."""
        u = state

        # Calculate spatial derivatives
        u_x = u.gradient(self.bc)[0]
        u_xxx = u.gradient(self.bc)[0].gradient(self.bc)[0].gradient(self.bc)[0]

        # Calculate mean for damping term
        u_mean = u.average

        # KdV equation: du/dt = -c*du/dx - alpha*u*du/dx - beta*d3u/dx3 - damping*(u - mean(u))
        rhs = -self.c * u_x - self.alpha * u * u_x - self.beta * u_xxx - self.damping * (u - u_mean)

        return rhs


class Solver:
    def __init__(self, L=40, N=2000, T=20, c=5, alpha=1.5, beta=0.1,
                 bc_type='periodic', ic_type='sech'):
        self.L = L
        self.N = N
        self.T = T
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.bc_type = bc_type
        self.ic_type = ic_type

        # Create grid
        self.grid = CartesianGrid([[0, L]], [N], periodic=(bc_type == 'periodic'))
        self.x = self.grid.axes_coords[0]
        self.dx = L / N

        self.u = None
        self.t = None
        self.solve_time = None
        self.nfev = None
        self.storage = None

        self.u0 = self.get_initial_condition()

    # -------------------------------------------------------------------------
    def get_initial_condition(self):
        x0 = self.L / 4
        width = 2.0
        amplitude = 1.0

        if self.ic_type == 'sech':
            data = amplitude / np.cosh((self.x - x0) / width) ** 2
        elif self.ic_type == 'gaussian':
            data = amplitude * np.exp(-((self.x - x0) / width) ** 2)
        elif self.ic_type == 'sine':
            data = np.sin(2 * np.pi * self.x / self.L)
        elif self.ic_type == 'double_sech':
            x1, x2 = self.L / 4, 3 * self.L / 4
            data = (amplitude / np.cosh((self.x - x1) / width) ** 2 +
                    amplitude / np.cosh((self.x - x2) / width) ** 2)
        else:
            raise ValueError(f"Unknown initial condition type: {self.ic_type}")

        return ScalarField(self.grid, data)

    # -------------------------------------------------------------------------
    def solve(self, method='explicit', dt=0.01, tracker_interval=None):
        """
        Solve the PDE using py-pde.

        Args:
            method: 'explicit' or 'scipy' (for adaptive methods)
            dt: time step for explicit method, or max_step for scipy
            tracker_interval: interval for storing data (default: dt for explicit, 0.1 for scipy)
        """
        # Map boundary conditions
        if self.bc_type == 'periodic':
            bc = 'periodic'
        elif self.bc_type == 'dirichlet':
            bc = {'value': 0}
        elif self.bc_type == 'neumann':
            bc = {'derivative': 0}
        else:
            bc = 'auto-periodic-neumann'

        # Create the PDE
        eq = KdVPDE(c=self.c, alpha=self.alpha, beta=self.beta, damping=0.01, bc=bc)

        # Storage for the solution
        self.storage = MemoryStorage()

        # Set default tracker interval
        if tracker_interval is None:
            tracker_interval = dt if method == 'explicit' else 0.1

        # Solve
        start = time.time()
        # if method == 'explicit':
        result = eq.solve(
                self.u0,
                t_range=self.T,
                dt=dt,
                tracker=['progress', self.storage.tracker(tracker_interval)],
                solver=method
            )
        # else:  # scipy method
        #     result = eq.solve(
        #         self.u0,
        #         t_range=self.T,
        #         method=method,
        #         tracker=['progress', self.storage.tracker(tracker_interval)],
        #         solver='RK45',
        #         dt=dt,  # max_step in scipy
        #         rtol=1e-6,
        #         atol=1e-8
        #     )

        self.solve_time = time.time() - start

        # Extract data from storage
        self.t = np.array(self.storage.times)
        self.u = np.array([field.data for field in self.storage])

        # Store metadata
        self.last_method = method
        self.last_step = dt

        # Estimate number of function evaluations (approximate)
        if method == 'explicit':
            self.nfev = int(self.T / dt)
        else:
            self.nfev = len(self.t) * 6  # Rough estimate for RK45

        return self.u

    def get_stats(self) -> Dict:
        return {
            'method': getattr(self, 'last_method', 'Unknown'),
            'max_step': getattr(self, 'last_step', None),
            'solve_time': self.solve_time,
            'nfev': self.nfev,
            'N': self.N,
            'T': self.T,
            'alpha': self.alpha,
            'beta': self.beta,
            'bc_type': self.bc_type,
            'ic_type': self.ic_type,
            'max_u': np.max(np.abs(self.u)) if self.u is not None else None,
            'energy': np.trapz(self.u[-1, :] ** 2, self.x) if self.u is not None else None
        }

    # -------------------------------------------------------------------------
    def save_gif(self, filename="evolution.gif", every=5):
        """Save a GIF showing evolution of u(x,t)."""
        if self.u is None:
            raise ValueError("Run solve() before saving GIF.")

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        # Calculate safe y-limits
        u_min, u_max = np.nanmin(self.u), np.nanmax(self.u)
        if np.isfinite(u_min) and np.isfinite(u_max):
            margin = 0.1 * (u_max - u_min) if u_max != u_min else 0.1
            ylim = (u_min - margin, u_max + margin)
        else:
            ylim = (-1, 1)  # fallback

        images = []
        for i in range(0, len(self.t), every):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.x, self.u[i], lw=2)
            ax.set_ylim(ylim)
            ax.set_title(f"{self.last_method}, N={self.N}, dt={self.last_step:.4f}, t={self.t[i]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            images.append(imageio.imread(buf))
        imageio.mimsave(filename, images, fps=10)
        print(f"[save_gif] Saved evolution to {filename}")

    def extract_pde_observables(self, filename="pde_data.npz"):
        if self.u is None or self.t is None:
            raise ValueError("Run solver.solve() first.")

        u, t, x = self.u, self.t, self.x

        A = np.max(u, axis=1)
        du_dx = np.gradient(u, x, axis=1)
        B = np.mean(np.abs(du_dx), axis=1)
        d3u_dx3 = np.gradient(np.gradient(np.gradient(u, x, axis=1), x, axis=1), x, axis=1)
        C = np.mean(np.abs(d3u_dx3), axis=1)
        D = np.trapz(u ** 2, x, axis=1)

        np.savez(filename, t=t, A_pde=A, B_pde=B, C_pde=C, D_pde=D)
        print(f"[extract_pde_observables] Saved to '{filename}'")

        return {"t": t, "A_pde": A, "B_pde": B, "C_pde": C, "D_pde": D}


# Example usage
if __name__ == "__main__":
    solver = Solver(L=40, N=500, T=10, ic_type='sech')

    # Solve using explicit method
    solver.solve(method='scipy', dt=0.005)

    # Print statistics
    stats = solver.get_stats()
    print("\nSolver Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save GIF
    solver.save_gif("evolution.gif", every=10)

    # Extract observables
    observables = solver.extract_pde_observables("pde_data.npz")