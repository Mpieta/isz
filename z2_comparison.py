import matplotlib.pyplot as plt
import pandas as pd
import os
from pde_solver import Solver


class ComparativeStudy:
    def __init__(self):
        self.results = []
        os.makedirs("gifs", exist_ok=True)

    # -------------------------------------------------------------------------
    def compare_methods(self, methods=None, **kwargs):
        """Compare different solver methods (explicit vs scipy)."""
        if methods is None:
            methods = [
                ('explicit', 0.005),
                ('implicit', 0.01),
                ('adams-bashforth', 0.05)
            ]

        solvers = []
        method_labels = []

        for method_config in methods:
            if isinstance(method_config, tuple):
                method, dt = method_config
                label = f"{method}_dt{dt}"
            else:
                method = method_config
                dt = 0.01
                label = method

            solver = Solver(**kwargs)
            solver.solve(method=method, dt=dt)
            stats = solver.get_stats()
            solvers.append(solver)
            method_labels.append(label)
            self.results.append(stats)

            gif_name = f"gifs/{label}_N{solver.N}.gif"
            solver.save_gif(gif_name, every=5)
            print(f"{label}: time={stats['solve_time']:.3f}s, evals={stats['nfev']}")

        self._plot_method_comparison(solvers, method_labels)
        return pd.DataFrame(self.results)

    # -------------------------------------------------------------------------
    def compare_time_steps(self, method='explicit', steps=(0.01, 0.005, 0.002, 0.001), **kwargs):
        """Compare different time step sizes."""
        for s in steps:
            solver = Solver(**kwargs)
            solver.solve(method=method, dt=s)
            stats = solver.get_stats()
            self.results.append(stats)

            gif_name = f"gifs/{method}_N{solver.N}_dt{s}.gif"
            solver.save_gif(gif_name, every=5)
            print(f"dt={s:.4f}: time={stats['solve_time']:.3f}s, evals={stats['nfev']}")

        self._plot_timing("max_step")

    # -------------------------------------------------------------------------
    def compare_resolutions(self, method='explicit', dt=0.005,
                            resolutions=(200, 400, 800, 1600), **kwargs):
        """Compare different spatial resolutions."""
        for n in resolutions:
            solver = Solver(N=n, **kwargs)
            solver.solve(method=method, dt=dt)
            stats = solver.get_stats()
            self.results.append(stats)

            gif_name = f"gifs/{method}_N{n}_dt{dt}.gif"
            solver.save_gif(gif_name, every=5)
            print(f"N={n}: time={stats['solve_time']:.3f}s, evals={stats['nfev']}")

        self._plot_timing("N")

    # -------------------------------------------------------------------------
    def compare_boundary_conditions(self, method='scipy', dt=0.01,
                                    bc_types=('periodic', 'dirichlet', 'neumann'),
                                    **kwargs):
        """Compare how the solution behaves under different boundary conditions."""
        solvers = []

        for bc in bc_types:
            solver = Solver(bc_type=bc, **kwargs)
            solver.solve(method=method, dt=dt)
            stats = solver.get_stats()
            self.results.append(stats)
            solvers.append(solver)

            gif_name = f"gifs/{method}_N{solver.N}_L{solver.L}_{bc}.gif"
            solver.save_gif(gif_name, every=5)
            print(f"[BC={bc}] time={stats['solve_time']:.3f}s, evals={stats['nfev']}")

        self._plot_bc_comparison(solvers, bc_types)
        return pd.DataFrame(self.results)

    # -------------------------------------------------------------------------
    def compare_initial_conditions(self, method='explicit', dt=0.005,
                                   ic_types=('sech', 'gaussian', 'sine', 'double_sech'),
                                   **kwargs):
        """Compare different initial conditions."""
        solvers = []

        for ic in ic_types:
            solver = Solver(ic_type=ic, **kwargs)
            solver.solve(method=method, dt=dt)
            stats = solver.get_stats()
            self.results.append(stats)
            solvers.append(solver)

            gif_name = f"gifs/{method}_N{solver.N}_{ic}.gif"
            solver.save_gif(gif_name, every=5)
            print(f"[IC={ic}] time={stats['solve_time']:.3f}s, max_u={stats['max_u']:.3f}")

        self._plot_ic_comparison(solvers, ic_types)
        return pd.DataFrame(self.results)

    # -------------------------------------------------------------------------
    def _plot_method_comparison(self, solvers, method_labels):
        """Plot solution evolution for different methods."""
        fig, axs = plt.subplots(1, len(solvers), figsize=(5 * len(solvers), 5))
        if len(solvers) == 1:
            axs = [axs]

        for ax, solver, label in zip(axs, solvers, method_labels):
            Nt = len(solver.t)
            for idx in [0, Nt // 3, 2 * Nt // 3, Nt - 1]:
                ax.plot(solver.x, solver.u[idx], label=f"t={solver.t[idx]:.1f}")
            ax.set_title(f"{label}\n({solver.solve_time:.2f}s)")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        fig.savefig("method_comparison.png", dpi=150)
        print("[Saved method_comparison.png]")

    # -------------------------------------------------------------------------
    def _plot_bc_comparison(self, solvers, bc_types):
        """Plot final profiles for different boundary conditions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Initial conditions
        for solver, bc in zip(solvers, bc_types):
            ax1.plot(solver.x, solver.u[0], label=f"{bc}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,0)")
        ax1.set_title("Initial conditions")
        ax1.legend()
        ax1.grid(True)

        # Final states
        for solver, bc in zip(solvers, bc_types):
            ax2.plot(solver.x, solver.u[-1], label=f"{bc}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("u(x,T)")
        ax2.set_title("Final states under different BCs")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("boundary_condition_comparison.png", dpi=150)
        print("[Saved boundary_condition_comparison.png]")

    # -------------------------------------------------------------------------
    def _plot_ic_comparison(self, solvers, ic_types):
        """Plot initial and final profiles for different initial conditions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Initial conditions
        for solver, ic in zip(solvers, ic_types):
            ax1.plot(solver.x, solver.u[0], label=f"{ic}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,0)")
        ax1.set_title("Different initial conditions")
        ax1.legend()
        ax1.grid(True)

        # Final states
        for solver, ic in zip(solvers, ic_types):
            ax2.plot(solver.x, solver.u[-1], label=f"{ic}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("u(x,T)")
        ax2.set_title("Evolution from different ICs")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("initial_condition_comparison.png", dpi=150)
        print("[Saved initial_condition_comparison.png]")

    # -------------------------------------------------------------------------
    def _plot_timing(self, param):
        """Plot timing results vs parameter."""
        df = pd.DataFrame(self.results)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Solve time
        ax1.plot(df[param], df["solve_time"], "o-", lw=2, markersize=8)
        ax1.set_xlabel(param)
        ax1.set_ylabel("solve_time [s]")
        ax1.set_title(f"Solver timing vs {param}")
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(f"timing_vs_{param}.png", dpi=150)
        print(f"[Saved timing_vs_{param}.png]")


def run_comprehensive_study():
    """Run a comprehensive comparison study."""
    study = ComparativeStudy()

    # print("\n### 1. Comparing Methods (explicit vs scipy with different dt) ###")
    # df1 = study.compare_methods(
    #     methods=[
    #         ('explicit', 0.001),
    #         ('scipy', 0.001),
    #         ('crank-nicolson', 0.001)
    #     ],
    #     L=40, N=512, T=20
    # )
    # df1.to_csv("method_comparison.csv", index=False)
    #
    # print("\n### 2. Comparing Time Steps (explicit method) ###")
    # for m in ['explicit', 'scipy', 'crank-nicolson']:
    #     study.compare_time_steps(
    #         method=m,#'scipy',
    #         steps=[0.01, 0.005, 0.002, 0.001],
    #         L=40, N=1024, T=20
    #     )
    #
    #     print("\n### 3. Comparing Resolutions ###")
    #     study.compare_resolutions(
    #         method=m,#'scipy',
    #         dt=0.005,
    #         resolutions=[128, 256, 512],
    #         L=40, T=20
    #     )
    #
    # print("\n### 4. Comparing Boundary Conditions ###")
    # study.compare_boundary_conditions(
    #     method='scipy',
    #     dt=0.1,
    #     bc_types=['periodic', 'neumann'],
    #     L=40, N=256, T=20
    # )
    #
    # print("\n### 5. Comparing Initial Conditions ###")
    # study.compare_initial_conditions(
    #     method='scipy',
    #     dt=0.005,
    #     ic_types=['sech', 'gaussian', 'sine', 'double_sech'],
    #     L=40, N=1024, T=10
    # )

    print("\n### 6. Final high-resolution run for observables ###")
    solver = Solver(T=20, L=40, N=1024, c=2.6, alpha=2.5, beta=0.1)
    solver.solve(method='scipy', dt=0.01)
    solver.extract_pde_observables(filename="pde_data.npz")
    solver.save_gif("evolution_final.gif", every=4)

    # Save all results combined
    pd.DataFrame(study.results).to_csv("full_comparative_study.csv", index=False)
    print("\n✅ Saved all comparison data to full_comparative_study.csv")
    print(f"✅ Total runs: {len(study.results)}")


if __name__ == "__main__":
    run_comprehensive_study()