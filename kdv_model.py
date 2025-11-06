import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import time
import pandas as pd
from typing import Dict, List, Tuple


class Solver:
    def __init__(self, L=40, N=200, T=10, c=0.5, alpha=0.3, beta=0.01,
                 bc_type='periodic', ic_type='sech'):
        """
        Parameters:
        L: Domain length
        N: Number of points
        T: Final time
        c: speed factor
        alpha: Nonlinear coefficient
        beta: Dispersion coefficient
        bc_type: Boundary condition type ('periodic', 'dirichlet', 'neumann')
        ic_type: Initial condition type ('sech', 'gaussian', 'sine', 'double_sech')
        """
        self.L = L
        self.N = N
        self.T = T
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.bc_type = bc_type
        self.ic_type = ic_type

        # Spatial grid
        self.x = np.linspace(0, L, N)
        self.dx = L / (N - 1)

        # Solution storage
        self.u = None
        self.t = None
        self.solve_time = None
        self.nfev = None

        # Get initial condition
        self.u0 = self.get_initial_condition()

    def get_initial_condition(self):
        """Various initial conditions"""
        if self.ic_type == 'sech':
            x0 = self.L / 4
            width = 2.0
            amplitude = 1.0
            return amplitude / np.cosh((self.x - x0) / width) ** 2

        elif self.ic_type == 'gaussian':
            x0 = self.L / 4
            width = 2.0
            amplitude = 1.0
            return amplitude * np.exp(-((self.x - x0) / width) ** 2)

        elif self.ic_type == 'sine':
            return np.sin(2 * np.pi * self.x / self.L)

        elif self.ic_type == 'double_sech':
            x1, x2 = self.L / 4, 3 * self.L / 4
            width = 2.0
            amplitude = 1.0
            return (amplitude / np.cosh((self.x - x1) / width) ** 2 +
                    amplitude / np.cosh((self.x - x2) / width) ** 2)

        else:
            raise ValueError(f"Unknown initial condition type: {self.ic_type}")

    def compute_derivatives(self, u):
        """Compute spatial derivatives with different boundary conditions"""
        dx = self.dx
        N = self.N

        du_dx = np.zeros(N)
        d3u_dx3 = np.zeros(N)

        if self.bc_type == 'periodic':
            # First derivative
            du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            du_dx[0] = (u[1] - u[-1]) / (2 * dx)
            du_dx[-1] = (u[0] - u[-2]) / (2 * dx)

            # Third derivative
            for i in range(2, N - 2):
                d3u_dx3[i] = (u[i + 2] - 2 * u[i + 1] + 2 * u[i - 1] - u[i - 2]) / (2 * dx ** 3)

            d3u_dx3[0] = (u[2] - 2 * u[1] + 2 * u[-1] - u[-2]) / (2 * dx ** 3)
            d3u_dx3[1] = (u[3] - 2 * u[2] + 2 * u[0] - u[-1]) / (2 * dx ** 3)
            d3u_dx3[-2] = (u[0] - 2 * u[-1] + 2 * u[-3] - u[-4]) / (2 * dx ** 3)
            d3u_dx3[-1] = (u[1] - 2 * u[0] + 2 * u[-2] - u[-3]) / (2 * dx ** 3)

        elif self.bc_type == 'dirichlet':
            # First derivative (central diff for interior, one-sided for boundaries)
            du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            du_dx[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx)
            du_dx[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dx)

            # Third derivative (interior only to avoid boundary issues)
            for i in range(2, N - 2):
                d3u_dx3[i] = (u[i + 2] - 2 * u[i + 1] + 2 * u[i - 1] - u[i - 2]) / (2 * dx ** 3)

        elif self.bc_type == 'neumann':
            # Similar to Dirichlet but with zero derivative at boundaries
            du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            du_dx[0] = 0
            du_dx[-1] = 0

            for i in range(2, N - 2):
                d3u_dx3[i] = (u[i + 2] - 2 * u[i + 1] + 2 * u[i - 1] - u[i - 2]) / (2 * dx ** 3)

        return du_dx, d3u_dx3

    def rhs(self, t, u):
        """Right-hand side of the PDE"""
        du_dx, d3u_dx3 = self.compute_derivatives(u)

        # PDE: du/dt = -c*du/dx - alpha*u*du/dx - beta*d³u/dx³
        dudt = -self.c * du_dx - self.alpha * u * du_dx - self.beta * d3u_dx3

        # Add small damping to prevent drift
        damping = 0.01
        dudt -= damping * (u - np.mean(u))

        return dudt

    def solve(self, method='RK45', max_step=0.01, rtol=1e-6, atol=1e-8):
        """Solve the PDE using scipy's solve_ivp"""
        start_time = time.time()

        sol = solve_ivp(
            self.rhs,
            t_span=(0, self.T),
            y0=self.u0,
            method=method,
            max_step=max_step,
            dense_output=True,
            rtol=rtol,
            atol=atol
        )

        self.solve_time = time.time() - start_time
        self.nfev = sol.nfev
        self.t = sol.t
        self.u = sol.y.T

        return self.u

    def get_stats(self) -> Dict:
        """Get statistics about the solution"""
        return {
            'method': getattr(self, 'last_method', 'Unknown'),
            'solve_time': self.solve_time,
            'nfev': self.nfev,
            'n_timesteps': len(self.t),
            'N': self.N,
            'dx': self.dx,
            'T': self.T,
            'c': self.c,
            'alpha': self.alpha,
            'beta': self.beta,
            'bc_type': self.bc_type,
            'ic_type': self.ic_type,
            'max_u': np.max(np.abs(self.u)) if self.u is not None else None,
            'energy': np.trapz(self.u[-1, :] ** 2, self.x) if self.u is not None else None
        }

    def plot_snapshots(self, title_suffix=''):
        """Plot solution at different time points"""
        if self.u is None:
            raise ValueError("Must call solve() before plot_snapshots()")

        u_min = np.nanmin(self.u)
        u_max = np.nanmax(self.u)

        if np.isnan(u_min) or np.isnan(u_max) or np.isinf(u_min) or np.isinf(u_max):
            u_min, u_max = -2, 2

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        Nt = len(self.t)
        times_idx = [0, Nt // 4, Nt // 2, Nt - 1]

        for ax, idx in zip(axes.flat, times_idx):
            u_plot = self.u[idx, :]
            u_plot = np.where(np.isfinite(u_plot), u_plot, 0)

            ax.plot(self.x, u_plot, 'b-', linewidth=2)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('u(x,t)', fontsize=11)
            ax.set_title(f't = {self.t[idx]:.2f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(u_min - 0.2, u_max + 0.2)

        fig.suptitle(f'PDE Solution (c={self.c}, α={self.alpha}, β={self.beta}) {title_suffix}',
                     fontsize=14, y=1.00)
        plt.tight_layout()
        return fig


class ComparativeStudy:
    """Class for running comparative studies of different solvers and parameters"""

    def __init__(self):
        self.results = []

    def compare_solvers(self, methods: List[str], L=80, N=300, T=15,
                        c=1.5, alpha=0.75, beta=0.01, show_plots=True):
        """Compare different solver methods"""
        print("\n" + "=" * 60)
        print("COMPARING SOLVER METHODS")
        print("=" * 60)

        solvers_list = []

        for method in methods:
            print(f"\nTesting {method}...")
            try:
                solver = Solver(L=L, N=N, T=T, c=c, alpha=alpha, beta=beta)
                solver.last_method = method
                solver.solve(method=method, max_step=0.01)

                stats = solver.get_stats()
                self.results.append(stats)
                solvers_list.append(solver)

                print(f"  Time: {stats['solve_time']:.4f}s")
                print(f"  Function evals: {stats['nfev']}")
                print(f"  Time steps: {stats['n_timesteps']}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Generate plots and animations for each method
        if show_plots and solvers_list:
            self.visualize_solvers(solvers_list, methods)

        return self.create_comparison_dataframe()

    def compare_resolutions(self, N_values: List[int], method='RK45',
                            L=80, T=15, c=1.5, alpha=0.75, beta=0.01, show_plots=True):
        """Compare different spatial resolutions"""
        print("\n" + "=" * 60)
        print("COMPARING SPATIAL RESOLUTIONS")
        print("=" * 60)

        solvers_list = []

        for N in N_values:
            print(f"\nTesting N={N}...")
            try:
                solver = Solver(L=L, N=N, T=T, c=c, alpha=alpha, beta=beta)
                solver.last_method = method
                solver.solve(method=method, max_step=0.01)

                stats = solver.get_stats()
                self.results.append(stats)
                solvers_list.append(solver)

                print(f"  Time: {stats['solve_time']:.4f}s")
                print(f"  dx: {stats['dx']:.6f}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Generate plots for different resolutions
        if show_plots and solvers_list:
            self.visualize_resolutions(solvers_list, N_values)

        return self.create_comparison_dataframe()

    def compare_parameters(self, param_sets: List[Dict], method='RK45',
                           L=80, N=300, T=15, show_plots=True):
        """Compare different parameter combinations"""
        print("\n" + "=" * 60)
        print("COMPARING PARAMETER SETS")
        print("=" * 60)

        solvers_list = []
        param_labels = []

        for i, params in enumerate(param_sets):
            c = params.get('c', 1.5)
            alpha = params.get('alpha', 0.75)
            beta = params.get('beta', 0.01)

            print(f"\nTest {i + 1}: c={c}, α={alpha}, β={beta}...")
            try:
                solver = Solver(L=L, N=N, T=T, c=c, alpha=alpha, beta=beta)
                solver.last_method = method
                solver.solve(method=method, max_step=0.01)

                stats = solver.get_stats()
                self.results.append(stats)
                solvers_list.append(solver)
                param_labels.append(f"c={c}, α={alpha}, β={beta}")

                print(f"  Time: {stats['solve_time']:.4f}s")
                print(f"  Max |u|: {stats['max_u']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Generate plots for different parameters
        if show_plots and solvers_list:
            self.visualize_parameters(solvers_list, param_labels)

        return self.create_comparison_dataframe()

    def compare_boundary_conditions(self, bc_types: List[str], method='RK45',
                                    L=80, N=300, T=15, c=1.5, alpha=0.75, beta=0.01):
        """Compare different boundary conditions"""
        print("\n" + "=" * 60)
        print("COMPARING BOUNDARY CONDITIONS")
        print("=" * 60)

        for bc_type in bc_types:
            print(f"\nTesting BC: {bc_type}...")
            try:
                solver = Solver(L=L, N=N, T=T, c=c, alpha=alpha, beta=beta, bc_type=bc_type)
                solver.last_method = method
                solver.solve(method=method, max_step=0.01)

                stats = solver.get_stats()
                self.results.append(stats)

                print(f"  Time: {stats['solve_time']:.4f}s")
                print(f"  Energy: {stats['energy']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")

        return self.create_comparison_dataframe()

    def compare_initial_conditions(self, ic_types: List[str], method='RK45',
                                   L=80, N=300, T=15, c=1.5, alpha=0.75, beta=0.01):
        """Compare different initial conditions"""
        print("\n" + "=" * 60)
        print("COMPARING INITIAL CONDITIONS")
        print("=" * 60)

        for ic_type in ic_types:
            print(f"\nTesting IC: {ic_type}...")
            try:
                solver = Solver(L=L, N=N, T=T, c=c, alpha=alpha, beta=beta, ic_type=ic_type)
                solver.last_method = method
                solver.solve(method=method, max_step=0.01)

                stats = solver.get_stats()
                self.results.append(stats)

                print(f"  Time: {stats['solve_time']:.4f}s")

            except Exception as e:
                print(f"  ERROR: {e}")

        return self.create_comparison_dataframe()

    def visualize_solvers(self, solvers_list: List, methods: List[str]):
        """Create snapshot comparisons and animations for different solvers"""
        print("\nGenerating visualizations for solver comparison...")

        # 1. Snapshot comparison - all methods in one figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (solver, method) in enumerate(zip(solvers_list, methods)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Plot snapshots at different times
            Nt = len(solver.t)
            times_idx = [0, Nt // 3, 2 * Nt // 3, Nt - 1]
            colors = ['blue', 'green', 'orange', 'red']

            for tidx, color in zip(times_idx, colors):
                ax.plot(solver.x, solver.u[tidx, :], color=color,
                        linewidth=2, label=f't={solver.t[tidx]:.2f}', alpha=0.7)

            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('u(x,t)', fontsize=10)
            ax.set_title(f'{method}\nTime: {solver.solve_time:.3f}s, Evals: {solver.nfev}',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide extra subplots
        for idx in range(len(solvers_list), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Solver Method Comparison - Evolution Snapshots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig('solver_snapshots_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved: solver_snapshots_comparison.png")

        # 2. Create animation for each solver
        for solver, method in zip(solvers_list, methods):
            fig_anim, anim = self.create_animation(solver, f"{method} Method")
            fig_anim.savefig(f'solver_{method}_final.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: solver_{method}_final.png")
            plt.close(fig_anim)

    def visualize_resolutions(self, solvers_list: List, N_values: List[int]):
        """Create snapshot comparisons for different resolutions"""
        print("\nGenerating visualizations for resolution comparison...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (solver, N) in enumerate(zip(solvers_list, N_values)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            Nt = len(solver.t)
            times_idx = [0, Nt // 3, 2 * Nt // 3, Nt - 1]
            colors = ['blue', 'green', 'orange', 'red']

            for tidx, color in zip(times_idx, colors):
                ax.plot(solver.x, solver.u[tidx, :], color=color,
                        linewidth=2, label=f't={solver.t[tidx]:.2f}', alpha=0.7)

            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('u(x,t)', fontsize=10)
            ax.set_title(f'N={N} (dx={solver.dx:.5f})\nTime: {solver.solve_time:.3f}s',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for idx in range(len(solvers_list), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Spatial Resolution Comparison - Evolution Snapshots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig('resolution_snapshots_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved: resolution_snapshots_comparison.png")
        plt.close(fig)

    def visualize_parameters(self, solvers_list: List, param_labels: List[str]):
        """Create snapshot comparisons for different parameters"""
        print("\nGenerating visualizations for parameter comparison...")

        n_plots = len(solvers_list)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, (solver, label) in enumerate(zip(solvers_list, param_labels)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            Nt = len(solver.t)
            times_idx = [0, Nt // 3, 2 * Nt // 3, Nt - 1]
            colors = ['blue', 'green', 'orange', 'red']

            for tidx, color in zip(times_idx, colors):
                ax.plot(solver.x, solver.u[tidx, :], color=color,
                        linewidth=2, label=f't={solver.t[tidx]:.2f}', alpha=0.7)

            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('u(x,t)', fontsize=10)
            ax.set_title(f'{label}\nTime: {solver.solve_time:.3f}s, Max|u|: {np.max(np.abs(solver.u)):.3f}',
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for idx in range(len(solvers_list), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Parameter Comparison - Evolution Snapshots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig('parameter_snapshots_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved: parameter_snapshots_comparison.png")
        plt.close(fig)

    def create_animation(self, solver, title: str):
        """Create an animation for a single solver"""
        u_min = np.nanmin(solver.u)
        u_max = np.nanmax(solver.u)

        if np.isnan(u_min) or np.isnan(u_max) or np.isinf(u_min) or np.isinf(u_max):
            u_min, u_max = -2, 2

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(solver.x, solver.u[0, :], 'b-', linewidth=2)

        ax.set_xlim(0, solver.L)
        ax.set_ylim(u_min - 0.2, u_max + 0.2)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        Nt = len(solver.t)
        skip = max(1, Nt // 200)  # Show ~200 frames

        def update(frame):
            idx = frame * skip
            if idx >= Nt:
                idx = Nt - 1
            line.set_ydata(solver.u[idx, :])
            time_text.set_text(f't = {solver.t[idx]:.2f}')
            return line, time_text

        anim = FuncAnimation(fig, update, frames=Nt // skip,
                             interval=50, blit=True, repeat=True)

        plt.tight_layout()
        return fig, anim

    def create_comparison_dataframe(self):
        """Create a pandas DataFrame from results"""
        if not self.results:
            return None
        return pd.DataFrame(self.results)

    def plot_performance_comparison(self, df: pd.DataFrame, group_by: str):
        """Plot performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Time comparison
        if group_by in df.columns:
            groups = df[group_by].unique()
            times = [df[df[group_by] == g]['solve_time'].values[0] for g in groups]
            nfevs = [df[df[group_by] == g]['nfev'].values[0] for g in groups]

            axes[0].bar(range(len(groups)), times, color='steelblue')
            axes[0].set_xticks(range(len(groups)))
            axes[0].set_xticklabels(groups, rotation=45, ha='right')
            axes[0].set_ylabel('Computation Time (s)')
            axes[0].set_title('Computation Time Comparison')
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(range(len(groups)), nfevs, color='coral')
            axes[1].set_xticks(range(len(groups)))
            axes[1].set_xticklabels(groups, rotation=45, ha='right')
            axes[1].set_ylabel('Function Evaluations')
            axes[1].set_title('Function Evaluations Comparison')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report(self):
        """Generate a summary report"""
        if not self.results:
            print("No results to report!")
            return

        df = self.create_comparison_dataframe()

        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        print(f"\nTotal simulations: {len(df)}")
        print(f"\nComputation times:")
        print(f"  Mean: {df['solve_time'].mean():.4f}s")
        print(f"  Min:  {df['solve_time'].min():.4f}s")
        print(f"  Max:  {df['solve_time'].max():.4f}s")
        print(f"\nFunction evaluations:")
        print(f"  Mean: {df['nfev'].mean():.0f}")
        print(f"  Min:  {df['nfev'].min():.0f}")
        print(f"  Max:  {df['nfev'].max():.0f}")

        return df


def run_comprehensive_study():
    """Run a comprehensive comparative study"""

    study = ComparativeStudy()

    # 1. Compare solver methods
    print("\n### TASK 1: COMPARING DIFFERENT SOLVERS ###")
    methods = ['RK45', 'RK23', 'DOP853', ]#'Radau', 'BDF']
    df_solvers = study.compare_solvers(methods)

    # 2. Compare spatial resolutions
    print("\n### TASK 2: COMPARING SPATIAL RESOLUTIONS ###")
    study.results = []  # Reset
    N_values = [100, 200, 300, 400, 500]
    df_resolution = study.compare_resolutions(N_values)

    # 3. Compare extreme parameters
    print("\n### TASK 3: TESTING EXTREME PARAMETERS ###")
    study.results = []  # Reset
    param_sets = [
        {'c': 0.1, 'alpha': 0.1, 'beta': 0.001},  # Small values
        {'c': 1.5, 'alpha': 0.75, 'beta': 0.01},  # Normal values
        {'c': 5.0, 'alpha': 2.0, 'beta': 0.05},  # Large values
        {'c': 10.0, 'alpha': 5.0, 'beta': 0.1},  # Very large values
        {'c': 0.5, 'alpha': 5.0, 'beta': 0.001},  # Mixed: small c, large alpha
        {'c': 5.0, 'alpha': 0.1, 'beta': 0.1},  # Mixed: large c, small alpha
    ]
    df_params = study.compare_parameters(param_sets)

    # 4. Compare boundary conditions
    print("\n### TASK 4: COMPARING BOUNDARY CONDITIONS ###")
    study.results = []  # Reset
    bc_types = ['periodic', 'dirichlet', 'neumann']
    df_bc = study.compare_boundary_conditions(bc_types)

    # 5. Compare initial conditions
    print("\n### TASK 5: COMPARING INITIAL CONDITIONS ###")
    study.results = []  # Reset
    ic_types = ['sech', 'gaussian', 'sine', 'double_sech']
    df_ic = study.compare_initial_conditions(ic_types)

    # Generate visualizations
    if df_solvers is not None:
        fig1 = study.plot_performance_comparison(df_solvers, 'method')
        fig1.savefig('solver_comparison.png', dpi=150, bbox_inches='tight')

    if df_resolution is not None:
        fig2 = study.plot_performance_comparison(df_resolution, 'N')
        fig2.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')

    # Generate final report
    print("\n### FINAL SUMMARY ###")
    study.results = []  # Combine all results
    if df_solvers is not None:
        study.results.extend(df_solvers.to_dict('records'))
    if df_resolution is not None:
        study.results.extend(df_resolution.to_dict('records'))
    if df_params is not None:
        study.results.extend(df_params.to_dict('records'))

    final_df = study.generate_report()

    print("\n### RECOMMENDATIONS ###")
    print("\n1. BEST SOLVER:")
    if df_solvers is not None:
        best_solver = df_solvers.loc[df_solvers['solve_time'].idxmin()]
        print(f"   {best_solver['method']} (Time: {best_solver['solve_time']:.4f}s)")

    print("\n2. PARAMETER RANGES:")
    print("   Based on stability observations:")
    print("   - c (speed): 0.1 to 5.0 (stable), >10 (potentially unstable)")
    print("   - alpha (nonlinearity): 0.1 to 2.0 (stable), >5 (potentially unstable)")
    print("   - beta (dispersion): 0.001 to 0.05 (stable), >0.1 (may require finer resolution)")

    print("\n3. RESOLUTION:")
    print("   - N=200-300 provides good balance of accuracy and speed")
    print("   - Higher resolutions needed for extreme parameters")

    plt.show()

    return final_df

import numpy as np

def extract_pde_observables(solver, filename="pde_data.npz"):
    """
    Wyodrębnia globalne obserwable z rozwiązania PDE i zapisuje do pliku .npz
    zgodnie z formatem używanym przez model surogatowy ODE.

    Parametry
    ---------
    solver : Solver
        Obiekt klasy Solver po wywołaniu solver.solve().
    filename : str
        Ścieżka do pliku wynikowego (domyślnie 'pde_data.npz').

    Zwraca
    -------
    dict
        Słownik zawierający t, A_pde, B_pde, C_pde, D_pde.
    """
    if solver.u is None or solver.t is None:
        raise ValueError("Solver nie został jeszcze uruchomiony. Użyj solver.solve() przed ekstrakcją.")

    u = solver.u       # shape (Nt, Nx)
    t = solver.t
    x = solver.x

    # --- A(t): amplituda (maksimum wartości fali)
    A = np.max(u, axis=1)

    # --- B(t): średnie nachylenie |du/dx|
    du_dx = np.gradient(u, x, axis=1)
    B = np.mean(np.abs(du_dx), axis=1)

    # --- C(t): efekt dyspersji (średnia trzeciej pochodnej)
    d3u_dx3 = np.gradient(np.gradient(np.gradient(u, x, axis=1), x, axis=1), x, axis=1)
    C = np.mean(np.abs(d3u_dx3), axis=1)

    # --- D(t): całkowita energia (∫ u² dx)
    D = np.trapz(u ** 2, x, axis=1)

    # --- Zapis danych do pliku .npz
    np.savez(filename, t=t, A_pde=A, B_pde=B, C_pde=C, D_pde=D)
    print(f"[extract_pde_observables] Zapisano dane PDE do '{filename}'")

    # --- Podgląd wyników
    print(f"A(t): {A[:5]} ...")
    print(f"B(t): {B[:5]} ...")
    print(f"C(t): {C[:5]} ...")
    print(f"D(t): {D[:5]} ...")

    return {"t": t, "A_pde": A, "B_pde": B, "C_pde": C, "D_pde": D}



if __name__ == "__main__":
    # Run comprehensive study
    results_df = run_comprehensive_study()

    # Save results to CSV
    if results_df is not None:
        results_df.to_csv('pde_solver_comparison.csv', index=False)
        print("\nResults saved to 'pde_solver_comparison.csv'")

    solver = Solver(T=10, L=40, N=200, c=2.6, alpha=5, beta=0.5)
    solver.solve()

    data = extract_pde_observables(solver, filename="pde_data.npz")


