"""
surogat_ode.py

Skrypt: implementacja i symulacja modelu surogatowego ODE:

  dA/dt = -alpha_s * A * B + beta_s * C
  dB/dt = -gamma_s * B + delta_s * A
  dC/dt = -lambda_s * C + mu_s * A**2
  dD/dt = nu_s * B - xi_s * D

Funkcjonalności:
- definicja parametrów (przykładowe wartości)
- integracja układu ODE (scipy.solve_ivp)
- możliwość wczytania danych referencyjnych (PDE) z pliku .npz z kluczami: t, A_pde, B_pde, C_pde, D_pde
- obliczenie RMSE między rozwiązaniem ODE a danymi PDE (jeśli są dostępne)
- wykresy porównawcze i zapis wyników do .npz

Jak używać:
1) Jeśli masz już wyekstrahowane A(t),B(t),C(t),D(t) z symulacji PDE, zapisz je jako plik NumPy .npz z kluczami:
   np.savez('pde_data.npz', t=t_array, A_pde=A_array, B_pde=B_array, C_pde=C_array, D_pde=D_array)

2) Uruchom:
   python surogat_ode.py        # użyje domyślnych wartości początkowych
   python surogat_ode.py pde_data.npz   # porówna ODE z danymi PDE

Uwaga: ten skrypt nie zakłada konkretnych BC ani implementacji PDE — oczekuje jedynie wyekstrahowanych sygnałów A,B,C,D z PDE.

"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# ---------------------------
# Parametry surogatu (przykładowe)
# ---------------------------
params = {
    'alpha_s': 1,   # odpowiada -alpha * A * B
    'beta_s' : 0.1,   # wzmocnienie terminu C w dA/dt
    'gamma_s': 0.3,   # tłumienie B
    'delta_s': 0.2,   # wpływ A na B
    'lambda_s': 0.4,  # tłumienie C
    'mu_s': 0.05,     # nieliniowy wzrost C ~ A^2
    'nu_s': 0.1,      # wpływ B na D
    'xi_s': 0.02      # tłumienie D
}

# Zakres czasowy (przykładowy)
T = 10.0
n_eval = 1000
t_eval = np.linspace(0, T, n_eval)

# ---------------------------
# Definicja RHS
# ---------------------------

def surrogate_rhs(t, y, p):
    A, B, C, D = y
    alpha_s = p['alpha_s']
    beta_s = p['beta_s']
    gamma_s = p['gamma_s']
    delta_s = p['delta_s']
    lambda_s = p['lambda_s']
    mu_s = p['mu_s']
    nu_s = p['nu_s']
    xi_s = p['xi_s']

    dA = -alpha_s * A * B + beta_s * C
    dB = -gamma_s * B + delta_s * A
    dC = -lambda_s * C + mu_s * A**2
    dD = nu_s * B - xi_s * D
    return [dA, dB, dC, dD]

# ---------------------------
# Default initial conditions (przykładowe)
# Jeśli masz dane PDE, lepiej ustawić je zgodnie z PDE w t=0
# ---------------------------
A0 = 1.0
B0 = 0.0
C0 = 0.0
D0 = 0.0
y0_default = [A0, B0, C0, D0]

# ---------------------------
# Funkcje pomocnicze
# ---------------------------

def integrate_surogat(y0, params, t_span=(0.0, T), t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(lambda t, y: surrogate_rhs(t, y, params), t_span, y0,
                    t_eval=t_eval, method='RK45', atol=1e-8, rtol=1e-6)
    return sol.t, sol.y  # sol.y shape (4, nt)


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b)**2))


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

def surrogate_ode(t, y, p):
    α, β, γ, δ, λ, μ, ν, ξ = p
    A, B, C, D = y
    dA = -α * A * B + β * C
    dB = -γ * B + δ * A
    dC = -λ * C + μ * A**2
    dD = ν * B - ξ * D
    return [dA, dB, dC, dD]


def fit_parameters(t, A_pde, B_pde, C_pde, D_pde, y0=None):
    """Fit surrogate ODE parameters to PDE observables."""
    y_pde = np.vstack([A_pde, B_pde, C_pde, D_pde])

    if y0 is None:
        y0 = y_pde[:, 0]

    def simulate(p):
        sol = solve_ivp(lambda t, y: surrogate_ode(t, y, p),
                        (t[0], t[-1]), y0, t_eval=t, method='RK45', max_step=0.01)
        return sol.y

    def residuals(p):
        y_ode = simulate(p)
        return (y_ode - y_pde).ravel()

    # Initial guess
    p0 = np.random.uniform(0.01, 1.0, 8)
    result = least_squares(residuals, p0, bounds=(0, 5))

    print("Fitted parameters:")
    param_names = ['α', 'β', 'γ', 'δ', 'λ', 'μ', 'ν', 'ξ']
    for name, val in zip(param_names, result.x):
        print(f"  {name} = {val:.4f}")

    return result.x


# ---------------------------
# Główna procedura
# ---------------------------

def main():
    global params
    data = np.load("pde_data.npz")
    params = fit_parameters(data["t"], data["A_pde"], data["B_pde"], data["C_pde"], data["D_pde"])

    # Czy użytkownik podał plik z danymi PDE?
    pde_file = "pde_data.npz"
    if len(sys.argv) > 1:
        pde_file = sys.argv[1]

    if pde_file is not None and os.path.exists(pde_file):
        print(f'Wczytuję dane PDE z {pde_file} ...')
        data = np.load(pde_file)
        # oczekujemy kluczy: t, A_pde, B_pde, C_pde, D_pde
        t_pde = data['t']
        A_pde = data['A_pde']
        B_pde = data['B_pde']
        C_pde = data['C_pde']
        D_pde = data['D_pde']

        # dopasuj przestrzeń czasową surogatu do PDE (użyj t_pde jako t_eval)
        t_eval_local = t_pde
        # inicjalne warunki z PDE na t=0
        y0 = [A_pde[0], B_pde[0], C_pde[0], D_pde[0]]
        t_sol, y_sol = integrate_surogat(y0, params, t_span=(t_pde[0], t_pde[-1]), t_eval=t_eval_local)

        A_sol, B_sol, C_sol, D_sol = y_sol

        # Oblicz RMSE dla każdej zmiennej
        rmse_A = rmse(A_sol, A_pde)
        rmse_B = rmse(B_sol, B_pde)
        rmse_C = rmse(C_sol, C_pde)
        rmse_D = rmse(D_sol, D_pde)

        print('RMSE porównania ODE (surogat) vs PDE:')
        print(f'  A: {rmse_A:.6f}, B: {rmse_B:.6f}, C: {rmse_C:.6f}, D: {rmse_D:.6f}')

        # Zapis wyników
        outname = 'surogat_results.npz'
        np.savez(outname, t=t_sol, A_sol=A_sol, B_sol=B_sol, C_sol=C_sol, D_sol=D_sol,
                 A_pde=A_pde, B_pde=B_pde, C_pde=C_pde, D_pde=D_pde)
        print(f'Zapisano wyniki do {outname}')

        # Rysunki
        plot_compare(t_sol, A_sol, B_sol, C_sol, D_sol, A_pde, B_pde, C_pde, D_pde)

    else:
        print('Brak pliku PDE — uruchamiam symulację surogatu z wartościami domyślnymi.')
        t_sol, y_sol = integrate_surogat(y0_default, params, t_span=(0.0, T), t_eval=t_eval)
        A_sol, B_sol, C_sol, D_sol = y_sol
        np.savez('surogat_only.npz', t=t_sol, A_sol=A_sol, B_sol=B_sol, C_sol=C_sol, D_sol=D_sol)
        plot_only(t_sol, A_sol, B_sol, C_sol, D_sol)


# ---------------------------
# Wykresy
# ---------------------------

def plot_compare(t, A_sol, B_sol, C_sol, D_sol, A_pde, B_pde, C_pde, D_pde):
    plt.figure(figsize=(10,8))
    ax = plt.subplot(2,2,1)
    ax.plot(t, A_pde, label='PDE A', linewidth=2)
    ax.plot(t, A_sol, '--', label='ODE A')
    ax.set_title('A(t)')
    ax.legend()

    ax = plt.subplot(2,2,2)
    ax.plot(t, B_pde, label='PDE B', linewidth=2)
    ax.plot(t, B_sol, '--', label='ODE B')
    ax.set_title('B(t)')
    ax.legend()

    ax = plt.subplot(2,2,3)
    ax.plot(t, C_pde, label='PDE C', linewidth=2)
    ax.plot(t, C_sol, '--', label='ODE C')
    ax.set_title('C(t)')
    ax.legend()

    ax = plt.subplot(2,2,4)
    ax.plot(t, D_pde, label='PDE D', linewidth=2)
    ax.plot(t, D_sol, '--', label='ODE D')
    ax.set_title('D(t)')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_only(t, A_sol, B_sol, C_sol, D_sol):
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1); plt.plot(t, A_sol); plt.title('A(t)')
    plt.subplot(2,2,2); plt.plot(t, B_sol); plt.title('B(t)')
    plt.subplot(2,2,3); plt.plot(t, C_sol); plt.title('C(t)')
    plt.subplot(2,2,4); plt.plot(t, D_sol); plt.title('D(t)')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
