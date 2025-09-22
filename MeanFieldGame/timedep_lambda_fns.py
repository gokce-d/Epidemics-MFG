import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy.linalg as la
from datetime import date

from CDC2025_vaxx import simulateEQ_contact_rate


# Lambda(t) functions
def logistic_lambda(t, L0=2.0, Lmin=0.5, k=0.1, t0=40):
    """Logistic decrease in social contact regulation"""
    return Lmin + (L0 - Lmin) / (1 + np.exp(k * (t - t0)))


def linear_lambda(t, L0=2.0, Lend=0.5, T=100):
    """Linear decline in regulation"""
    return np.maximum(Lend, L0 - (L0 - Lend) * t / T)


def exp_lambda(t, L0=2.0, rate=0.05, Lmin=0.5):
    """Exponential decay of regulation"""
    return Lmin + (L0 - Lmin) * np.exp(-rate * t)


def oscillatory_lambda(t, L0=1.5, amp=0.5, freq=0.2):
    """Oscillatory (on-off lockdown waves)"""
    return np.maximum(0.1, L0 + amp * np.sin(freq * t))


# Simulation setup
T = 100
Nt = 200
t_grid = np.linspace(0, T, Nt)

n_blocks = 1
n_states = 3  # S, I, R
death = 0  # no explicit death compartment

# Epidemiological parameters
beta = 0.5
gamma = 0.1
kappa = 0.05
rho = 0.9

# Cost parameters
c_lambda = 1.0
c_inf = 1.0
c_dead = 1.0
c_nu = 1.0

# Initial conditions
p0 = np.array([0.99, 0.01, 0.0])  # start with 1% infected
uT = np.zeros(n_blocks * n_states)  # terminal costates

# Graphon/density placeholder
graphon = np.array([[1.0]])  # fully connected
block_dens = np.array([1.0])  # single block

# constant lambda_i and lambda_r
lambda_i_in = np.array([[0.8, 0.8]])
lambda_r_in = np.array(
    [[1.0, 1.0]]
)  # formatted so initializer doesn't need to be changed
lambda_duration = [
    np.array([Nt // 2, Nt // 2]),
    np.array([Nt // 2, Nt // 2]),
    np.array([Nt // 2, Nt // 2]),
]

epsilon = 1e-3
n_print = 5
exp_id = "demo"

# Dict of lambda functions
lambda_functions = {
    "Logistic": logistic_lambda,
    "Linear": lambda t: linear_lambda(t, T=T),
    "Exponential": exp_lambda,
    "Oscillatory": oscillatory_lambda,
}

results = {}
lambda_traces = {}

# ----------------------------
# Run simulations
# ----------------------------
for name, func in lambda_functions.items():
    lambda_s_in = np.array([func(t) for t in t_grid])
    lambda_traces[name] = lambda_s_in  # lambda_s smooth time-dependent function

    after_u, after_p, alpha_s, alpha_i, alpha_r, nu, Z, conv_p, conv_u = (
        simulateEQ_contact_rate(
            n_blocks,
            n_states,
            Nt,
            lambda_s_in,
            lambda_i_in,
            lambda_r_in,
            graphon,
            beta,
            kappa,
            gamma,
            rho,
            c_lambda,
            c_inf,
            c_dead,
            c_nu,
            t_grid,
            T,
            p0,
            uT,
            n_print,
            exp_id,
            block_dens,
            lambda_type=2,  # time-dependent lambda
            lambda_duration=lambda_duration,
            death=death,
            epsilon=epsilon,
        )
    )
    results[name] = after_p


# Plot results
# SIR evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

for ax, (name, p) in zip(axes, results.items()):
    S = p[0, :]
    I = p[1, :]
    R = p[2, :]
    ax.plot(t_grid, S, label="S")
    ax.plot(t_grid, I, label="I")
    ax.plot(t_grid, R, label="R")
    ax.set_title(f"{name} lambda(t)")
    ax.set_xlabel("Time")
    ax.legend()

axes[0].set_ylabel("Proportion")
axes[2].set_ylabel("Proportion")
plt.tight_layout()
today = date.today().strftime("%Y-%m-%d")
plt.savefig(f"timedep_lambda_states_{today}.png", dpi=300, bbox_inches="tight")
plt.show()

# Lambda evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

for ax, (name, lam) in zip(axes, lambda_traces.items()):
    ax.plot(t_grid, lam, color="tab:blue")
    ax.set_title(f"{name} lambda_s(t)")
    ax.set_xlabel("Time")
    ax.set_ylabel("lambda_s")

plt.tight_layout()
plt.savefig(f"timedep_lambda_{today}.png", dpi=300, bbox_inches="tight")
plt.show()
