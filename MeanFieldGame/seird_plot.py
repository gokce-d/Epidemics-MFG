import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from SEIRD import (
    simulateEQ_contact_rate,
    SimulationSetup,
    EpidemicParams,
    CostParams,
    ControlParams,
)


# Plotting utilities
def plot_evolution(results, setup, epi, title_suffix=""):
    """Plot evolution of states, contact controls, and vaccination control."""
    p = results["p"].reshape(setup.n_states, setup.Nt, setup.n_blocks)
    controls = results["controls"]
    alpha_s = controls["alpha_s"]
    alpha_i = controls["alpha_i"]
    nu = controls["nu"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # Epidemic state trajectories
    labels = ["S", "E", "I", "R", "D"] if setup.exposed else ["S", "I", "R", "D"]
    for i, lbl in enumerate(labels):
        ax1.plot(setup.t_grid, p[i, :, 0], label=lbl)
    ax1.set_title(f"State Proportions Over Time {title_suffix}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Fraction of Population")
    ax1.legend()
    ax1.grid(True)

    # Social contact control α_S and α_I
    ax2.plot(setup.t_grid, alpha_s[0, :], label="α_S")
    if "alpha_i" in controls and controls["alpha_i"] is not None:
        ax2.plot(setup.t_grid, np.ravel(controls["alpha_i"]), label="α_I")
    ax2.set_title("Social Contact Control Levels")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control Value")
    ax2.legend()
    ax2.grid(True)

    # Vaccination control ν
    ax3.plot(
        setup.t_grid, np.ravel(nu), label="ν (Vaccination Control)", color="tab:green"
    )
    ax3.set_title("Vaccination Control")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Control Value")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# Comparison utilities
def compare_settings(
    results1, results2, setup, label1="Setting 1", label2="Setting 2", title_suffix=""
):
    """Compare two simulation results: state trajectories and control levels."""
    p1 = results1["p"].reshape(setup.n_states, setup.Nt, setup.n_blocks)
    p2 = results2["p"].reshape(setup.n_states, setup.Nt, setup.n_blocks)

    c1, c2 = results1["controls"], results2["controls"]
    alpha_s1, alpha_s2 = c1["alpha_s"], c2["alpha_s"]
    alpha_i1, alpha_i2 = c1["alpha_i"], c2["alpha_i"]
    nu1, nu2 = c1["nu"], c2["nu"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # Compare state proportions
    labels = ["S", "E", "I", "R", "D"] if setup.exposed else ["S", "I", "R", "D"]
    for i, lbl in enumerate(labels):
        ax1.plot(setup.t_grid, p1[i, :, 0], "--", label=f"{lbl} ({label1})")
        ax1.plot(setup.t_grid, p2[i, :, 0], "-", label=f"{lbl} ({label2})")
    ax1.set_title(f"Comparison of State Proportions {title_suffix}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Fraction of Population")
    ax1.legend()
    ax1.grid(True)

    # Compare α_S and α_I
    ax2.plot(setup.t_grid, alpha_s1[0, :], "--", label=f"α_S ({label1})")
    ax2.plot(setup.t_grid, alpha_s2[0, :], "-", label=f"α_S ({label2})")
    if alpha_i1 is not None:
        ax2.plot(setup.t_grid, np.ravel(alpha_i1), "--", label=f"α_I ({label1})")
    if alpha_i2 is not None:
        ax2.plot(setup.t_grid, np.ravel(alpha_i2), "-", label=f"α_I ({label2})")
    ax2.set_title("Social Contact Control Comparison")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control Value")
    ax2.legend()
    ax2.grid(True)

    # Compare ν
    ax3.plot(setup.t_grid, np.ravel(nu1), "--", label=f"ν ({label1})")
    ax3.plot(setup.t_grid, np.ravel(nu2), "-", label=f"ν ({label2})")
    ax3.set_title("Vaccination Control Comparison")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Control Value")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Setup base case
    n_blocks = 1
    Nt = 200
    T = 100
    t_grid = np.linspace(0, T, Nt)
    p_0 = np.array([0.99, 0.01, 0, 0, 0])

    setup = SimulationSetup(
        n_blocks=n_blocks,
        n_states=5,
        Nt=Nt,
        T=T,
        t_grid=t_grid,
        graphon=np.ones((n_blocks, n_blocks)),
        block_dens=np.ones(n_blocks),
        p_0=p_0,
        u_T=np.zeros(5 * n_blocks),
        death=True,
        exposed=True,
        epsilon=1e-5,
    )

    epi1 = EpidemicParams(beta=0.8, gamma=0.1, rho=0.1, chi=0.7, kappa=0.8, eta=0.01)
    epi2 = deepcopy(epi1)
    epi2.beta = 1.0  # comparison: higher transmission

    cost = CostParams(c_lambda=0.1, c_inf=0.3, c_dead=0.5, c_nu=0.1)
    ctrl = ControlParams(
        lambda_type=0,
        lambda_duration=None,
        lambda_s_in=np.array([0.99]),
        lambda_e_in=np.array([0.71]),
        lambda_i_in=np.array([1.0]),
        lambda_r_in=np.array([1.0]),
    )

    # ---- Run two simulations ----
    print("Running baseline simulation...")
    results1 = simulateEQ_contact_rate(setup, epi1, cost, ctrl)
    print("Running high-transmission simulation...")
    results2 = simulateEQ_contact_rate(setup, epi2, cost, ctrl)

    # ---- Plot and compare ----
    plot_evolution(results1, setup, epi1, title_suffix="(Baseline)")
    compare_settings(results1, results2, setup, label1="β=0.8", label2="β=1.0")
