import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy.linalg as la
from dataclasses import dataclass
import matplotlib.pyplot as plt

# from Graphon import Z_calculator


# ==========================
# Parameter Dataclasses
# ==========================


@dataclass
class EpidemicParams:
    beta: float
    gamma: float
    rho: float
    chi: float
    kappa: float
    eta: float


@dataclass
class CostParams:
    c_lambda: float
    c_inf: float
    c_dead: float
    c_nu: float


@dataclass
class ControlParams:
    lambda_type: int
    lambda_duration: np.ndarray | None
    lambda_s_in: np.ndarray
    lambda_e_in: np.ndarray
    lambda_i_in: np.ndarray
    lambda_r_in: np.ndarray


@dataclass
class SimulationSetup:
    n_blocks: int
    n_states: int
    Nt: int
    T: float
    t_grid: np.ndarray
    graphon: np.ndarray
    block_dens: np.ndarray
    p_0: np.ndarray
    u_T: np.ndarray
    death: bool
    exposed: bool
    epsilon: float
    n_print: int = 5
    exp_id: int = 0
    max_iter: int = 200


# ==========================
# Helper: Indexing
# ==========================


def get_state_indices(n_blocks, n_states, death=False, exposed=False):
    idx = {}
    pos = 0
    idx["S"] = slice(pos, pos + n_blocks)
    pos += n_blocks
    idx["I"] = slice(pos, pos + n_blocks)
    pos += n_blocks
    idx["R"] = slice(pos, pos + n_blocks)
    pos += n_blocks
    if death:
        idx["D"] = slice(pos, pos + n_blocks)
        pos += n_blocks
    if exposed:
        idx["E"] = slice(pos, pos + n_blocks)
        pos += n_blocks
    if pos != n_states * n_blocks:
        raise ValueError(
            f"Inconsistent indexing: expected {n_states*n_blocks}, got {pos}"
        )
    return idx


# ==========================
# Initialization
# ==========================


def initializer(setup: SimulationSetup, ctrl: ControlParams, cost: CostParams):
    before_p = np.tile(setup.p_0.reshape(-1, 1), (1, setup.Nt))
    before_u = np.zeros((setup.n_blocks * setup.n_states, setup.Nt))
    before_u[(setup.n_states - 1) * setup.n_blocks :, :] = cost.c_dead

    if ctrl.lambda_type == 0:  # same for every block, time independent
        lambda_s = np.tile(ctrl.lambda_s_in, setup.n_blocks)
        lambda_e = np.tile(ctrl.lambda_e_in, setup.n_blocks)
        lambda_i = np.tile(ctrl.lambda_i_in, setup.n_blocks)
        lambda_r = np.tile(ctrl.lambda_r_in, setup.n_blocks)
    elif ctrl.lambda_type == 1:  # different per block, time independent
        lambda_s, lambda_e, lambda_i, lambda_r = (
            ctrl.lambda_s_in,
            ctrl.lambda_e_in,
            ctrl.lambda_i_in,
            ctrl.lambda_r_in,
        )
    elif ctrl.lambda_type == 2:  # time dependent
        lambda_i = np.repeat(
            ctrl.lambda_i_in, ctrl.lambda_duration[1].astype(int), axis=1
        )
        lambda_r = np.repeat(
            ctrl.lambda_r_in, ctrl.lambda_duration[2].astype(int), axis=1
        )
        lambda_s = np.asarray(ctrl.lambda_s_in)
        lambda_e = np.asarray(ctrl.lambda_e_in)
    else:
        raise ValueError("Invalid lambda_type")
    return before_p, before_u, lambda_s, lambda_e, lambda_i, lambda_r


def Z_calculator(n_blocks, block_dens, lambda_i, graphon, p, lambda_type):
    """
    Compute Z over time. Ensures Z has shape (n_blocks, T).

    Args:
        n_blocks: int, number of blocks
        block_dens: (n_blocks,) density vector
        lambda_i: (n_blocks,) or (n_blocks, T), intensity scaling
        graphon: (n_blocks, n_blocks) adjacency/interaction matrix
        p: (5, T) compartment populations (S,E,I,R,D stacked by row)
        lambda_type: int, which formula variant to use
    """
    # Extract infected population trajectory: shape (n_blocks, T)
    I = p[1, :].reshape(1, -1)  # assuming idx["I"] == 1 (E=0,I=1,...)
    I = np.tile(I, (n_blocks, 1))  # replicate for each block

    block_dens = block_dens.reshape((n_blocks, 1))  # column vector

    if lambda_type in [0, 1]:
        term = (block_dens * lambda_i).reshape(n_blocks, 1) * I
        Z = graphon @ term
    elif lambda_type == 2:
        term = block_dens * lambda_i * I
        Z = graphon @ term
    else:
        raise ValueError(f"Invalid lambda_type: {lambda_type}")

    # Make sure shape is (n_blocks, T)
    return Z.reshape(n_blocks, -1)


# ==========================
# Control Calculations
# ==========================


def contact_rate_control_calc(
    ctrl: ControlParams,
    setup: SimulationSetup,
    epi: EpidemicParams,
    cost: CostParams,
    lambda_s,
    lambda_e,
    lambda_i,
    lambda_r,
    Z,
    u,
):
    idx = get_state_indices(
        setup.n_blocks, setup.n_states, death=setup.death, exposed=setup.exposed
    )

    if setup.Nt != u.shape[1]:
        raise ValueError("Mismatch between Nt and u timeline length")

    if setup.n_blocks != Z.shape[0]:
        raise ValueError("Mismatch between n_blocks and Z dimension")

    if setup.n_states * setup.n_blocks != u.shape[0]:
        raise ValueError("Mismatch in state dimensions")

    if setup.Nt != Z.shape[1]:
        # Interpolation needed for Z
        pass

    if setup.n_blocks == 1:
        Z_use = Z.flatten()
    else:
        Z_use = Z

    if setup.Nt == 1:
        u_S = u[idx["S"]].reshape(-1, 1)
        u_I = u[idx["I"]].reshape(-1, 1)
        u_R = u[idx["R"]].reshape(-1, 1)
    else:
        u_S = u[idx["S"], :]
        u_I = u[idx["I"], :]
        u_R = u[idx["R"], :]

    if ctrl.lambda_type in (0, 1):
        alpha_s = np.reshape(lambda_s, (setup.n_blocks, 1)) + np.reshape(
            epi.beta / (2 * cost.c_lambda), (setup.n_blocks, 1)
        ) * Z_use * (u_S - u_I)
        alpha_e = np.tile(np.reshape(lambda_e, (setup.n_blocks, 1)), setup.Nt)
        alpha_i = np.tile(np.reshape(lambda_i, (setup.n_blocks, 1)), setup.Nt)
        alpha_r = np.tile(np.reshape(lambda_r, (setup.n_blocks, 1)), setup.Nt)
        nu = np.reshape(epi.kappa / (2 * cost.c_nu), (setup.n_blocks, 1)) * (u_S - u_R)
    else:  # lambda_type == 2
        alpha_s = lambda_s + np.reshape(
            epi.beta / cost.c_lambda, (setup.n_blocks, 1)
        ) * (u_S - u_I)
        alpha_e, alpha_i, alpha_r = lambda_e, lambda_i, lambda_r
        nu = np.reshape(epi.kappa / (2 * cost.c_lambda), (setup.n_blocks, 1)) * (
            u_S - u_R
        )

    return alpha_s, alpha_e, alpha_i, alpha_r, nu


# ==========================
# Forward ODE (KFP)
# ==========================


def seird_rate_ODE_p(
    t, p, inter_alpha_s, inter_Z, inter_nu, setup: SimulationSetup, epi: EpidemicParams
):
    idx = get_state_indices(
        setup.n_blocks, setup.n_states, death=setup.death, exposed=setup.exposed
    )
    alpha_s, nu, Z = inter_alpha_s(t), inter_nu(t), inter_Z(t)
    rate = np.zeros_like(p)

    rate[idx["S"]] = (
        -epi.beta * alpha_s * Z * p[idx["S"]]
        - epi.kappa * nu * p[idx["S"]]
        + epi.eta * p[idx["R"]]
    )
    if not setup.death and not setup.exposed:  # SIR
        rate[idx["I"]] = epi.beta * alpha_s * Z * p[idx["S"]] - epi.gamma * p[idx["I"]]
        rate[idx["R"]] = (
            epi.gamma * p[idx["I"]]
            + epi.kappa * nu * p[idx["S"]]
            - epi.eta * p[idx["R"]]
        )
    elif setup.death and not setup.exposed:  # SIRD
        rate[idx["I"]] = epi.beta * alpha_s * Z * p[idx["S"]] - epi.gamma * p[idx["I"]]
        rate[idx["R"]] = (
            (1 - epi.rho) * epi.gamma * p[idx["I"]]
            + epi.kappa * nu * p[idx["S"]]
            - epi.eta * p[idx["R"]]
        )
        rate[idx["D"]] = epi.rho * epi.gamma * p[idx["I"]]
    elif setup.death and setup.exposed:  # SEIRD
        rate[idx["E"]] = epi.beta * alpha_s * Z * p[idx["S"]] - epi.chi * p[idx["E"]]
        rate[idx["I"]] = epi.chi * p[idx["E"]] - epi.gamma * p[idx["I"]]
        rate[idx["R"]] = (
            (1 - epi.rho) * epi.gamma * p[idx["I"]]
            + epi.kappa * nu * p[idx["S"]]
            - epi.eta * p[idx["R"]]
        )
        rate[idx["D"]] = epi.rho * epi.gamma * p[idx["I"]]
    return rate


def cdc_solver_KFP(
    setup: SimulationSetup, epi: EpidemicParams, inter_alpha_s, inter_Z, inter_nu
):
    sol_p = solve_ivp(
        seird_rate_ODE_p,
        [0, setup.T],
        setup.p_0.flatten(),
        t_eval=setup.t_grid,
        args=(inter_alpha_s, inter_Z, inter_nu, setup, epi),
    )
    return sol_p.y


# ==========================
# Backward ODE (HJB)
# ==========================


def seird_rate_ODE_u(
    t,
    u,
    inter_alpha_s,
    inter_Z,
    inter_nu,
    setup: SimulationSetup,
    epi: EpidemicParams,
    cost: CostParams,
    lambda_s,
):
    idx = get_state_indices(
        setup.n_blocks, setup.n_states, death=setup.death, exposed=setup.exposed
    )
    Z, nu, alpha_s = inter_Z(t), inter_nu(t), inter_alpha_s(t)
    rate = np.zeros_like(u)

    rate[idx["S"]] = (
        epi.beta * alpha_s * Z * (u[idx["S"]] - u[idx["I"]])
        - cost.c_lambda * ((lambda_s - alpha_s) ** 2)
        + epi.kappa * nu * (u[idx["S"]] - u[idx["R"]])
        - cost.c_nu * (nu**2)
    )
    rate[idx["R"]] = epi.eta * (u[idx["R"]] - u[idx["S"]])  # correct
    if not setup.death and not setup.exposed:  # SIR
        rate[idx["I"]] = epi.gamma * (u[idx["I"]] - u[idx["R"]]) + cost.c_inf
    elif setup.death and not setup.exposed:  # SIRD
        rate[idx["I"]] = (
            (1 - epi.rho) * epi.gamma * (u[idx["I"]] - u[idx["R"]])
            + epi.rho * epi.gamma * (u[idx["I"]] - u[idx["D"]])
            - cost.c_inf
        )
        rate[idx["D"]] = 0
    elif setup.death and setup.exposed:  # SEIRD
        rate[idx["S"]] = (
            epi.beta * alpha_s * Z * (u[idx["S"]] - u[idx["E"]])
            - cost.c_lambda * ((lambda_s - alpha_s) ** 2)
            + epi.kappa * nu * (u[idx["S"]] - u[idx["R"]])
            - cost.c_nu * (nu**2)
        )  # correct
        rate[idx["I"]] = (
            (1 - epi.rho) * epi.gamma * (u[idx["I"]] - u[idx["R"]])
            + epi.rho * epi.gamma * (u[idx["I"]] - u[idx["D"]])
            - cost.c_inf
        )
        rate[idx["D"]] = 0
        rate[idx["E"]] = epi.chi * (u[idx["E"]] - u[idx["I"]])
    return rate


def cdc_solver_HJB(
    setup: SimulationSetup,
    epi: EpidemicParams,
    cost: CostParams,
    inter_alpha_s,
    inter_Z,
    inter_nu,
    lambda_s,
):
    backward_t_grid = setup.T - setup.t_grid
    sol_u = solve_ivp(
        seird_rate_ODE_u,
        [setup.T, 0],
        setup.u_T,
        t_eval=backward_t_grid,
        args=(inter_alpha_s, inter_Z, inter_nu, setup, epi, cost, lambda_s),
    )
    return np.flip(sol_u.y, axis=1)


# ==========================
# Main Simulator
# ==========================


def simulateEQ_contact_rate(
    setup: SimulationSetup, epi: EpidemicParams, cost: CostParams, ctrl: ControlParams
):
    before_p, before_u, lambda_s, lambda_e, lambda_i, lambda_r = initializer(
        setup, ctrl, cost
    )

    inter_lambda_s = (
        interp1d(setup.t_grid, lambda_s, kind="previous")
        if ctrl.lambda_type == 2
        else lambda_s
    )

    Z = Z_calculator(
        setup.n_blocks,
        setup.block_dens,
        lambda_i,
        setup.graphon,
        before_p,
        ctrl.lambda_type,
    )
    # inter_Z = interp1d(setup.t_grid, Z.T, axis=0)
    inter_Z = interp1d(setup.t_grid, Z)

    alpha_s, alpha_e, alpha_i, alpha_r, nu = contact_rate_control_calc(
        ctrl, setup, epi, cost, lambda_s, lambda_e, lambda_i, lambda_r, Z, before_u
    )
    inter_alpha_s, inter_nu = interp1d(setup.t_grid, alpha_s), interp1d(
        setup.t_grid, nu
    )

    after_p = cdc_solver_KFP(setup, epi, inter_alpha_s, inter_Z, inter_nu)
    after_u = cdc_solver_HJB(
        setup, epi, cost, inter_alpha_s, inter_Z, inter_nu, inter_lambda_s
    )

    convergence_p = [la.norm(after_p - before_p)]
    convergence_u = [la.norm(after_u - before_u)]
    print(
        f"iter: 0 | p conv: {convergence_p[-1]:.3e} | u conv: {convergence_u[-1]:.3e}"
    )

    i = 0
    while (
        (convergence_p[-1] > setup.epsilon) or (convergence_u[-1] > setup.epsilon)
    ) and i < setup.max_iter:
        i += 1
        Z = Z_calculator(
            setup.n_blocks,
            setup.block_dens,
            lambda_i,
            setup.graphon,
            after_p,
            ctrl.lambda_type,
        )
        inter_Z = interp1d(setup.t_grid, Z)
        # inter_Z = interp1d(setup.t_grid, Z.T, axis=0)
        alpha_s, alpha_e, alpha_i, alpha_r, nu = contact_rate_control_calc(
            ctrl, setup, epi, cost, lambda_s, lambda_e, lambda_i, lambda_r, Z, after_u
        )
        inter_alpha_s, inter_nu = interp1d(setup.t_grid, alpha_s), interp1d(
            setup.t_grid, nu
        )

        before_p, before_u = after_p.copy(), after_u.copy()
        after_p = cdc_solver_KFP(setup, epi, inter_alpha_s, inter_Z, inter_nu)
        after_u = cdc_solver_HJB(
            setup, epi, cost, inter_alpha_s, inter_Z, inter_nu, inter_lambda_s
        )

        convergence_p.append(la.norm(after_p - before_p))
        convergence_u.append(la.norm(after_u - before_u))
        if i % setup.n_print == 0:
            print(
                f"iter: {i} | p conv: {convergence_p[-1]:.3e} | u conv: {convergence_u[-1]:.3e}"
            )

    return {
        "p": after_p,
        "u": after_u,
        "controls": {
            "alpha_s": alpha_s,
            "alpha_e": alpha_e,
            "alpha_i": alpha_i,
            "alpha_r": alpha_r,
            "nu": nu,
        },
        "Z": Z,
        "convergence": {"p": np.array(convergence_p), "u": np.array(convergence_u)},
        "iterations": i,
    }


# ==========================
# Testing
# ==========================


def main():
    n_blocks = 1
    n_states = 5
    Nt = 200
    T = 100
    p_0 = np.array([0.99, 0.01, 0.0, 0.0, 0.0])
    t_grid = np.linspace(0, T, Nt)
    setup = SimulationSetup(
        n_blocks=n_blocks,
        n_states=n_states,
        Nt=Nt,
        T=T,
        t_grid=t_grid,
        graphon=np.ones((n_blocks, n_blocks)),
        block_dens=np.ones(n_blocks),
        p_0=p_0,
        u_T=np.zeros(n_states * n_blocks),
        death=True,
        exposed=True,
        epsilon=1e-5,
    )
    # setup.u_T[(n_states - 1) * n_blocks :] = 0.7  # terminal cost of death?

    epi = EpidemicParams(beta=0.2, gamma=0.1, rho=0.5, chi=0.9, kappa=0.2, eta=0.1)
    cost = CostParams(c_lambda=0.1, c_inf=0.1, c_dead=0.1, c_nu=0.1)
    ctrl = ControlParams(
        lambda_type=0,
        lambda_duration=None,
        lambda_s_in=np.array([1.0]),
        lambda_e_in=np.array([1.0]),
        lambda_i_in=np.array([1.0]),
        lambda_r_in=np.array([1.0]),
    )

    results = simulateEQ_contact_rate(setup, epi, cost, ctrl)
    p = results["p"].reshape(setup.n_states, setup.Nt, setup.n_blocks)
    S, E, I, R, D = p[:, :, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(setup.t_grid, S, label="S")
    plt.plot(setup.t_grid, E, label="E")
    plt.plot(setup.t_grid, I, label="I")
    plt.plot(setup.t_grid, R, label="R")
    plt.plot(setup.t_grid, D, label="D")
    plt.xlabel("Time")
    plt.ylabel("Fraction of population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
