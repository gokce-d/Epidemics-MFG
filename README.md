# Mean Field Game for Epidemic Control
This repository contains Python code for simulating and solving a Stackelberg Mean Field Game (MFG) model applied to epidemic control, as described in the paper "Optimal Incentives to Mitigate Epidemics: A Stackelberg Mean Field Game Approach" by Alexander Aurell, René Carmona, Gökçe Dayanıklı, and Mathieu Laurière (SIAM J. Control Optim., 2022).

## Overview
The code implements a numerical solver for an MFG model of epidemic dynamics in a population divided into blocks (e.g., age groups, regions), using the SIR (Susceptible, Infected, Removed) or SIRD (Susceptible, Infected, Removed, Deceased) framework. Individuals control their contact rates (αₛ)to minimize personal costs (infection risk, deviation from regulator’s policy), while a regulator sets recommended contact and vaccination rates (λₛ,λᵢ,λᵣ)to influence behavior and mitigate the epidemic. A graphon models heterogeneous interactions between blocks.
### Key Feartures
- Solves forward Kolmogorov-Fokker-Planck (KFP) equations for population densities (𝑝).
- Solves backward Hamilton-Jacobi-Bellman (HJB) equations for value functions (𝑢).
- Computes optimal controls per block:
	- Contact rates (αₛ, αᵢ) reflecting socialization behavior
	- Vaccination rates (ν) based on cost-benefit tradeoffs between infection and vaccination
- Supports constant or block-specific regulator policies (𝜆).
- Models SIR (death=0) or SIRD (death=1) dynamics.
- Visualizes population states, controls, vaccination effort, interactions (𝑍), and convergence, with comparison plots for different runs.
The code approximates the mean field limit with a finite number of blocks, focusing on the population’s equilibrium response to fixed regulator policies.


### Contributors
- Gökçe Dayanıklı
- Yichen Zhou (Spring 2025)
- Arseniy Titov (Spring 2025)

## Dependencies
Required Python libraries:
- `numpy` 
- `scipy`: ODE solving (`solve_ivp`), interpolation (`interp1d`).
- `matplotlib`
- `seaborn`
- `pandas`
- `tqdm`

## Code Structure
- `initializer`: Sets initial population density (p₀), value function (u), and regulator policies (λₛ,λᵢ,λᵣ). Supports: 
    - `lambda_type=0`: Same 𝜆 for all blocks, time-independent.
    - `lambda_type=1`: Block-specific 𝜆, time-independent.
    - `lambda_type=2`: Block-specific 𝜆, time-dependent.

- `Z_calculator`: Computes interaction term 𝑍, the expected contact rate with infected individuals:
    𝑍 = graphon ⋅ (block_dens ⋅ λᵢ ⋅ p_I)
    𝑍 drives infection risk p_I = β⋅αₛ⋅Z⋅p_S

- `opt_control_calculator`: Calculates optimal controls:
    αₛ = λₛ + (β/c_λ) Z (u_S - u_I)
    αₛ balance regulator guidance (λₛ) with infection risk (𝑍, u_S - u_I)

- `rate_ODE_p`, `solver_KFP`: Solve KFP equations for population dynamics
    - p_S= - β⋅αₛ⋅Z⋅p_S+κP_R
    - p_I=β⋅αₛ⋅Z⋅p_S-γP_I
    - p_R=γ P_I-κ P_R (SIR) or p_R= ρ γ P_I-κP_R, p_D=(1-ρ)γP_I (SIRD)

- `rate_ODE_u`, `solver_HJB`: Solve HJB equations for value functions:
    - u_s=-(β λₛ 𝑍(u_I - u_S)- (β²)/(2c_λ) Z² (u_I - u_S)²)
    - u_I= - (γ (u_R- u_I)+ c_inf)
    - u_D= -c_dead (SIRD)

- `stoch_block_fixed`: Iteratively solves KFP and HJB until convergence 

- `plotting`: Visualizes one run’s results (density, controls, 𝑍, convergence).
    - Density: Tacks p_S, p_I, p_R, p_D per block
    - Interaction (𝑍): High 𝑍 early signals epidemic spread; it drops as p_I declines
    - Controls (αₛ): Lower αₛ. reflects isolation during high 𝑍, rising later as risk fades.

- `comparison_plotting `: Compares two runs, showing differences in infected density (p_I), interaction (𝑍), and controls (αₛ)
    - Infected Density (p_I): Lower peaks show better control
    - Interaction (𝑍): Lower 𝑍 means less mixing with infected people, reducing risk
    - Controls (αₛ): Stricter λᵢ may indirectly lower αₛ via 𝑍 as people avoid contacts

- contact_rate_control_calc: Computes optimal controls including vaccination:
	- Contact controls:
		- αₛ = λₛ + (β / 2c_λ) ⋅ Z ⋅ (u_S − u_I)
		- αᵢ and αᵣ are repeated across time from λᵢ and λᵣ.
	- Vaccination control:
		- ν = (κ / 2c_ν) ⋅ (u_S − u_R)
		- Reflects optimal vaccination effort balancing benefit and cost.

- cdc_rate_ODE_p, cdc_solver_KFP: Solve modified KFP equations with vaccination:
	- p_S = −β αₛ Z p_S − κ ν p_S
	- p_I = β αₛ Z p_S − γ p_I
	- p_R = γ p_I + κ ν p_S (SIR)
	or
	- p_R = ρ γ p_I − κ p_R, p_D = (1 − ρ) γ p_I (SIRD)

- cdc_rate_ODE_u, cdc_solver_HJB: Solve modified HJB equations with vaccination:

	- u_S = β αₛ Z (u_S − u_I) − c_λ(λₛ − αₛ)² + κ ν (u_S − u_R) − c_ν ν²
	- u_I = −γ (u_R − u_I) − c_inf
	- u_R = 0 (SIR) or solved from κ-related terms (SIRD)

- simulateEQ_contact_rate_vaccination: Runs iterative scheme combining modified KFP and HJB equations, updating αₛ, ν, and Z at each step until convergence.

- plot_vaccination_only: Visualizes vaccination control ν(t) for each block over time.
Highlights the time-varying vaccination effort in response to the epidemic.



## Usage
```python
import numpy as np

T = 200.0  # Time horizon
Nt = 20000  # Time points
t_grid = np.linspace(0, T, Nt)
n_blocks = 4
n_states = 4  # S, I, R, D
death = 1  # SIRD model
lambda_type = 1  # Block-specific, time-independent lambda
Delta_t = t_grid[1]-t_grid[0]
date=5

# Epidemic parameters (per block)
beta = np.array([0.4, 0.3, 0.3, 0.3])  # Infection rate
gamma = np.array([0.1, 0.1, 0.05, 0.05])  # Recovery rate
rho = np.array([1.0, 1.0, 0.9, 0.75])  # Recovery probability
kappa = np.zeros(4)  # No recycling (R to S)
c_lambda = np.array([10.0, 10.0, 10.0, 10.0])  # Deviation cost
c_inf = np.array([1., 1., 1., 1.,])   # Infection cost
c_dead = np.array([1., 1., 1., 1.,])   # Death cost

# Graphon: Connectivity between blocks
graphon = np.array([[1.0, 0.9, 0.8, 0.7],
                    [0.9, 0.9, 0.8, 0.8],
                    [0.8, 0.8, 0.9, 0.8],
                    [0.7, 0.8, 0.8, 0.8]])
block_dens = [0.27, 0.33, 0.27, 0.13]  # Block sizes

# Initial conditions
p_0 = np.array([[0.95], [0.97], [0.97], [0.97], [0.05], [0.03], [0.03], [0.03], \
                [0.00], [0.00], [0.00], [0.00], [0.00], [0.00], [0.00], [0.00]])
u_T = np.array([[0], [0], [0], [0], [0], [0], [0], [0], \
                [0], [0], [0], [0], [0], [0], [0], [0]])  

# Regulator policy
lambda_s_in = [1.0, 1.0, 1.0, 1.0]
lambda_i_in = [1.0, 1.0, 1.0, 1.0]
lambda_r_in = [1.0, 1.0, 1.0, 1.0]
lambda_duration = np.zeros(0)  # Not used for lambda_type=1

epsilon = 1e-7  # Convergence threshold
n_print = 10  # Print every 10 iterations

```
