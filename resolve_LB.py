from pyomo.environ import *

from calculate_hat_discretization import hat_variable  

from problem_data import (Ns, Nf, Pr, Feed, z_feed, Reflux, H_feed, kk,
                          liq_coeffs, vap_coeffs, Bott, Dist,
                          Qcond_upper, Qcond_lower, Qreb_upper, Qreb_lower,
                          x_upper, x_lower) 

"""
    Solve the Lower Bound (LB) model for the distillation column.

    This function constructs and solves a MILP model that represents a lower bound
    for the original distillation column problem. The model uses discretization
    of the continuous variables (temperature, liquid flow, vapor flow).

    Parameters:
    -----------
    variable_bounds : dict
        A dictionary with keys 'T', 'L', 'V'. Each key maps to another dictionary with 
        'lower' and 'upper' keys, which are lists of bounds for each stage.
        Example: 
            {'T': {'lower': [350, ...], 'upper': [385, ...]},
             'L': {'lower': [50, ...],  'upper': [190, ...]},
             'V': {'lower': [120, ...], 'upper': [140, ...]}}
    Card : dict
        A dictionary with the number of discretization for each variable.
        Keys: 'T', 'L', 'V'

    Returns:
    --------
    dict
        A dictionary containing the solution and model results. Key elements include:
        - 'termination_condition': The solver termination condition.
        - 'objective_value': The value of the objective function (Qr).
        - 'Qr': Reboiler duty (kJ/h).
        - 'Qc': Condenser duty (kJ/h).
        - 'T': List of temperatures by stage (K).
        - 'L': List of liquid flow rates by stage (kmol/h). Note: L[Ns] is NaN.
        - 'V': List of vapor flow rates by stage (kmol/h). Note: V[1] is NaN.
        - 'x': Dictionary of compositions by component and stage.
        - 'phi': Dictionary of phi values by component and stage.
        - 'lx', 'vy', 'hl', 'hv', 'hx': Auxiliary variables for linearization.
        - 'lambda', 'omega', 'rho': Binary variables for discretization.
        - 'a', 'b', 'c', 'd': Auxiliary variables for products.
        - 'alpha', 'sigma': Auxiliary binary variables for products.
        - 'model': The Pyomo model object (if available).
        - 'results': The solver results object (if available).

    Notes:
    ------
    - The model fixes the composition of benzene in the condenser (x[1,1]) to 0.98 and 
      the composition of toluene in the reboiler (x[2,Ns]) to 0.98.
    - The model uses the hat_variable function to generate discretization points.
    - The solver used is CPLEX via GAMS.
"""


def solve_LB(variable_bounds, Card):
 
    model = ConcreteModel()

    # =============================================================================
    # Sets
    # =============================================================================
    model.components = Set(initialize=[1, 2])       # Components (1=Benzene, 2=Toluene)
    model.stages = Set(initialize=range(1, Ns+1))   # Ns stages (1=condenser, Ns=reboiler)


    M_card = Card['T']  # Number of points for temperature
    N_card = Card['L']  # Number of points for liquid flow rate
    P_card = Card['V']  # Number of points for vapor flow rate

    model.M_set = Set(initialize=range(1, M_card+1))  # goes from 1 to M_card
    model.N_set = Set(initialize=range(1, N_card+1))  
    model.P_set = Set(initialize=range(1, P_card+1))      

    model.M_prime = Set(initialize=range(1, M_card)) # goes from 1 to M_card-1
    model.N_prime = Set(initialize=range(1, N_card))
    model.P_prime = Set(initialize=range(1, P_card))


    # =============================================================================
    # Variables
    # =============================================================================
    # Flow rates - using limits from variable_bounds dictionary
    model.L = Var(
        model.stages - {Ns},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variable_bounds['L']['lower'][j-1] if j < Ns else None,
            variable_bounds['L']['upper'][j-1] if j < Ns else None
        )
    )

    model.V = Var(
        model.stages - {1},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variable_bounds['V']['lower'][j-1] if j > 1 else None,
            variable_bounds['V']['upper'][j-1] if j > 1 else None
        )
    )

    # Compositions
    model.x = Var(
        model.components, model.stages,
        domain=NonNegativeReals,
        bounds=(x_lower, x_upper)
    )

    # Temperatures - using limits from variable_bounds dictionary
    model.T = Var(
        model.stages,
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variable_bounds['T']['lower'][j-1],
            variable_bounds['T']['upper'][j-1]
        )
    )

    # Heat duties
    model.Qc = Var(
        domain=Reals,
        bounds=(Qcond_lower, Qcond_upper)
    )
    model.Qr = Var(
        domain=Reals,
        bounds=(Qreb_lower, Qreb_upper)
    )


    # Auxiliary variables for linearization
    model.phi = Var(model.components, model.stages, domain=NonNegativeReals)
    model.lx = Var(model.components, model.stages - {Ns}, domain=NonNegativeReals)
    model.vy = Var(model.components, model.stages - {1}, domain=NonNegativeReals)
    model.hl = Var(model.components, model.stages - {Ns}, domain=Reals)
    model.hv = Var(model.components, model.stages - {1}, domain=Reals)
    model.hx = Var(model.components, model.stages, domain=Reals)    

    # Binary variables for discretization
    model.lambda_var = Var(model.stages, model.M_prime, domain=Binary)
    model.omega_var = Var(model.stages - {Ns}, model.N_prime, domain=Binary)
    model.rho_var = Var(model.stages - {1}, model.P_prime, domain=Binary)    

    # Auxiliary variables for products
    model.a = Var(model.components, model.stages, model.M_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.b = Var(model.components, model.stages - {Ns}, model.N_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.c = Var(model.components, model.stages - {1}, model.M_prime, model.P_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.d = Var(model.components, model.stages - {Ns}, model.M_prime, model.N_prime, domain=NonNegativeReals, bounds=(0, x_upper))    

    # Auxiliary binary variables for products
    model.alpha = Var(model.stages - {1}, model.M_prime, model.P_prime, domain=Binary)
    model.sigma = Var(model.stages - {Ns}, model.M_prime, model.N_prime, domain=Binary)



 
    # =============================================================================
    # Print discretized variables
    # =============================================================================
    def init_T_hat(model, j, m):
        return hat_variable(m, M_card, [variable_bounds['T']['lower'][j-1], variable_bounds['T']['upper'][j-1]])

    def init_L_hat(model, j, n):
        if j < Ns:  # L is only defined for stages 1 to 16
            return hat_variable(n, N_card, [variable_bounds['L']['lower'][j-1], variable_bounds['L']['upper'][j-1]])
        else:
            return 0  # Default value for stage 17 (not used)

    def init_V_hat(model, j, p):
        if j > 1:  # V is only defined for stages 2 to 17
            return hat_variable(p, P_card, [variable_bounds['V']['lower'][j-1], variable_bounds['V']['upper'][j-1]])
        else:
            return 0  # Default value for stage 1 (not used)

    model.T_hat = Param(model.stages, model.M_set, initialize=init_T_hat)
    model.L_hat = Param(model.stages, model.N_set, initialize=init_L_hat)
    model.V_hat = Param(model.stages, model.P_set, initialize=init_V_hat)

    # Fix compositions as specified
    model.x[1, 1].fix(0.98)  # x11 is not a variable (fixed at 0.98)
    model.x[2, Ns].fix(0.98) # x2_Ns is not a variable (fixed at 0.98)

    model.reflux_constraint = Constraint(expr = model.L[1] == Reflux * Dist)

    # =============================================================================
    # Constraints
    # =============================================================================

    # Discretization constraints for T, L, V
    def temp_discretization_lower(model, j):
        return sum(model.T_hat[j, m] * model.lambda_var[j, m] for m in model.M_prime) <= model.T[j]

    def temp_discretization_upper(model, j):
        return model.T[j] <= sum(model.T_hat[j, m+1] * model.lambda_var[j, m] for m in model.M_prime)

    model.temp_lower = Constraint(model.stages, rule=temp_discretization_lower)
    model.temp_upper = Constraint(model.stages, rule=temp_discretization_upper)    

    def liquid_discretization_lower(model, j):
        return sum(model.L_hat[j, n] * model.omega_var[j, n] for n in model.N_prime) <= model.L[j]
        
    def liquid_discretization_upper(model, j):
        return model.L[j] <= sum(model.L_hat[j, n+1] * model.omega_var[j, n] for n in model.N_prime)
        
    model.liquid_lower = Constraint(model.stages - {Ns}, rule=liquid_discretization_lower)
    model.liquid_upper = Constraint(model.stages - {Ns}, rule=liquid_discretization_upper)

    def vapor_discretization_lower(model, j):
        return sum(model.V_hat[j, p] * model.rho_var[j, p] for p in model.P_prime) <= model.V[j]
        
    def vapor_discretization_upper(model, j):
        return model.V[j] <= sum(model.V_hat[j, p+1] * model.rho_var[j, p] for p in model.P_prime)

    model.vapor_lower = Constraint(model.stages - {1}, rule=vapor_discretization_lower)
    model.vapor_upper = Constraint(model.stages - {1}, rule=vapor_discretization_upper)

    def lambda_sum_rule(model, j):
        return sum(model.lambda_var[j, m] for m in model.M_prime) == 1
        
    def omega_sum_rule(model, j):
        return sum(model.omega_var[j, n] for n in model.N_prime) == 1
        
    def rho_sum_rule(model, j):
        return sum(model.rho_var[j, p] for p in model.P_prime) == 1
        
    model.lambda_sum = Constraint(model.stages, rule=lambda_sum_rule)
    model.omega_sum = Constraint(model.stages - {Ns}, rule=omega_sum_rule)
    model.rho_sum = Constraint(model.stages - {1}, rule=rho_sum_rule)


    # Composition normalization (excludes stages 1 and Ns due to fixations)
    def norm_x_rule(model, j):
        if j == 1 or j == Ns:
            return Constraint.Skip
        return sum(model.x[i, j] for i in model.components) == 1
    model.norm_x_eq = Constraint(model.stages, rule=norm_x_rule)

    def norm_phi_rule(model, j):
        return (1/Pr) * sum(model.phi[i, j] for i in model.components) == 1
    model.norm_phi_eq = Constraint(model.stages, rule=norm_phi_rule)

    # Constraints for φ
    def phi_lower_rule(model, i, j):
        return sum(exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m])) * model.a[i, j, m] 
                    for m in model.M_prime) <= model.phi[i, j]
        
    def phi_upper_rule(model, i, j):
        return model.phi[i, j] <= sum(exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m+1])) * model.a[i, j, m] 
                    for m in model.M_prime)
        

    model.phi_lower = Constraint(model.components, model.stages, rule=phi_lower_rule)
    model.phi_upper = Constraint(model.components, model.stages, rule=phi_upper_rule)


    # Material balances
    def mass_balance_total(model, j):
        if j == 1:  # Total condenser
            return model.V[j+1] == model.L[j] + Dist
        elif j == Ns:  # Partial reboiler
            return model.L[j-1] == model.V[j] + Bott
        else:  # Internal stages
            if j == Nf:  # Feed stage
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1] + Feed
            else:
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1]
    model.mass_balance_total_eq = Constraint(model.stages, rule=mass_balance_total)

    # Component balances
    def mass_balance_component(model, i, j):
        if j == 1:  # Condenser
            return (1/Pr) * model.vy[i, j+1] - model.lx[i, j] - Dist * model.x[i, j] == 0
        elif j == Ns:  # Reboiler
            return model.lx[i, j-1] == (1/Pr) * model.vy[i, j] + Bott * model.x[i, j]
        else:  # Internal stages
            if j == Nf:
                return (model.lx[i, j] + (1/Pr) * model.vy[i, j] - model.lx[i, j-1] - 
                        (1/Pr) * model.vy[i, j+1] - Feed * z_feed == 0)
            else:
                return (model.lx[i, j] + (1/Pr) * model.vy[i, j] - model.lx[i, j-1] - 
                        (1/Pr) * model.vy[i, j+1] == 0)
    model.mass_balance_component_eq = Constraint(model.components, model.stages, rule=mass_balance_component)

    # Constraints for lx
    def lx_lower_rule(model, i, j):
        return sum(model.L_hat[j, n] * model.b[i, j, n] for n in model.N_prime) <= model.lx[i, j]
        
    def lx_upper_rule(model, i, j):
        return model.lx[i, j] <= sum(model.L_hat[j, n+1] * model.b[i, j, n] for n in model.N_prime)
        
    model.lx_lower = Constraint(model.components, model.stages-{Ns}, rule=lx_lower_rule)
    model.lx_upper = Constraint(model.components, model.stages-{Ns}, rule=lx_upper_rule)


    def b_constraint1(model, i, j, n):
        return model.b[i, j, n] - x_upper * model.omega_var[j, n] <= 0
        
    def b_constraint2(model, i, j, n):
        return (model.x[i, j] - model.b[i, j, n]) - x_upper * (1 - model.omega_var[j, n]) <= 0
        
    def b_constraint3(model, i, j, n):
        return -model.x[i, j] + model.b[i, j, n] <= 0
        
    model.b_constr1 = Constraint(model.components, model.stages-{Ns}, model.N_prime, rule=b_constraint1)
    model.b_constr2 = Constraint(model.components, model.stages-{Ns}, model.N_prime, rule=b_constraint2)
    model.b_constr3 = Constraint(model.components, model.stages-{Ns}, model.N_prime, rule=b_constraint3)
        
    # Constraints for vy 
    def vy_lower_rule(model, i, j):
            return sum(model.V_hat[j, p] * exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m])) * 
                model.c[i, j, m, p] for m in model.M_prime for p in model.P_prime) <= model.vy[i, j]
        
    def vy_upper_rule(model, i, j):
            return model.vy[i, j] <= sum(model.V_hat[j, p+1] * exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m+1])) * 
                model.c[i, j, m, p] for m in model.M_prime for p in model.P_prime)

    model.vy_lower = Constraint(model.components, model.stages - {1}, rule=vy_lower_rule)
    model.vy_upper = Constraint(model.components, model.stages - {1}, rule=vy_upper_rule)

    def alpha_constraint1(model, j, m, p):
        return model.alpha[j, m, p] <= model.rho_var[j, p]
        
    def alpha_constraint2(model, j, m, p):
        return model.alpha[j, m, p] <= model.lambda_var[j, m]
        
    def alpha_constraint3(model, j, m, p):
        return model.rho_var[j, p] + model.lambda_var[j, m] - 1 <= model.alpha[j, m, p]
        
    model.alpha_constr1 = Constraint(model.stages - {1}, model.M_prime, model.P_prime, rule=alpha_constraint1)
    model.alpha_constr2 = Constraint(model.stages - {1}, model.M_prime, model.P_prime, rule=alpha_constraint2)
    model.alpha_constr3 = Constraint(model.stages - {1}, model.M_prime, model.P_prime, rule=alpha_constraint3)


    def c_constraint1(model, i, j, m, p):
        return model.c[i, j, m, p] - x_upper * model.alpha[j, m, p] <= 0
        
    def c_constraint2(model, i, j, m, p):
        return (model.x[i, j] - model.c[i, j, m, p]) - x_upper * (1 - model.alpha[j, m, p]) <= 0
        
    def c_constraint3(model, i, j, m, p):
        return -model.x[i, j] + model.c[i, j, m, p] <= 0

    model.c_constr1 = Constraint(model.components, model.stages - {1}, model.M_prime, model.P_prime, rule=c_constraint1)
    model.c_constr2 = Constraint(model.components, model.stages - {1}, model.M_prime, model.P_prime, rule=c_constraint2)
    model.c_constr3 = Constraint(model.components, model.stages - {1}, model.M_prime, model.P_prime, rule=c_constraint3)

    # =============================================================================
    # Energy balances 
    # =============================================================================
    def energy_balance(model, j):
        if j == 1:  # Condenser
            return ((1/Pr) * sum(model.hv[i, 2] for i in model.components) - 
                        sum(model.hl[i, 1] for i in model.components) - 
                        Dist* sum(model.hx[i, 1] for i in model.components) - 
                        model.Qc == 0)
        elif j == Ns:  # Reboiler
            return (sum(model.hl[i, Ns-1] for i in model.components) + model.Qr - 
                        (1/Pr) * sum(model.hv[i, Ns] for i in model.components) - 
                        Bott * sum(model.hx[i, Ns] for i in model.components) == 0)
        else:  # Internal stages
            if j == Nf:
                return (sum(model.hl[i, j] for i in model.components) + 
                        (1/Pr) * sum(model.hv[i, j] for i in model.components) - 
                            sum(model.hl[i, j-1] for i in model.components) - 
                            (1/Pr) * sum(model.hv[i, j+1] for i in model.components) - 
                            Feed * H_feed == 0)
            else:
                return (sum(model.hl[i, j] for i in model.components) + 
                            (1/Pr) * sum(model.hv[i, j] for i in model.components) - 
                            sum(model.hl[i, j-1] for i in model.components) - 
                            (1/Pr) * sum(model.hv[i, j+1] for i in model.components) == 0)
    model.energy_balance_eq = Constraint(model.stages, rule=energy_balance)
        
    # Functions for enthalpy calculation
    def h0_liq_expr(model, i, T_val):
            return (liq_coeffs[i, 1] + 
                    liq_coeffs[i, 2] * T_val + 
                    liq_coeffs[i, 3] * T_val**2 + 
                    liq_coeffs[i, 4] * T_val**3)
        
    def h0_vap_expr(model, i, T_val):
            return (vap_coeffs[i, 1] + 
                    vap_coeffs[i, 2] * T_val + 
                    vap_coeffs[i, 3] * T_val**2 + 
                    vap_coeffs[i, 4] * T_val**3)
        
    # Constraints for hl
    def hl_lower_rule(model, i, j):
            return sum(h0_liq_expr(model, i, model.T_hat[j, m]) * model.L_hat[j, n] * model.d[i, j, m, n] 
                for m in model.M_prime for n in model.N_prime) <= model.hl[i, j]
        
    def hl_upper_rule(model, i, j):
            return model.hl[i, j] <= sum(h0_liq_expr(model, i, model.T_hat[j, m+1]) * model.L_hat[j, n+1] * model.d[i, j, m, n] 
                for m in model.M_prime for n in model.N_prime)

    model.hl_lower = Constraint(model.components, model.stages - {Ns}, rule=hl_lower_rule)
    model.hl_upper = Constraint(model.components, model.stages - {Ns}, rule=hl_upper_rule)

    def sigma_constraint1(model, j, m, n):
        return model.sigma[j, m, n] <= model.lambda_var[j, m]
        
    def sigma_constraint2(model, j, m, n):
        return model.sigma[j, m, n] <= model.omega_var[j, n]
        
    def sigma_constraint3(model, j, m, n):
        return model.lambda_var[j, m] + model.omega_var[j, n] - 1 <= model.sigma[j, m, n]
        
    model.sigma_constr1 = Constraint(model.stages - {Ns}, model.M_prime, model.N_prime, rule=sigma_constraint1)
    model.sigma_constr2 = Constraint(model.stages - {Ns}, model.M_prime, model.N_prime, rule=sigma_constraint2)
    model.sigma_constr3 = Constraint(model.stages - {Ns}, model.M_prime, model.N_prime, rule=sigma_constraint3)

    def d_constraint1(model, i, j, m, n):
        return model.d[i, j, m, n] - x_upper * model.sigma[j, m, n] <= 0
        
    def d_constraint2(model, i, j, m, n):
        return (model.x[i, j] - model.d[i, j, m, n]) - x_upper * (1 - model.sigma[j, m, n]) <= 0
        
    def d_constraint3(model, i, j, m, n):
        return -model.x[i, j] + model.d[i, j, m, n] <= 0
        
    model.d_constr1 = Constraint(model.components, model.stages - {Ns}, model.M_prime, model.N_prime, rule=d_constraint1)
    model.d_constr2 = Constraint(model.components, model.stages - {Ns}, model.M_prime, model.N_prime, rule=d_constraint2)
    model.d_constr3 = Constraint(model.components, model.stages - {Ns}, model.M_prime, model.N_prime, rule=d_constraint3)
        
    # Constraints for hv 
    def hv_lower_rule(model, i, j):
        return sum(exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m])) * 
                    h0_vap_expr(model, i, model.T_hat[j, m]) * model.V_hat[j, p] * model.c[i, j, m, p] 
                for m in model.M_prime for p in model.P_prime) <= model.hv[i, j]
        
    def hv_upper_rule(model, i, j):
        return model.hv[i, j] <= sum(exp(kk[i, 1] - kk[i, 2]/(kk[i, 3] + model.T_hat[j, m+1])) * 
                    h0_vap_expr(model, i, model.T_hat[j, m+1]) * model.V_hat[j, p+1] * model.c[i, j, m, p] 
                for m in model.M_prime for p in model.P_prime)
        
    model.hv_lower = Constraint(model.components, model.stages - {1}, rule=hv_lower_rule)
    model.hv_upper = Constraint(model.components, model.stages - {1}, rule=hv_upper_rule)
        
    # Constraints for hx
    def hx_lower_rule(model, i, j):
        return sum(h0_liq_expr(model, i, model.T_hat[j, m]) * model.a[i, j, m] for m in model.M_prime) <= model.hx[i, j]
        
    def hx_upper_rule(model, i, j):
        return model.hx[i, j] <= sum(h0_liq_expr(model, i, model.T_hat[j, m+1]) * model.a[i, j, m] for m in model.M_prime)
        
    model.hx_lower = Constraint(model.components, model.stages, rule=hx_lower_rule)
    model.hx_upper = Constraint(model.components, model.stages, rule=hx_upper_rule)
        
    def a_constraint1(model, i, j, m):
        return model.a[i, j, m] - x_upper * model.lambda_var[j, m] <= 0
        
    def a_constraint2(model, i, j, m):
        return (model.x[i, j] - model.a[i, j, m]) - x_upper * (1 - model.lambda_var[j, m]) <= 0
        
    def a_constraint3(model, i, j, m):
        return model.x[i, j] - model.a[i, j, m] >= 0
        
    model.a_constr1 = Constraint(model.components, model.stages, model.M_prime, rule=a_constraint1)
    model.a_constr2 = Constraint(model.components, model.stages, model.M_prime, rule=a_constraint2)
    model.a_constr3 = Constraint(model.components, model.stages, model.M_prime, rule=a_constraint3)

    # =============================================================================
    # Total Energy Balance
    # =============================================================================
    def total_energy_balance(model):
            return (Feed * H_feed + model.Qr == 
                Dist * (sum(model.hx[i, 1] for i in model.components)) + 
                Bott * (sum(model.hx[i, Ns] for i in model.components)) + 
                model.Qc)
    model.total_energy_eq = Constraint(rule=total_energy_balance)

    # Objective function
    def objective_rule(model):
        return model.Qr
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # =============================================================================
    # Solve the model
    # =============================================================================
    solver = SolverFactory('gams')
    solver.options['solver'] = 'cplex'
    results = solver.solve(model)
    
    # =============================================================================
    # Collect and return results
    # =============================================================================
    # Check if solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        # Collect values of main variables
        termination_condition = results.solver.termination_condition
        Qr_val = value(model.Qr)
        Qc_val = value(model.Qc)
        
        # Collect values by stage
        T_values = [value(model.T[j]) for j in model.stages]
        L_values = [value(model.L[j]) if j in model.L else float('nan') for j in model.stages]
        V_values = [value(model.V[j]) if j in model.V else float('nan') for j in model.stages]
        
        # Collect compositions
        x_values = {}
        for i in model.components:
            x_values[i] = [value(model.x[i, j]) for j in model.stages]
        
        # Collect auxiliary variables
        phi_values = {}
        lx_values = {}
        vy_values = {}
        hl_values = {}
        hv_values = {}
        hx_values = {}
        for i in model.components:
            phi_values[i] = [value(model.phi[i, j]) for j in model.stages]
            lx_values[i] = [value(model.lx[i, j]) for j in model.stages - {Ns}]
            vy_values[i] = [value(model.vy[i, j]) for j in model.stages - {1}]
            hl_values[i] = [value(model.hl[i, j]) for j in model.stages - {Ns}]
            hv_values[i] = [value(model.hv[i, j]) for j in model.stages - {1}]
            hx_values[i] = [value(model.hx[i, j]) for j in model.stages]
        
        # Collect binary variables
        lambda_values = {}
        for j in model.stages:
            lambda_values[j] = [value(model.lambda_var[j, m]) for m in model.M_prime]
        
        omega_values = {}
        for j in model.stages - {Ns}:
            omega_values[j] = [value(model.omega_var[j, n]) for n in model.N_prime]
        
        rho_values = {}
        for j in model.stages - {1}:
            rho_values[j] = [value(model.rho_var[j, p]) for p in model.P_prime]
        
        # Collect product variables
        a_values = {}
        for i in model.components:
            a_values[i] = {}
            for j in model.stages:
                a_values[i][j] = [value(model.a[i, j, m]) for m in model.M_prime]
        
        b_values = {}
        for i in model.components:
            b_values[i] = {}
            for j in model.stages - {Ns}:
                b_values[i][j] = [value(model.b[i, j, n]) for n in model.N_prime]
        
        c_values = {}
        for i in model.components:
            c_values[i] = {}
            for j in model.stages - {1}:
                c_values[i][j] = {}
                for m in model.M_prime:
                    c_values[i][j][m] = [value(model.c[i, j, m, p]) for p in model.P_prime]
        
        d_values = {}
        for i in model.components:
            d_values[i] = {}
            for j in model.stages - {Ns}:
                d_values[i][j] = {}
                for m in model.M_prime:
                    d_values[i][j][m] = [value(model.d[i, j, m, n]) for n in model.N_prime]
        
        # Collect auxiliary binary variables
        alpha_values = {}
        for j in model.stages - {1}:
            alpha_values[j] = {}
            for m in model.M_prime:
                alpha_values[j][m] = [value(model.alpha[j, m, p]) for p in model.P_prime]
        
        sigma_values = {}
        for j in model.stages - {Ns}:
            sigma_values[j] = {}
            for m in model.M_prime:
                sigma_values[j][m] = [value(model.sigma[j, m, n]) for n in model.N_prime]
        
        # Return dictionary with all results
        results_dict = {
            'termination_condition': termination_condition,
            'objective_value': Qr_val,
            'Qr': Qr_val,
            'Qc': Qc_val,
            'T': T_values,
            'L': L_values,
            'V': V_values,
            'x': x_values,
            'phi': phi_values,
            'lx': lx_values,
            'vy': vy_values,
            'hl': hl_values,
            'hv': hv_values,
            'hx': hx_values,
            'lambda': lambda_values,
            'omega': omega_values,
            'rho': rho_values,
            'a': a_values,
            'b': b_values,
            'c': c_values,
            'd': d_values,
            'alpha': d_values,
            'sigma': d_values,
            'model': model,  # Optional: return complete model if needed
            'results': results  # Optional: return complete results
        }
 

        return results_dict
        
    else:
        #print(f"Solver did not converge. Termination condition: {results.solver.termination_condition}")
        return {
            'termination_condition': results.solver.termination_condition,
            'error': 'Solver did not converge to an optimal solution'
        }
