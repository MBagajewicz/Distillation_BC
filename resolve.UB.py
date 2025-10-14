from pyomo.environ import *         

from problem_data import (Ns, Nf, Pr, Feed, z_feed, Reflux, H_feed, kk, 
                          liq_coeffs, vap_coeffs, Bott, Dist,
                          Qcond_upper, Qcond_lower, Qreb_upper, Qreb_lower,
                          x_upper, x_lower) 


"""
    SOLVE UPPER BOUND (UB) MODEL FOR DISTILLATION COLUMN OPTIMIZATION
    
    This function constructs and solves the Upper Bound (UB) NLP model for a 
    benzene-toluene distillation column. The UB model represents the
    nonlinear problem and provides an upper bound on the optimal solution.
    
    The UB model uses the solution from the Lower Bound (LB) model as initial
    point to warm-start the NLP solver.

    Parameters:
    -----------
    variable_bounds : dict
        Dictionary containing upper and lower bounds for decision variables.
        Structure: {
            'T': {'lower': [list], 'upper': [list]},  # Temperature bounds [K]
            'L': {'lower': [list], 'upper': [list]},  # Liquid flow bounds [kmol/h]
            'V': {'lower': [list], 'upper': [list]}   # Vapor flow bounds [kmol/h]
        }
        Each list has length Ns (number of stages)
    
    UB_init : dict
        Dictionary containing initial values for all variables from LB solution.
        Structure: {
            'T': [list],      # Temperature initial values [K]
            'L': [list],      # Liquid flow initial values [kmol/h]
            'V': [list],      # Vapor flow initial values [kmol/h]
            'x': {            # Composition initial values
                1: [list],    # Benzene compositions
                2: [list]     # Toluene compositions
            },
            'Qc': float,      # Condenser duty initial value [kJ/h]
            'Qr': float       # Reboiler duty initial value [kJ/h]
        }
    
    Returns:
    --------
    dict
        Comprehensive results dictionary containing:
        - termination_condition: Solver termination status
        - objective_value: Optimal objective function value (Qr) [kJ/h]
        - Qr: Reboiler duty [kJ/h]
        - Qc: Condenser duty [kJ/h]
        - T: Temperature profile by stage [K]
        - L: Liquid flow rate profile by stage [kmol/h]
        - V: Vapor flow rate profile by stage [kmol/h]
        - x: Composition profiles by component and stage
        - model: Pyomo model instance (optional)
        - results: Solver results object (optional)
    
    Notes:
    ------
    The UB model implements the original nonlinear distillation column model:
    1. Nonlinear Vapor-Liquid Equilibrium (VLE) using Antoine equation
    2. Nonlinear component mass balances with equilibrium relationships
    3. Nonlinear energy balances with temperature-dependent enthalpies
    4. Total mass balances with feed stage consideration
    5. Fixed compositions: x₁₁ = 0.98 (distillate), x₂_Ns = 0.98 (bottoms)
    
    Mathematical Approach:
    ----------------------
    - Represents the original MINLP problem as an NLP
    - Uses exact nonlinear expressions for:
        * VLE: K_i = exp(A_i - B_i/(T + C_i))/Pr
        * Liquid enthalpy: polynomial function of temperature
        * Vapor enthalpy: polynomial function of temperature
    - Implements rigorous MESH equations (Mass, Equilibrium, Summation, Heat)
    
    
    Solver Configuration:
    ---------------------
    - Solver: CONOPT via GAMS interface
    - Problem Type: NLP (Nonlinear Programming)
    - Objective: Minimize reboiler duty (Qr)
 
    Notes:
    ------
    - Stage numbering: 1 (condenser) to Ns (reboiler)
    - Feed stage is at position Nf
    - Components: 1 (Benzene), 2 (Toluene)
    - Liquid flow L not defined at stage Ns (reboiler)
    - Vapor flow V not defined at stage 1 (condenser)
    - Pressure (Pr) is constant throughout the column
    - Uses polynomial correlations for enthalpy calculations
"""

def solve_UB(variable_bounds, UB_init):

    model = ConcreteModel()

    # =============================================================================
    # Sets
    # =============================================================================
    model.components = Set(initialize=[1, 2])       # Components (1=Benzene, 2=Toluene)
    model.stages = Set(initialize=range(1, Ns+1))   # Ns stages (1=condenser, Ns=reboiler)


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

     # Initialize liquid flow rates (only stages 1 to 16)
    for j in model.stages:
        if j < Ns:  # Stages 1 to 16
            model.L[j].value = UB_init['L'][j-1]

    model.V = Var(
        model.stages - {1},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variable_bounds['V']['lower'][j-1] if j > 1 else None,
            variable_bounds['V']['upper'][j-1] if j > 1 else None
        )
    )

    # Initialize vapor flow rates (only stages 2 to 17)
    for j in model.stages:
        if j > 1:  # Stages 2 to 17
            model.V[j].value = UB_init['V'][j-1]

    # Compositions
    model.x = Var(
        model.components, model.stages,
        domain=NonNegativeReals,
        bounds=(x_lower, x_upper)
    )

    # Initialize compositions
    for i in model.components:
        for j in model.stages:
            model.x[i, j].value = UB_init['x'][i][j-1]

    # Temperatures - using limits from variable_bounds dictionary
    model.T = Var(
        model.stages,
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variable_bounds['T']['lower'][j-1],
            variable_bounds['T']['upper'][j-1]
        )
    )

    if UB_init is not None:
        # Initialize temperatures
        for j in model.stages:
            model.T[j].value = UB_init['T'][j-1]

    # Heat duties
    model.Qc = Var(
        domain=Reals,
        bounds=(Qcond_lower, Qcond_upper)
    )

    model.Qr = Var(
        domain=Reals,
        bounds=(Qreb_lower, Qreb_upper)
    )

    # Initialize heat duties
    model.Qc.value = UB_init['Qc']
    model.Qr.value = UB_init['Qr']

    # Fix compositions as specified
    model.x[1, 1].fix(0.98)  # x11 is not a variable (fixed at 0.98)
    model.x[2, Ns].fix(0.98) # x2_Ns is not a variable (fixed at 0.98)

    
    model.reflux_constraint = Constraint(expr = model.L[1] == Reflux * Dist)

    # =============================================================================
    # Constraints
    # =============================================================================

    # Composition normalization (excludes stages 1 and Ns due to fixations)
    def norm_x_rule(model, j):
        if j == 1 or j == Ns:
            return Constraint.Skip
        return sum(model.x[i, j] for i in model.components) == 1
    model.norm_x_eq = Constraint(model.stages, rule=norm_x_rule)

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
            return (model.V[j+1] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]) 
                    == model.L[j] * model.x[i, j] + Dist * model.x[i, j])
        elif j == Ns:  # Reboiler
            return (model.L[j-1] * model.x[i, j-1] 
                    == model.V[j] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                    + Bott * model.x[i, j])
        else:  # Internal stages
            if j == Nf:
                return (model.L[j] * model.x[i, j] 
                        + model.V[j] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                        == model.L[j-1] * model.x[i, j-1] 
                        + model.V[j+1] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]) + 
                        Feed * z_feed)
            else:
                return (model.L[j] * model.x[i, j] 
                        + model.V[j] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                        == model.L[j-1] * model.x[i, j-1] 
                        + model.V[j+1] * (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]))
    model.mass_balance_component_eq = Constraint(model.components, model.stages, rule=mass_balance_component)

    # =============================================================================
    # Energy balances 
    # =============================================================================

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
    
    # Energy balances
    def energy_balance(model, j):
        if j == 1:  # Condenser
            return model.V[j+1] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ) == (model.L[j] + Dist) * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.Qc
        
        elif j == Ns:  # Reboiler
            return model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.Qr == model.V[j] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) + Bott * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        )
        
        else:  # Internal stages
            if j == Nf:
                return (model.L[j] * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.V[j] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) == 
                        model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.V[j+1] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ) +  Feed * H_feed)
            
            else:
                return (model.L[j] * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.V[j] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j] + kk[i, 3])))/Pr) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) == 
                        model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.V[j+1] * sum(
                (((exp(kk[i, 1] - kk[i, 2]/(model.T[j+1] + kk[i, 3])))/Pr) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ))
    model.energy_balance = Constraint(model.stages, rule=energy_balance)


    # Objective function
    def objective_rule(model):
        return model.Qr
    model.obj = Objective(rule=objective_rule, sense=minimize)


    # =============================================================================
    # Solve the model
    # =============================================================================
    solver = SolverFactory('gams')
    solver.options['solver'] = 'conopt'
    results = solver.solve(model)


    # =============================================================================
    # Collect and return results
    # =============================================================================
    # Check if solution was found
    if results.solver.termination_condition == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
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
