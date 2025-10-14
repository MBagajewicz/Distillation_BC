from pyomo.environ import *
import numpy as np
from problem_data import Ns, min_interval_width
from calculate_hat_discretized import hat_variable
from resolve_LB import solve_LB

def perform_bound_contraction(bounds, Card_var, results_LB, results_UB, Fobj_UB, Tol):
    """Executes the Bound Contraction algorithm"""
    var_UB = extract_UB_solutions(results_UB)
    binary_values = extract_binary_values(results_LB)
    
    # Dictionary to count how many stages did not undergo contraction for each variable
    aux_var = {'L': 0, 'V': 0, 'T': 0}
    
    for var_name in ['L', 'V', 'T']:
        valid_stages = get_valid_stages(var_name)
        
        for stage in valid_stages:
            if should_skip_stage(var_name, stage):
                continue
                
            current_width = get_interval_width(bounds, var_name, stage)
            if current_width < min_interval_width:
                print(f"Interval for {var_name} at stage {stage+1} is already too narrow ({current_width:.6f}). Do not contract.")
                continue
            
            # Performs bound contraction for this variable and stage
            contraction_occurred = contract_bounds(bounds, Card_var, var_name, stage, 
                                                 var_UB, binary_values, Fobj_UB, Tol)
            
            # If no contraction occurred, increment the counter
            if not contraction_occurred:
                aux_var[var_name] += 1
    
    return aux_var

def extract_UB_solutions(results_UB):
    """Extracts UB solutions for use in Bound Contraction"""
    return {
        'T': results_UB['T'],
        'L': results_UB['L'],
        'V': results_UB['V'],
    }

def extract_binary_values(results_LB):
    """Extracts binary values from the LB solution"""
    return {
        'T': results_LB['lambda'],
        'L': results_LB['omega'],
        'V': results_LB['rho']
    }

def get_valid_stages(var_name):
    """Returns valid stages for a variable"""
    if var_name == 'T':
        return range(Ns)  # All stages for temperature
    elif var_name == 'L':
        return range(Ns - 1)  # Stages 1 to 16 for liquid
    elif var_name == 'V':
        return range(1, Ns)  # Stages 2 to 17 for vapor
    return []

def should_skip_stage(var_name, stage):
    """Checks if a stage should be skipped"""
    if var_name == 'V' and stage == 0:  # V not defined in stage 1
        return True
    if var_name == 'L' and stage == Ns - 1:  # L not defined in stage 17
        return True
    return False

def get_interval_width(bounds, var_name, stage):
    """Calculates the current interval width"""
    lower = bounds[var_name]['lower'][stage]
    upper = bounds[var_name]['upper'][stage]
    return upper - lower

def contract_bounds(bounds, Card_var, var_name, stage, var_UB, binary_values, Fobj_UB, Tol):
    """Performs bound contraction for a specific variable and stage"""
    
    # Saves original bounds
    orig_lower = bounds[var_name]['lower'][stage]
    orig_upper = bounds[var_name]['upper'][stage]
    
    # Gets current partition
    lower = orig_lower
    upper = orig_upper
    card = Card_var[var_name]
    points = [hat_variable(i, card, [lower, upper]) for i in range(1, card + 1)]
    
    # Gets UB value for this variable and stage
    ub_val = var_UB[var_name][stage]
    
    # Gets binary vector for this stage
    bin_vec = binary_values.get(var_name, {})
    lambda_stage_value = bin_vec.get(stage + 1, None)
    
    # Determines the prohibited index
    prohibited_index_idx = find_prohibited_index(lambda_stage_value, card, Tol)
    
    if prohibited_index_idx is None:
        return False  # Not enough information, no contraction
    
    # Calculates distances to determine which side to contract
    P_initial = points[0]
    P_final = points[-1]
    dist_initial = abs(P_initial - ub_val)
    dist_final = abs(P_final - ub_val)
    
    # Decides which side to contract
    if dist_initial <= dist_final:
        # Eliminates extremes near the beginning
        if (card - 2) < len(points) and (card - 2) >= 0:
            bounds[var_name]['lower'][stage] = points[card - 2]
        else:
            bounds[var_name]['lower'][stage] = points[1] if len(points) > 1 else points[0]
    else:
        # Eliminates extremes near the end
        bounds[var_name]['upper'][stage] = points[1] if len(points) > 1 else points[-1]
    
    # Solves LB with new bounds
    print(f"Executing LB for {var_name} at stage {stage+1} with new bounds: [{bounds[var_name]['lower'][stage]:.6f}, {bounds[var_name]['upper'][stage]:.6f}]")
    
    results_LB_new = solve_LB(bounds, Card_var)
    Fobj_LB_new = results_LB_new.get('objective_value', None)
    term_LB_new = results_LB_new.get('termination_condition', None)
    
    # Checks optimality conditions
    def is_optimal_flag(flag):
        try:
            return ((flag == TerminationCondition.optimal) 
                    or (flag == TerminationCondition.locallyOptimal) 
                    or (str(flag).lower().find('optimal') != -1))
        except:
            return str(flag).lower() in ('optimal', 'locallyoptimal')
    
    def is_infeasible_flag(flag):
        try:
            return ((flag == TerminationCondition.infeasible) 
                    or (str(flag).lower().find('infeasible') != -1))
        except:
            return str(flag).lower() in ('infeasible', 'locallyinfeasible')
    
    contraction_occurred = False
    
    # Decision logic based on LB results
    if Fobj_LB_new is not None and Fobj_UB is not None and (Fobj_LB_new < Fobj_UB):
        # LB is feasible and LB < UB → NO CONTRACTION
        print(f"LB feasible and LB < UB → NO CONTRACTION for {var_name} at stage {stage+1}")
        # Restores original bounds
        bounds[var_name]['lower'][stage] = orig_lower
        bounds[var_name]['upper'][stage] = orig_upper
        contraction_occurred = False
    
    elif is_infeasible_flag(term_LB_new) or (Fobj_LB_new is not None and Fobj_UB is not None and (Fobj_LB_new > Fobj_UB)):
        # LB is infeasible or LB > UB → CONTRACTION (keeps only the prohibited interval)
        print(f"LB infeasible or LB > UB → CONTRACTION for {var_name} at stage {stage+1}")
        
        # Reverts the change and keeps only the prohibited interval
        if dist_initial <= dist_final:
            # Restores lower and adjusts upper to the appropriate point
            bounds[var_name]['lower'][stage] = orig_lower
            if (card - 2) < len(points):
                bounds[var_name]['upper'][stage] = points[card - 2]
        else:
            bounds[var_name]['upper'][stage] = orig_upper
            if 1 < len(points):
                bounds[var_name]['lower'][stage] = points[1]
        
        contraction_occurred = True
    
    else:
        # Unexpected case, restores original bounds
        bounds[var_name]['lower'][stage] = orig_lower
        bounds[var_name]['upper'][stage] = orig_upper
        contraction_occurred = False
    
    return contraction_occurred

def find_prohibited_index(lambda_stage_value, card, Tol):
    """Finds the prohibited interval index based on binary values"""
    if lambda_stage_value is None:
        return None
    
    # If lambda_stage_value is a list (binary vector)
    if isinstance(lambda_stage_value, (list, tuple, np.ndarray)):
        vec = list(lambda_stage_value)
        # Finds positions where lambda != 1 (with tolerance)
        indices = [i + 1 for i, v in enumerate(vec) if (v >= 1 + Tol) or (v <= 1 - Tol)]
        if len(indices) > 0:
            return indices[0]
    else:
        # Case where LB returns only an integer indicating the active partition
        try:
            v = int(lambda_stage_value)
            if 1 <= v <= card:
                return v
        except:
            return None
    
    return None

def check_all_stages_contracted(aux_var):
    """Checks if any variable did not undergo contraction in any stage"""
    for var_name in ['L', 'V', 'T']:
        num_valid_stages = Ns - 1 if var_name in ['L', 'V'] else Ns
        if aux_var[var_name] == num_valid_stages:
            print(f"No contraction occurred for {var_name} in any stage")
            return var_name
    return None

def increase_partition(Card_var, var_name):
    """Increases the number of partitions for a variable"""
    Card_var[var_name] += 1
    print(f"Increased partitions for {var_name}: Card_{var_name} = {Card_var[var_name]}")
    return Card_var[var_name]
