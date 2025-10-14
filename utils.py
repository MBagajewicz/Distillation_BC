from pyomo.environ import *
import numpy as np
from problem_data import Ns, Nf, Tol


"""
    The utils.py file is a utilities module:
    a collection of helper functions that are used in
    various parts of the Bound Contraction algorithm.

    Main function categories in utils.py:

    1. Initialization and Configuration
        initialize_bounds(): Defines initial variable bounds
        get_variable_cardinality(): Returns initial cardinality

    2. Calculations and Verifications
        calculate_gap(): Calculates convergence gap
        check_convergence(): Checks if algorithm converged
        check_bounds_consistency(): Validates bounds consistency

    3. Results Handling
        create_UB_initialization(): Prepares initialization for UB
        is_optimal_flag(), is_infeasible_flag(): Check solver status

    4. Visualization and Logging
        print_*() functions: Format and display results
        print_detailed_*_results(): Show detailed results by stage

    5. History Management
        create_convergence_history(): Structure for history
        update_convergence_history(): Updates history

    6. General Utilities
        format_scientific(): Formats numbers
        calculate_statistics(): Calculates statistics
        get_stage_type(): Identifies stage type
"""

def initialize_bounds():
    """Initialize variable bounds"""
    T_upper = [385] * Ns  # Maximum temperature: 385K in all stages
    T_lower = [350] * Ns  # Minimum temperature: 350K in all stages
    L_upper = [190] * Ns  # Maximum liquid flow rate: 190 kmol/h in all stages
    L_lower = [50] * Ns   # Minimum liquid flow rate: 50 kmol/h in all stages
    V_upper = [140] * Ns  # Maximum vapor flow rate: 140 kmol/h in all stages
    V_lower = [120] * Ns  # Minimum vapor flow rate: 120 kmol/h in all stages
    
    return {
        'L': {'upper': L_upper, 'lower': L_lower},
        'V': {'upper': V_upper, 'lower': V_lower},
        'T': {'upper': T_upper, 'lower': T_lower},
    }

def calculate_gap(UB, LB):
    """Calculate relative convergence gap"""
    if UB == 0:
        return float('inf')
    return abs((UB - LB) / UB)

def create_UB_initialization(results_LB):
    """Create initialization vector for UB from LB solution"""
    return {
        'T': results_LB['T'],
        'L': results_LB['L'], 
        'V': results_LB['V'],
        'x': results_LB['x'],
        'Qc': results_LB['Qc'],
        'Qr': results_LB['Qr']
    }

def is_optimal_flag(flag):
    """Check if a solver converged to an optimal solution (global or local)"""
    try:
        return ((flag == TerminationCondition.optimal) 
                or (flag == TerminationCondition.locallyOptimal) 
                or (str(flag).lower().find('optimal') != -1))
    except:
        return str(flag).lower() in ('optimal', 'locallyoptimal')

def is_infeasible_flag(flag):
    """Check if the problem is infeasible (no feasible solution exists)"""
    try:
        return ((flag == TerminationCondition.infeasible) 
                or (str(flag).lower().find('infeasible') != -1))
    except:
        return str(flag).lower() in ('infeasible', 'locallyinfeasible')

def check_convergence(gap, tolerance):
    """Check if the algorithm converged based on the gap"""
    return gap <= tolerance

def print_iteration_header(iteration):
    """Print iteration header"""
    message = f"\n{'+' * 100}\nIteration {iteration}\n{'+' * 100}\n"
    print(message)

def print_section_header(section_name):
    """Print section header"""
    message = f"\n{'-' * 100}\n{section_name}\n{'-' * 100}\n"
    print(message)

def print_iteration_results(iteration, Fobj_LB, Fobj_UB, gap, tolerance):
    """Print iteration results"""
    message = (
        f"\n{'=' * 100}\n"
        f"ITERATION: {iteration} - RESULTS\n"
        f"{'=' * 100}\n"
        f"LB: {Fobj_LB:.6f}\n"
        f"UB: {Fobj_UB:.6f}\n"
        f"Gap: {gap:.6f} (Tol: {tolerance})\n"
        f"{'=' * 100}"
    )
    print(message)

def print_LB_results(results_LB):
    """Print Lower Bound results"""
    if results_LB['termination_condition'] == TerminationCondition.optimal:
        message = (
            f"\n{'=' * 50}\n"
            f"RESULTS LB\n"
            f"{'=' * 50}\n"
            f"status = {results_LB['termination_condition']}\n"
            f"Fobj_LB = {results_LB['objective_value']:.6f}\n"
            f"Reboiler duty (Qr): {results_LB['Qr']:.2f} kJ/h\n"
            f"Condenser duty (Qc): {results_LB['Qc']:.2f} kJ/h"
        )
        print(message)
    else:
        message = f"The solver did not converge to an optimal solution.\nTermination condition: {results_LB['termination_condition']}"
        print(message)

def print_UB_results(results_UB):
    """Print Upper Bound results"""
    if is_optimal_flag(results_UB['termination_condition']):
        message = (
            f"\n{'=' * 50}\n"
            f"RESULTS UB\n"
            f"{'=' * 50}\n"
            f"Reboiler duty (Qr): {results_UB['Qr']:.2f} kJ/h\n"
            f"Condenser duty (Qc): {results_UB['Qc']:.2f} kJ/h"
        )
        print(message)
    else:
        message = f"The solver did not converge to an optimal solution.\nTermination condition: {results_UB['termination_condition']}"
        print(message)

def validate_bounds_structure(bounds):
    """Validate the structure of the bounds dictionary"""
    required_keys = ['L', 'V', 'T']
    required_subkeys = ['lower', 'upper']
    
    for var_name in required_keys:
        if var_name not in bounds:
            raise ValueError(f"Missing variable '{var_name}' in bounds dictionary")
        
        for subkey in required_subkeys:
            if subkey not in bounds[var_name]:
                raise ValueError(f"Missing '{subkey}' for variable '{var_name}' in bounds dictionary")
            
            if len(bounds[var_name][subkey]) != Ns:
                raise ValueError(f"Bounds for '{var_name}.{subkey}' should have length {Ns}")

def get_variable_cardinality():
    """Return the initial cardinality of variables"""
    from problem_data import Card_L, Card_V, Card_T
    return {'L': Card_L, 'V': Card_V, 'T': Card_T}

def format_scientific(value, precision=6):
    """Format a value in readable scientific notation"""
    if value == 0:
        return "0.0"
    
    if abs(value) >= 1e6 or abs(value) <= 1e-3:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def calculate_statistics(values):
    """Calculate basic statistics from a list of values"""
    if not values:
        return {}
    
    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    
    if not valid_values:
        return {}
    
    return {
        'min': min(valid_values),
        'max': max(valid_values),
        'mean': np.mean(valid_values),
        'std': np.std(valid_values)
    }

def check_bounds_consistency(bounds):
    """Check bounds consistency (lower <= upper)"""
    for var_name in ['L', 'V', 'T']:
        for i in range(len(bounds[var_name]['lower'])):
            lower = bounds[var_name]['lower'][i]
            upper = bounds[var_name]['upper'][i]
            
            if lower > upper:
                raise ValueError(f"Inconsistent bounds for {var_name}[{i}]: lower({lower}) > upper({upper})")

def get_stage_type(stage):
    """Return stage type based on stage number"""
    if stage == 1:
        return "Condenser"
    elif stage == Ns:
        return "Reboiler"
    elif stage == Nf:
        return "Feed Stage"
    else:
        return "Internal Stage"

def create_convergence_history():
    """Create structure for convergence history"""
    return {
        'iteration': [],
        'LB': [],
        'UB': [],
        'gap': [],
        'time': []
    }

def update_convergence_history(history, iteration, LB, UB, gap, elapsed_time):
    """Update convergence history"""
    history['iteration'].append(iteration)
    history['LB'].append(LB)
    history['UB'].append(UB)
    history['gap'].append(gap)
    history['time'].append(elapsed_time)

def print_convergence_summary(history):
    """Print algorithm convergence summary"""
    if not history['iteration']:
        return
    
    final_iter = history['iteration'][-1]
    final_gap = history['gap'][-1]
    total_time = history['time'][-1] if history['time'] else 0
    
    message = (
        f"\n{'#' * 60}\n"
        f"CONVERGENCE SUMMARY\n"
        f"{'#' * 60}\n"
        f"Final Iteration: {final_iter}\n"
        f"Final Gap: {final_gap:.6f}\n"
        f"Total Time: {total_time:.2f} seconds\n"
        f"Final LB: {history['LB'][-1]:.6f}\n"
        f"Final UB: {history['UB'][-1]:.6f}\n"
        f"{'#' * 60}"
    )
    
    print(message)

def save_bounds_to_file(bounds, filename):
    """Save current bounds to a file"""
    import json
    import numpy as np
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_bounds = {}
    for var_name in bounds:
        serializable_bounds[var_name] = {
            'lower': [float(x) for x in bounds[var_name]['lower']],
            'upper': [float(x) for x in bounds[var_name]['upper']]
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_bounds, f, indent=2)

def load_bounds_from_file(filename):
    """Load bounds from a file"""
    import json
    
    with open(filename, 'r') as f:
        serializable_bounds = json.load(f)
    
    return serializable_bounds

def get_memory_usage():
    """Return current memory usage (approx.)"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def print_detailed_LB_results(results_LB):
    """Print detailed LB results including variables by stage"""
    if results_LB['termination_condition'] != TerminationCondition.optimal:
        return
    
    print("\nDETAILED LB RESULTS BY STAGE:")
    print("-" * 80)
    
    # Temperatures
    print("\nTEMPERATURES (K):")
    for j, T_val in enumerate(results_LB['T'], start=1):
        stage_type = get_stage_type(j)
        print(f"  Stage {j:2d} ({stage_type:12s}): {T_val:.2f} K")
    
    # Liquid flow rates
    print("\nLIQUID FLOW RATES (kmol/h):")
    for j, L_val in enumerate(results_LB['L'], start=1):
        if not np.isnan(L_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {L_val:.2f} kmol/h")
    
    # Vapor flow rates
    print("\nVAPOR FLOW RATES (kmol/h):")
    for j, V_val in enumerate(results_LB['V'], start=1):
        if not np.isnan(V_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {V_val:.2f} kmol/h")
    
    # Compositions
    if 'x' in results_LB:
        print("\nBENZENE COMPOSITIONS (x1):")
        for j, x1_val in enumerate(results_LB['x'][1], start=1):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {x1_val:.6f}")

def print_detailed_UB_results(results_UB):
    """Print detailed UB results including variables by stage"""
    if not is_optimal_flag(results_UB['termination_condition']):
        return
    
    print("\nDETAILED UB RESULTS BY STAGE:")
    print("-" * 80)
    
    # Temperatures
    print("\nTEMPERATURES (K):")
    for j, T_val in enumerate(results_UB['T'], start=1):
        stage_type = get_stage_type(j)
        print(f"  Stage {j:2d} ({stage_type:12s}): {T_val:.2f} K")
    
    # Liquid flow rates
    print("\nLIQUID FLOW RATES (kmol/h):")
    for j, L_val in enumerate(results_UB['L'], start=1):
        if not np.isnan(L_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {L_val:.2f} kmol/h")
    
    # Vapor flow rates
    print("\nVAPOR FLOW RATES (kmol/h):")
    for j, V_val in enumerate(results_UB['V'], start=1):
        if not np.isnan(V_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {V_val:.2f} kmol/h")
    
    # Compositions
    if 'x' in results_UB:
        print("\nBENZENE COMPOSITIONS (x1):")
        for j, x1_val in enumerate(results_UB['x'][1], start=1):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {x1_val:.6f}")

def print_bounds_summary(bounds, iteration=None):
    """Print summary of current bounds"""
    if iteration:
        print(f"\nBOUNDS SUMMARY - Iteration {iteration}")
    else:
        print(f"\nBOUNDS SUMMARY")
    print("-" * 60)
    
    for var_name in ['T', 'L', 'V']:
        print(f"\n{var_name}:")
        valid_stages = Ns if var_name == 'T' else (Ns - 1)
        
        for stage in range(valid_stages):
            if (var_name == 'V' and stage == 0) or (var_name == 'L' and stage == Ns - 1):
                continue
                
            lower = bounds[var_name]['lower'][stage]
            upper = bounds[var_name]['upper'][stage]
            width = upper - lower
            stage_type = get_stage_type(stage + 1)
            
            print(f"  Stage {stage+1:2d} ({stage_type:12s}): [{lower:8.4f}, {upper:8.4f}] (width: {width:8.4f})")

def print_partition_info(bounds, Card_var):
    """Print information about current partitioning"""
    print("\nPARTITION INFORMATION:")
    print("-" * 60)
    
    for var_name in ['T', 'L', 'V']:
        card = Card_var[var_name]
        print(f"\n{var_name}: Card = {card}")
        
        valid_stages = Ns if var_name == 'T' else (Ns - 1)
        
        for stage in range(valid_stages):
            if (var_name == 'V' and stage == 0) or (var_name == 'L' and stage == Ns - 1):
                continue
                
            lower = bounds[var_name]['lower'][stage]
            upper = bounds[var_name]['upper'][stage]
            
            # Calculate discretization points
            from calculate_hat_discretized import hat_variable
            points = [hat_variable(i, card, [lower, upper]) for i in range(1, card + 1)]
            
            print(f"  Stage {stage+1}: {points}")
