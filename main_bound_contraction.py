from pyomo.environ import *
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from problem_data import (Ns, Nf, Tol, 
                          Card_L, Card_V, Card_T,
                          min_interval_width) 

from aspen_data import (T_aspen, L_aspen, V_aspen, 
                        x_Benz_aspen, x_Tol_aspen, 
                        Qcond_aspen, Qreb_aspen)  

 
from resolve_LB import solve_LB
from resolve_UB import solve_UB
from calculate_chapel_variable import chapel_variable

########################################################################################################
# Set the output file
output_file = open('OUTPUT.txt', 'w+', encoding='utf-8')
start_time = time.time()  # Calculates the beginning of time
########################################################################################################

# ======================================================================================================
# BOUND CONTRACTION ALGORITHM - VARIABLES AND PARAMETERS
# ======================================================================================================

# Partition variables: Variables discretized into intervals. ['L', 'V', 'T']
# Bound Contraction variables: Variables whose bounds will be contracted. ['L', 'V', 'T'] 
# γ: Convergence tolerance. (Tol from problem_data)
# Card - 1: Initial number of partition intervals. (Card_L, Card_V, Card_T from problem_data)

# Create a dictionary to store the number of partitions by variable
Card_var = {'L': Card_L, 'V': Card_V, 'T': Card_T}

# ======================================================================================================
# 1. INITIALIZATION
# ======================================================================================================

# 1.1 Set initial bounds for all variables.
T_upper = [385] * Ns # Maximum temperature: 385K in all stages
T_lower = [350] * Ns # Minimum temperature: 350K in all stages
L_upper = [190] * Ns # Maximum liquid flow rate: 190 kmol/h in all stages
L_lower = [50] * Ns # Minimum liquid flow rate: 50 kmol/h in all stages
V_upper = [140] * Ns # Maximum steam flow rate: 140 kmol/h in all stages
V_lower = [120] * Ns # Minimum steam flow rate: 120 kmol/h in all stages

bounds  = {
    'L': {'upper': L_upper, 'lower': L_lower},
    'V': {'upper': V_upper, 'lower': V_lower},
    'T': {'upper': T_upper, 'lower': T_lower},
}

########################################################################################################
# Check whether a solver has converged to an optimal solution (global or local).
def is_optimal_flag(flag):
    """Compatibility: Accepts string or TerminationCondition enum."""
    try:
        return ((flag == TerminationCondition.optimal) 
                or (flag == TerminationCondition.locallyOptimal) 
                or (str(flag).lower().find('optimal')!=-1))
    except:
        return str(flag).lower() in ('optimal', 'locallyoptimal', 'locallyoptimal')

# Check whether the problem is unfeasible (has no feasible solution).
def is_infeasible_flag(flag):
    try:
        return ((flag == TerminationCondition.infeasible) 
                or (str(flag).lower().find('infeasible')!=-1))
    except:
        return str(flag).lower() in ('infeasible', 'locallyinfeasible')
########################################################################################################

###################################################
iter = 1

print(" ", file = output_file)
print("+" * 100, file = output_file)
print(f"Iteration {iter}", file = output_file)  
print("+" * 100, file = output_file)
print(" ", file = output_file)
##################################################

print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Partition all variables", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)

# --------------------------------------------------------------------------------------------------- # 
# 1.2 Partition selected variables into Card intervals.
for var_name in bounds.keys(): 
    # ['L', 'V', 'T'] - Partition variables

    print(f"Variable {var_name}:", file=output_file)
    print("-" * 50, file=output_file)
    
    # Check if this variable has the expected structure
    if 'lower' not in bounds[var_name] or 'upper' not in bounds[var_name]:
        print(f"  Incomplete data structure for {var_name}", file=output_file) 
        continue
    
    # Determine how many stages this variable actually has
    num_stages = min(len(bounds[var_name]['lower']), 
                     len(bounds[var_name]['upper']))
    
    for stage in range(num_stages):
        # Skip specific invalid combinations if needed
        if (var_name == 'V' and stage == 0) or (var_name == 'L' and stage == Ns - 1):
            continue
        try:
            # Get the limits for this stage
            lower = bounds[var_name]['lower'][stage]
            upper = bounds[var_name]['upper'][stage]
            
            # Get the number of partitions for this variable
            card = Card_var[var_name]
            
            # Calculate the discretization points
            points = []
            for i in range(1, card + 1):
                value = chapel_variable(i, card, [lower, upper])
                points.append(value)
            
            # Formats the points for printing
            formatted_points = [f"{p:.7f}" for p in points]
            
            # Print the results
            print(f"Stage {stage+1}: [{', '.join(formatted_points)}]", file=output_file)
        except (IndexError, KeyError, TypeError) as e:
            print(f"  Error processing {var_name} at stage {stage+1}: {str(e)}", file=output_file)
            continue
    
    print("", file=output_file) 
# ------------------------------------------------------------------------------------------------------ #

print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Solve the lower bound model (LB)", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)

# 1.3 Construct the lower bound (LB) model using one of the techniques (e.g., DPP1)
# 1.4 Solve the LB model.
results_LB = solve_LB(bounds, Card_var)
Fobj_LB = results_LB['objective_value']

# Check if the solution was optimal
if results_LB['termination_condition'] == TerminationCondition.optimal:
    print("="*50, file = output_file)
    print("RESULTS LB", file = output_file)
    print("="*50, file = output_file)

    print(" ", file = output_file)    
    print(f"status = {results_LB['termination_condition']}", file = output_file)
    print(" ", file = output_file)
    print(f"Fobj_LB = {Fobj_LB:.6f}", file = output_file)
    print(" ", file = output_file)
        
    # Print objective and heat duties
    print(f"\nReboiler duty (Qr): {results_LB['Qr']:.2f} kJ/h", file = output_file)
    print(f"Condenser duty (Qc): {results_LB['Qc']:.2f} kJ/h", file = output_file)
    
    # Print temperatures
    print("\nTemperatures by stage:", file = output_file)
    for j, T_val in enumerate(results_LB['T'], start=1):
        print(f"Stage {j}: {T_val:.2f} K", file = output_file)
    
    # Print liquid flow rates
    print("\nLiquid flow rates by stage:", file = output_file)
    for j, L_val in enumerate(results_LB['L'], start=1):
        if not np.isnan(L_val):
            print(f"Stage {j}: {L_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Stage {j}: Not defined", file = output_file)
    
    # Print vapor flow rates
    print("\nVapor flow rates by stage:", file = output_file)
    for j, V_val in enumerate(results_LB['V'], start=1):
        if not np.isnan(V_val):
            print(f"Stage {j}: {V_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Stage {j}: Not defined", file = output_file)
    
    # Print compositions
    print("\nBenzene compositions (x1) by stage:", file = output_file)
    for j, x1_val in enumerate(results_LB['x'][1], start=1):
        print(f"Stage {j}: {x1_val:.6f}", file = output_file)
    
    print("\nToluene compositions (x2) by stage:", file = output_file)
    for j, x2_val in enumerate(results_LB['x'][2], start=1):
        print(f"Stage {j}: {x2_val:.6f}", file = output_file)

    print("\nBinary variables lambda:", file = output_file)
    for j in results_LB['lambda']:
         print(f"Stage {j}: {results_LB['lambda'][j]}", file = output_file)

    print("\nBinary variables omega:", file = output_file)
    for j in results_LB['omega']:
         print(f"Stage {j}: {results_LB['omega'][j]}", file = output_file)   
    
    print("\nBinary variables rho:", file = output_file)
    for j in results_LB['rho']:
         print(f"Stage {j}: {results_LB['rho'][j]}", file = output_file)    
    
else:
    print("The solver did not converge to an optimal solution.", file = output_file)
    print(f"Termination condition: {results_LB['termination_condition']}", file = output_file)

print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Solve the UB model", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)

# 1.5 Use the LB solution as a starting point to solve the original NLP problem and obtain the upper bound (UB).
# Create initialization vector for UB
UB_init = {
    'T': results_LB['T'],  # List of temperatures by stage
    'L': results_LB['L'],  # List of liquid flow rates by stage
    'V': results_LB['V'],  # List of vapor flow rates by stage
    'x': results_LB['x'],  # Dictionary of compositions by component
    'Qc': results_LB['Qc'],  # Condenser duty value
    'Qr': results_LB['Qr']   # Reboiler duty value
}

results_UB = solve_UB(bounds, UB_init)
Fobj_UB = results_UB['objective_value']

# Check if the solution was optimal
if results_UB['termination_condition'] == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
    
    print("="*50, file = output_file)
    print("RESULTS UB", file = output_file)
    print("="*50, file = output_file)
    
    # Print objective and heat duties
    print(f"\nReboiler duty (Qr): {results_UB['Qr']:.2f} kJ/h", file = output_file)
    print(f"Condenser duty (Qc): {results_UB['Qc']:.2f} kJ/h", file = output_file)
    
    # Print temperatures
    print("\nTemperatures by stage:", file = output_file)
    for j, T_val in enumerate(results_UB['T'], start=1):
        print(f"Stage {j}: {T_val:.2f} K", file = output_file)
    
    # Print liquid flow rates
    print("\nLiquid flow rates by stage:", file = output_file)
    for j, L_val in enumerate(results_UB['L'], start=1):
        if not np.isnan(L_val):
            print(f"Stage {j}: {L_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Stage {j}: Not defined", file = output_file)
    
    # Print vapor flow rates
    print("\nVapor flow rates by stage:", file = output_file)
    for j, V_val in enumerate(results_UB['V'], start=1):
        if not np.isnan(V_val):
            print(f"Stage {j}: {V_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Stage {j}: Not defined", file = output_file)
    
    # Print compositions
    print("\nBenzene compositions (x1) by stage:", file = output_file)
    for j, x1_val in enumerate(results_UB['x'][1], start=1):
        print(f"Stage {j}: {x1_val:.6f}", file = output_file)
    
    print("\nToluene compositions (x2) by stage:", file = output_file)
    for j, x2_val in enumerate(results_UB['x'][2], start=1):
        print(f"Stage {j}: {x2_val:.6f}", file = output_file)

else:
    print("The solver did not converge to an optimal solution.", file = output_file)
    print(f"Termination condition: {results_UB['termination_condition']}", file = output_file)

"""
    Maps the results of the LB (Lower Bound) and UB (Upper Bound) solutions
    for use in the Bound Contraction algorithm.
"""
var_UB = {
    'T': results_UB['T'],
    'L': results_UB['L'],
    'V': results_UB['V'],
}

binary_values = {
    'T': results_LB['lambda'],
    'L': results_LB['omega'],
    'V': results_LB['rho']
}

# ======================================================================================================
# 2. MAIN LOOP
# ======================================================================================================

##################################################
# 2.1 Calculate gap = (UB - LB) / UB
##################################################
gap = abs((Fobj_UB - Fobj_LB) / Fobj_UB) if Fobj_UB != 0 else float('inf')
print(" ", file=output_file)
print("=" * 100, file=output_file)
print(f"ITERATION: {iter} - RESULTS", file=output_file)
print("=" * 100, file=output_file)
print(f"LB: {Fobj_LB:.6f}", file=output_file)
print(f"UB: {Fobj_UB:.6f}", file=output_file)
print(f"Gap: {gap:.6f} (Tol: {Tol})", file=output_file)
print("=" * 100, file=output_file)

# 2.2 If gap ≤ ε: Return the optimal solution. (Loop condition checks this)
# 2.3 Else: Perform Bound Contraction.
while (gap >= Tol):
    # Dictionary to count how many stages did not undergo contraction for each variable
    aux_var = {'L': 0, 'V': 0, 'T': 0}

    # ==================================================================================================
    # 2.3.1 Select a variable from the Bound Contraction variables set.
    # ==================================================================================================
    for var_name in ['L', 'V', 'T']:  # Bound Contraction variables

        if var_name not in bounds or var_name not in var_UB:
            continue  # Skip this variable if it doesn't exist

        print(" ", file = output_file)
        print("-" * 100, file = output_file)
        print(f"Select a variable from the set of concave variables.", file = output_file)
        print("-" * 100, file = output_file)

         # Determine valid stages for this variable
        valid_stages = min(
            len(bounds[var_name]['lower']),
            len(bounds[var_name]['upper']),
            len(var_UB[var_name])
        )

        for stage in range(valid_stages):

            if (var_name == 'V' and stage == 0) or (var_name == 'L' and stage == Ns - 1):
                continue  # Skip specifically defined invalid combinations
            

            # Check if the interval is already too narrow BEFORE trying to contract
            current_width = bounds[var_name]['upper'][stage] - bounds[var_name]['lower'][stage]
            if current_width < min_interval_width:
                print(f"Interval for {var_name} at stage {stage+1} is already too narrow ({current_width:.6f}). Do not contract.", file=output_file)
                continue  # Skip to next stage without contracting        

            print(f"\nVariable chosen:  {var_name} at stage {stage+1}", file=output_file)
            
            # Get partitioning of current stage
            lower = bounds[var_name]['lower'][stage]
            upper = bounds[var_name]['upper'][stage]
            
            # Get the number of partitions for this variable
            card = Card_var[var_name]
            points = [chapel_variable(i, card, [lower, upper]) for i in range(1, card+1)]

            # Retrieve binary vector for this stage (if exists)
            bin_vec = binary_values.get(var_name, {})
            lambda_stage_value = None
            if isinstance(bin_vec, dict):
                # try indexing by stage+1
                lambda_stage_value = bin_vec.get(stage+1, None)

            # If there is no binary value, skip (we don't have information about which interval is prohibited)
            if lambda_stage_value is None:
                continue

            
            prohibited_index_idx = None
            # If bin_vec[stage+1] is a list (e.g. list of lambdas by partition), we detect the index with value ~=1
            if isinstance(bin_vec.get(stage+1, None), (list, tuple, np.ndarray)):
                vec = list(bin_vec[stage+1])
                # find positions where lambda != 1 (with tol)
                indices = [i+1 for i, v in enumerate(vec) if (v >= 1+Tol) or (v <= 1-Tol)]
                if len(indices) > 0:
                    prohibited_index_idx = indices[0]
            else:
                # Case LB returns only an integer indicating the active partition:
                # If the structure stored the prohibited index directly
                try:
                    # if it's integer (1..Card), we consider
                    v = int(bin_vec.get(stage+1))
                    if 1 <= v <= card:
                        prohibited_index_idx = v
                except:
                    prohibited_index_idx = None

            if prohibited_index_idx is None:
                # without sufficient information, skip this stage
                continue

            P_initial = points[0]
            P_final = points[-1]
            # current UB value for this variable/stage
            ub_val = var_UB[var_name][stage]

            # if ub_val is None, skip
            if ub_val is None:
                continue

            dist_initial = abs(P_initial - ub_val)
            dist_final = abs(P_final - ub_val)

            aux_L = None
            aux_U = None

            # save original to restore if necessary
            orig_lower = bounds[var_name]['lower'][stage]
            orig_upper = bounds[var_name]['upper'][stage]

            # Determine which side to contract
            if dist_initial <= dist_final:
                # eliminate extremes close to the beginning: move lower to point corresponding to D_line-th partition
                aux_L = orig_lower
                # results[D_line-1] (0-based)
                if (card-2) < len(points) and (card-2) >= 0:
                    bounds[var_name]['lower'][stage] = points[card-2]
                else:
                    # fallback: move to second point
                    bounds[var_name]['lower'][stage] = points[1] if len(points) > 1 else points[0]
            else:
                # eliminate extremes close to the end
                aux_U = orig_upper
                # results[1] (0-based index 1 is the second point)
                bounds[var_name]['upper'][stage] = points[1] if len(points) > 1 else points[-1]

            
            print(" ", file = output_file)
            print("-" * 100, file=output_file)
            # ==========================================================================================
            # 2.3.2 Run the LB model while prohibiting the interval containing the UB solution
            # ==========================================================================================
            print("Execute the LB model, prohibiting the interval that contains", file=output_file)
            print("the UB solution of the variable chosen in step 3.1. (and its adjacent ones)", file=output_file)
            print("-" * 100, file=output_file)
            print(" ", file=output_file)

            print(f"\nExecute LB on interval {var_name} stage {stage+1} = [{bounds[var_name]['lower'][stage]:.6f}, {bounds[var_name]['upper'][stage]:.6f}]", file=output_file)
            print(f"UB solution: {ub_val :.6f}", file=output_file)
            
            results_LB = solve_LB(bounds, Card_var)
            Fobj_LB = results_LB.get('objective_value', None)
            term_LB = results_LB.get('termination_condition', None)

            if is_optimal_flag(term_LB):
                print(f"Fobj_LB = {Fobj_LB:.7f}", file=output_file)
                print(f"(Fobj_LB > UB) = {Fobj_LB > Fobj_UB}", file=output_file)

            print(f"status LB: {term_LB}", file=output_file)

            # ==========================================================================================
            # 2.3.3 Decision Logic
            # ==========================================================================================
            if Fobj_LB is not None and Fobj_UB is not None and (Fobj_LB < Fobj_UB):
                # Do not contract, just maintain the limits
                print("-"*100, file=output_file)
                print("LB is feasible and LB < UB → DO NOT CONTRACT INTERVALS.", file=output_file)
                print(f"Interval preserved = [{orig_lower:.8f}, {orig_upper:.8f}]", file=output_file)
                print("-"*100, file=output_file)

                # restore original limits
                bounds[var_name]['lower'][stage] = orig_lower
                bounds[var_name]['upper'][stage] = orig_upper
                
                # Increment counter for this variable (no contraction occurred)
                aux_var[var_name] += 1
            
            # ==========================================================================================
            # 2.3.3 If LB is infeasible or if LB is feasible but LB > UB:
            # 2.3.3.1 Eliminate all intervals except the prohibited one.
            # ==========================================================================================
            if is_infeasible_flag(term_LB) or (Fobj_LB is not None and Fobj_UB is not None and (Fobj_LB > Fobj_UB)):
                # revert the side we changed and keep the prohibited interval (only this one not eliminated)
                if aux_L is not None:
                    # restore lower and set upper to point D_line (keeping only prohibited interval)
                    bounds[var_name]['lower'][stage] = orig_lower
                    # upper -> point D_line-1
                    if (card-2) < len(points):
                        bounds[var_name]['upper'][stage] = points[card-2]
                else:
                    bounds[var_name]['upper'][stage] = orig_upper
                    if 1 < len(points):
                        bounds[var_name]['lower'][stage] = points[1]
                print("-"*100, file=output_file)
                print("If LB is infeasible or LB is feasible but LB > UB.", file=output_file)
                print("Eliminate all intervals, except the prohibited interval", file=output_file)
                print(f"Interval that was NOT eliminated = [{bounds[var_name]['lower'][stage]:.8f}, {bounds[var_name]['upper'][stage]:.8f}]", file=output_file)
                print("-"*100, file=output_file)

            # 2.3.3.2 Return to Step 2.3.1, selecting another variable. (Automatic via loops)
            # 2.3.3.3 Repeat until all Bound Contraction variables are exhausted. (Automatic via loops)

            print("*" * 50, file = output_file)

    # ==================================================================================================
    # 3. If no intervals are eliminated and the stopping criterion is still unmet, 
    #    increase the number of intervals and restart the process.
    # ==================================================================================================
    for var_name in ['L', 'V', 'T']:
        # Number of valid stages for this variable
        num_valid_stages = Ns - 1 if var_name in ['L', 'V'] else Ns
        
        if aux_var[var_name] == num_valid_stages:
            print(" ", file = output_file)
            print("-" * 100, file = output_file)
            print(f"No interval was eliminated for {var_name} and the stopping criterion was not met", file = output_file)
            print(f"must increase the number of intervals for {var_name} and restart", file = output_file)
                            
            Card_var[var_name] += 1
            print(f"Card_{var_name} = {Card_var[var_name]}", file = output_file)
            print("-" * 100, file = output_file)  

    ####################################################
    iter += 1

    print(" ", file = output_file)
    print("+" * 100, file = output_file)
    print(f"Iteration {iter}", file = output_file)  
    print("+" * 100, file = output_file)
    print(" ", file = output_file)
    ###################################################

    # RESTART PROCESS WITH UPDATED PARAMETERS
    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Partition all variables", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)

    # --------------------------------------------------------------------------------------------------- # 
    # Partition variables again with updated Card values
    for var_name in bounds.keys():
        print(f"Variable {var_name}:", file=output_file)
        print("-" * 50, file=output_file)
        
        # Check if this variable has the expected structure
        if 'lower' not in bounds[var_name] or 'upper' not in bounds[var_name]:
            print(f"  Incomplete data structure for {var_name}", file=output_file)
            continue
        
        # Determine how many stages this variable actually has
        num_stages = min(len(bounds[var_name]['lower']), 
                         len(bounds[var_name]['upper']))
        
        for stage in range(num_stages):
            # Skip specific invalid combinations if needed
            if (var_name == 'V' and stage == 0) or (var_name == 'L' and stage == Ns - 1):
                continue
                
            try:
                # Get the limits for this stage
                lower = bounds[var_name]['lower'][stage]
                upper = bounds[var_name]['upper'][stage]
                
                # Get the number of partitions for this variable
                card = Card_var[var_name]
                
                # Calculate the discretization points
                points = []
                for i in range(1, card + 1):
                    value = chapel_variable(i, card, [lower, upper])
                    points.append(value)
                
                # Format the points for printing
                formatted_points = [f"{p:.7f}" for p in points]
                
                # Print the results
                print(f"Stage {stage+1}: [{', '.join(formatted_points)}]", file=output_file)
            except (IndexError, KeyError, TypeError) as e:
                print(f"  Error processing {var_name} at stage {stage+1}: {str(e)}", file=output_file)
                continue
        
        print("", file=output_file)  # Blank line between variables
    # ------------------------------------------------------------------------------------------------------ #

    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Solve the lower bound model (LB)", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)

    # Solve LB again with updated bounds and partitions
    results_LB = solve_LB(bounds, Card_var)
    Fobj_LB = results_LB['objective_value']

    # Check if the solution was optimal
    if results_LB['termination_condition'] == TerminationCondition.optimal:
        print("="*50, file = output_file)
        print("RESULTS LB", file = output_file)
        print("="*50, file = output_file)

        print(" ", file = output_file)    
        print(f"status = {results_LB['termination_condition']}", file = output_file)
        print(" ", file = output_file)
        print(f"Fobj_LB = {Fobj_LB:.6f}", file = output_file)
        print(" ", file = output_file)
            
        # Print objective and heat duties
        print(f"\nReboiler duty (Qr): {results_LB['Qr']:.2f} kJ/h", file = output_file)
        print(f"Condenser duty (Qc): {results_LB['Qc']:.2f} kJ/h", file = output_file)
        
        # Print temperatures
        print("\nTemperatures by stage:", file = output_file)
        for j, T_val in enumerate(results_LB['T'], start=1):
            print(f"Stage {j}: {T_val:.2f} K", file = output_file)
        
        # Print liquid flow rates
        print("\nLiquid flow rates by stage:", file = output_file)
        for j, L_val in enumerate(results_LB['L'], start=1):
            if not np.isnan(L_val):
                print(f"Stage {j}: {L_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Stage {j}: Not defined", file = output_file)
        
        # Print vapor flow rates
        print("\nVapor flow rates by stage:", file = output_file)
        for j, V_val in enumerate(results_LB['V'], start=1):
            if not np.isnan(V_val):
                print(f"Stage {j}: {V_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Stage {j}: Not defined", file = output_file)
        
        # Print compositions
        print("\nBenzene compositions (x1) by stage:", file = output_file)
        for j, x1_val in enumerate(results_LB['x'][1], start=1):
            print(f"Stage {j}: {x1_val:.6f}", file = output_file)
        
        print("\nToluene compositions (x2) by stage:", file = output_file)
        for j, x2_val in enumerate(results_LB['x'][2], start=1):
            print(f"Stage {j}: {x2_val:.6f}", file = output_file)

        print("\nBinary variables lambda:", file = output_file)
        for j in results_LB['lambda']:
            print(f"Stage {j}: {results_LB['lambda'][j]}", file = output_file)

        print("\nBinary variables omega:", file = output_file)
        for j in results_LB['omega']:
            print(f"Stage {j}: {results_LB['omega'][j]}", file = output_file)   
        
        print("\nBinary variables rho:", file = output_file)
        for j in results_LB['rho']:
            print(f"Stage {j}: {results_LB['rho'][j]}", file = output_file)    
        
    else:
        print("The solver did not converge to an optimal solution.", file = output_file)
        print(f"Termination condition: {results_LB['termination_condition']}", file = output_file)

    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Solve the UB model", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)

    # Create initialization vector for UB
    UB_init = {
        'T': results_LB['T'],  # List of temperatures by stage
        'L': results_LB['L'],  # List of liquid flow rates by stage
        'V': results_LB['V'],  # List of vapor flow rates by stage
        'x': results_LB['x'],  # Dictionary of compositions by component
        'Qc': results_LB['Qc'],  # Condenser duty value
        'Qr': results_LB['Qr']   # Reboiler duty value
    }

    results_UB = solve_UB(bounds, UB_init)

    Fobj_UB = results_UB['objective_value']

    # Check if the solution was optimal
    if results_UB['termination_condition'] == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
        
        print("="*50, file = output_file)
        print("RESULTS UB", file = output_file)
        print("="*50, file = output_file)
        
        # Print objective and heat duties
        print(f"\nReboiler duty (Qr): {results_UB['Qr']:.2f} kJ/h", file = output_file)
        print(f"Condenser duty (Qc): {results_UB['Qc']:.2f} kJ/h", file = output_file)
        
        # Print temperatures
        print("\nTemperatures by stage:", file = output_file)
        for j, T_val in enumerate(results_UB['T'], start=1):
            print(f"Stage {j}: {T_val:.2f} K", file = output_file)
        
        # Print liquid flow rates
        print("\nLiquid flow rates by stage:", file = output_file)
        for j, L_val in enumerate(results_UB['L'], start=1):
            if not np.isnan(L_val):
                print(f"Stage {j}: {L_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Stage {j}: Not defined", file = output_file)
        
        # Print vapor flow rates
        print("\nVapor flow rates by stage:", file = output_file)
        for j, V_val in enumerate(results_UB['V'], start=1):
            if not np.isnan(V_val):
                print(f"Stage {j}: {V_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Stage {j}: Not defined", file = output_file)
        
        # Print compositions
        print("\nBenzene compositions (x1) by stage:", file = output_file)
        for j, x1_val in enumerate(results_UB['x'][1], start=1):
            print(f"Stage {j}: {x1_val:.6f}", file = output_file)
        
        print("\nToluene compositions (x2) by stage:", file = output_file)
        for j, x2_val in enumerate(results_UB['x'][2], start=1):
            print(f"Stage {j}: {x2_val:.6f}", file = output_file)

    else:
        print("The solver did not converge to an optimal solution.", file = output_file)
        print(f"Termination condition: {results_UB['termination_condition']}", file = output_file)

    # Update mappings for next iteration
    var_UB = {
        'T': results_UB['T'],
        'L': results_UB['L'],
        'V': results_UB['V'],
    }

    binary_values = {
        'T': results_LB['lambda'],
        'L': results_LB['omega'],
        'V': results_LB['rho']
    }

    ##################################################
    # Check convergence again
    ##################################################
    gap = abs((Fobj_UB - Fobj_LB) / Fobj_UB) if Fobj_UB != 0 else float('inf')
    print(" ", file=output_file)
    print("=" * 100, file=output_file)
    print(f"ITERATION: {iter} - RESULTS", file=output_file)
    print("=" * 100, file=output_file)
    print(f"LB: {Fobj_LB:.6f}", file=output_file)
    print(f"UB: {Fobj_UB:.6f}", file=output_file)
    print(f"Gap: {gap:.6f} (Tol: {Tol})", file=output_file)
    print("=" * 100, file=output_file)

# ======================================================================================================
# ALGORITHM COMPLETED
# ======================================================================================================

print(f"Solution Time: {time.time() - start_time:.2f} seconds", file=output_file)

def plot_comparison_UB_LB_Aspen(results_UB, results_LB, aspen_data, Ns=17, Nf=8):
    """
    Function to plot comparison between UB, LB and Aspen data
    
    Parameters:
    results_UB: dictionary with Upper Bound results
    results_LB: dictionary with Lower Bound results  
    aspen_data: dictionary with Aspen data
    Ns: number of stages
    Nf: feed stage
    """
    
    # Extract UB values
    T_UB = results_UB['T']
    L_UB = results_UB['L']
    V_UB = results_UB['V']
    x1_UB = results_UB['x'][1]
    x2_UB = results_UB['x'][2]
    Qr_UB = results_UB['Qr']
    Qc_UB = results_UB['Qc']
    
    # Extract LB values
    T_LB = results_LB['T']
    L_LB = results_LB['L']
    V_LB = results_LB['V']
    x1_LB = results_LB['x'][1]
    x2_LB = results_LB['x'][2]
    Qr_LB = results_LB['Qr']
    Qc_LB = results_LB['Qc']
    
    # Aspen data (you need to provide this data)
    T_aspen = aspen_data['T']  # List with temperatures by stage
    L_aspen = aspen_data['L']  # List with liquid flow rates by stage  
    V_aspen = aspen_data['V']  # List with vapor flow rates by stage
    x1_aspen = aspen_data['x1']  # List with benzene compositions
    x2_aspen = aspen_data['x2']  # List with toluene compositions
    Qr_aspen = aspen_data['Qr']  # Aspen reboiler duty
    Qc_aspen = aspen_data['Qc']  # Aspen condenser duty
    
    # Adjust data to be consistent (NaN where doesn't exist)
    L_UB_adj = L_UB.copy()
    L_UB_adj[Ns-1] = float('nan')
    L_LB_adj = L_LB.copy() 
    L_LB_adj[Ns-1] = float('nan')
    L_aspen_adj = L_aspen.copy()
    L_aspen_adj[Ns-1] = float('nan')
    
    V_UB_adj = V_UB.copy()
    V_UB_adj[0] = float('nan')
    V_LB_adj = V_LB.copy()
    V_LB_adj[0] = float('nan') 
    V_aspen_adj = V_aspen.copy()
    V_aspen_adj[0] = float('nan')
    
    # Create DataFrame for comparison
    comparison = pd.DataFrame({
        'Stage': list(range(1, Ns+1)),
        'T_aspen': T_aspen,
        'T_UB': T_UB,
        'T_LB': T_LB,
        'L_aspen': L_aspen_adj,
        'L_UB': L_UB_adj,
        'L_LB': L_LB_adj,
        'V_aspen': V_aspen_adj,
        'V_UB': V_UB_adj,
        'V_LB': V_LB_adj,
        'x1_aspen': x1_aspen,
        'x1_UB': x1_UB,
        'x1_LB': x1_LB,
        'x2_aspen': x2_aspen,
        'x2_UB': x2_UB,
        'x2_LB': x2_LB,
    })
    
    # Create plots
    variables = [
        ('Temperature', 'T (K)', 'T_aspen', 'T_UB', 'T_LB', [350, 390]),
        ('Liquid Flow Rate', 'L (kmol/h)', 'L_aspen', 'L_UB', 'L_LB', [50, 200]),
        ('Vapor Flow Rate', 'V (kmol/h)', 'V_aspen', 'V_UB', 'V_LB', [110, 150]),
        ('x1 (Benzene)', 'Mole Fraction', 'x1_aspen', 'x1_UB', 'x1_LB', [0, 1]),
        ('x2 (Toluene)', 'Mole Fraction', 'x2_aspen', 'x2_UB', 'x2_LB', [0, 1]),
    ]
    
    for title, ylabel, col_aspen, col_UB, col_LB, ylim in variables:
        plt.figure(figsize=(12, 6))
        
        # Plot Aspen data
        plt.plot(comparison['Stage'], comparison[col_aspen], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)
        
        # Plot UB data
        plt.plot(comparison['Stage'], comparison[col_UB], 'bx--', 
                label='UB (Upper Bound)', linewidth=2, markersize=6)
        
        # Plot LB data
        plt.plot(comparison['Stage'], comparison[col_LB], 'g^--', 
                label='LB (Lower Bound)', linewidth=2, markersize=6)
        
        plt.title(f'{title} - UB/LB/Aspen Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Stage', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        if ylim:
            plt.ylim(ylim)
            
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, Ns+1))
        
        # Highlight feed stage
        plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Feed Stage ({Nf})')
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    # Bar chart for heat duties
    heat_duties = pd.DataFrame({
        'Type': ['Qr (Reboiler)', 'Qc (Condenser)'],
        'Aspen': [Qr_aspen, Qc_aspen],
        'UB': [Qr_UB, Qc_UB],
        'LB': [Qr_LB, Qc_LB]
    })
    
    # Plot heat duty comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Qr plot
    ax1.bar(['Aspen', 'UB', 'LB'], 
            [Qr_aspen, Qr_UB, Qr_LB], 
            color=['red', 'blue', 'green'], alpha=0.7)
    ax1.set_title('Reboiler Duty (Qr)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('kJ/h', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate([Qr_aspen, Qr_UB, Qr_LB]):
        ax1.text(i, v + 0.01*v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Qc plot
    ax2.bar(['Aspen', 'UB', 'LB'], 
            [Qc_aspen, Qc_UB, Qc_LB], 
            color=['red', 'blue', 'green'], alpha=0.7)
    ax2.set_title('Condenser Duty (Qc)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('kJ/h', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate([Qc_aspen, Qc_UB, Qc_LB]):
        ax2.text(i, v + 0.01*v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

 
aspen_data = {
'T': T_aspen,
'L': L_aspen,
'V': V_aspen,
'x1': x_Benz_aspen,
'x2': x_Tol_aspen,
'Qr': Qreb_aspen,
'Qc': Qcond_aspen
}

plot_comparison_UB_LB_Aspen(results_UB, results_LB, aspen_data)

# Close output file
output_file.close()
