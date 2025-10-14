from pyomo.environ import *
import time
import numpy as np
import sys
from problem_data import Ns, Nf, Tol, Card_L, Card_V, Card_T, min_interval_width
from aspen_data import T_aspen, L_aspen, V_aspen, x_Benz_aspen, x_Tol_aspen, Qcond_aspen, Qreb_aspen
from resolve_LB import solve_LB
from resolve_UB import solve_UB
from bound_contraction import perform_bound_contraction, check_all_stages_contracted, increase_partition
from utils import (initialize_bounds, calculate_gap, create_UB_initialization, 
                   print_iteration_header, print_section_header, print_iteration_results,
                   print_LB_results, print_UB_results, validate_bounds_structure,
                   get_variable_cardinality, create_convergence_history, update_convergence_history,
                   print_convergence_summary, check_convergence)
from results import (print_final_results, plot_comparison_UB_LB_Aspen, generate_summary_report,
                    save_results_to_excel)

def run_algorithm():
    """Main function of the Bound Contraction algorithm (without redirection)"""
    start_time = time.time()
    
    # Initialization
    print("Initializing Bound Contraction Algorithm...")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    bounds = initialize_bounds()
    Card_var = get_variable_cardinality()
    convergence_history = create_convergence_history()
    
    # Initial validation
    validate_bounds_structure(bounds)
    
    # Main loop
    iter = 1
    gap = float('inf')
    max_iterations = 50  # Maximum number of iterations to prevent infinite loop
    
    while gap >= Tol and iter <= max_iterations:
        iteration_start_time = time.time()
        
        print_iteration_header(iter)
        
        # 1. Partition variables (for logging only)
        # Logging is the process of recording
        # information about a program's execution.
        # It's like a "diary" that the software keeps to document
        # what's happening during its execution.
        
        print_section_header("Partition variables")
        log_partitioning(bounds, Card_var)
        
        # 2. Solve Lower Bound (LB)
        print_section_header("Solve lower bound model (LB)")
        results_LB = solve_LB(bounds, Card_var)
        
        if not is_optimal_flag(results_LB['termination_condition']):
            print(f"LB solver failed: {results_LB['termination_condition']}")
            break
            
        print_LB_results(results_LB)
        Fobj_LB = results_LB['objective_value']
        
        # 3. Solve Upper Bound (UB)
        print_section_header("Solve upper bound model (UB)")
        UB_init = create_UB_initialization(results_LB)
        results_UB = solve_UB(bounds, UB_init)
        
        if not is_optimal_flag(results_UB['termination_condition']):
            print(f"UB solver failed: {results_UB['termination_condition']}")
            break
            
        print_UB_results(results_UB)
        Fobj_UB = results_UB['objective_value']
        
        # 4. Calculate gap and check convergence
        gap = calculate_gap(Fobj_UB, Fobj_LB)
        iteration_time = time.time() - iteration_start_time
        
        print_iteration_results(iter, Fobj_LB, Fobj_UB, gap, Tol)
        update_convergence_history(convergence_history, iter, Fobj_LB, Fobj_UB, gap, iteration_time)
        
        # 5. Check convergence
        if check_convergence(gap, Tol):
            print("Optimal solution found!")
            break
        
        # 6. Perform Bound Contraction
        print_section_header("Perform bound contraction")
        aux_var = perform_bound_contraction(bounds, Card_var, results_LB, results_UB, Fobj_UB, Tol)
        
        # 7. Check if partitions need to be increased
        var_to_increase = check_all_stages_contracted(aux_var)
        if var_to_increase:
            print(f"Increasing partitions for {var_to_increase}...")
            increase_partition(Card_var, var_to_increase)
        
        iter += 1
    
    # Final processing
    total_time = time.time() - start_time
    
    # Final results
    print_final_results(results_UB, results_LB, start_time)
    
    # Generate complete report
    aspen_data = {
        'T': T_aspen, 'L': L_aspen, 'V': V_aspen,
        'x1': x_Benz_aspen, 'x2': x_Tol_aspen,
        'Qr': Qreb_aspen, 'Qc': Qcond_aspen
    }
    
    generate_summary_report(results_UB, results_LB, aspen_data, convergence_history)
    
    # Plot results (main comparison only)
    print("\nGenerating plots...")
    plot_comparison_UB_LB_Aspen(results_UB, results_LB, aspen_data)
    
    # Save results to file
    try:
        save_results_to_excel(results_UB, results_LB, aspen_data)
    except Exception as e:
        print(f"Warning: Could not save results to Excel/CSV: {e}")
    
    print(f"\nAlgorithm completed in {total_time:.2f} seconds")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'results_UB': results_UB,
        'results_LB': results_LB,
        'bounds': bounds,
        'convergence_history': convergence_history,
        'total_time': total_time
    }

def log_partitioning(bounds, Card_var):
    """Logs partitioning information (for logging only)"""
    for var_name in bounds.keys():
        print(f"Variable {var_name}: Card = {Card_var[var_name]}")
        
        num_stages = len(bounds[var_name]['lower'])
        for stage in range(num_stages):
            if should_skip_stage(var_name, stage):
                continue
                
            lower = bounds[var_name]['lower'][stage]
            upper = bounds[var_name]['upper'][stage]
            width = upper - lower
            
            print(f"  Stage {stage+1}: [{lower:.4f}, {upper:.4f}] (width: {width:.4f})")

def should_skip_stage(var_name, stage):
    """Checks if a stage should be skipped in logging"""
    if var_name == 'V' and stage == 0:  # V not defined in stage 1
        return True
    if var_name == 'L' and stage == Ns - 1:  # L not defined in stage 17
        return True
    return False

def is_optimal_flag(flag):
    """Checks if a solver converged to an optimal solution"""
    try:
        return ((flag == TerminationCondition.optimal) 
                or (flag == TerminationCondition.locallyOptimal) 
                or (str(flag).lower().find('optimal') != -1))
    except:
        return str(flag).lower() in ('optimal', 'locallyoptimal')

def main():
    """Main function with redirection to file only"""
    # Open log file to save all output
    log_file = open('algorithm_log.txt', 'w', encoding='utf-8')
    
    # Save original stdout
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout only to file (not to terminal)
        sys.stdout = log_file
        
        print("Bound Contraction Algorithm - Silent Mode")
        print("All output is being saved to algorithm_log.txt")
        print("=" * 80)
        
        # Execute the algorithm
        results = run_algorithm()
        
        print("\n" + "="*80)
        print("BOUND CONTRACTION ALGORITHM COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return results
        
    except Exception as e:
        # In case of error, restore stdout to show the error
        sys.stdout = original_stdout
        print(f"ERROR: Algorithm failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print("Algorithm execution completed. Check algorithm_log.txt for details.")

if __name__ == "__main__":
    results = main()
