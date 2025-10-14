import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from problem_data import Ns, Nf
from utils import calculate_gap

def print_final_results(results_UB, results_LB, start_time=None):
    """Prints final algorithm results"""
    final_gap = calculate_gap(results_UB['objective_value'], results_LB['objective_value'])
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\nCONVERGENCE:")
    print(f"  Final Gap: {final_gap:.6f}")
    print(f"  LB (Lower Bound): {results_LB['objective_value']:.2f} kJ/h")
    print(f"  UB (Upper Bound): {results_UB['objective_value']:.2f} kJ/h")
    
    print(f"\nHEAT DUTIES:")
    print(f"  Reboiler Duty (Qr): {results_UB['Qr']:.2f} kJ/h")
    print(f"  Condenser Duty (Qc): {results_UB['Qc']:.2f} kJ/h")
    
    if start_time:
        from time import time
        elapsed_time = time() - start_time
        print(f"\nPERFORMANCE:")
        print(f"  Total Time: {elapsed_time:.2f} seconds")
    
    print("\n" + "="*80)

def print_detailed_results(results, model_type="UB"):
    """Prints detailed results by stage"""
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS - {model_type}")
    print(f"{'='*60}")
    
    # Temperatures
    print(f"\nTEMPERATURES (K):")
    for j, T_val in enumerate(results['T'], start=1):
        stage_type = get_stage_type(j)
        print(f"  Stage {j:2d} ({stage_type:12s}): {T_val:.2f} K")
    
    # Liquid flow rates
    print(f"\nLIQUID FLOW RATES (kmol/h):")
    for j, L_val in enumerate(results['L'], start=1):
        if not np.isnan(L_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {L_val:.2f} kmol/h")
    
    # Vapor flow rates
    print(f"\nVAPOR FLOW RATES (kmol/h):")
    for j, V_val in enumerate(results['V'], start=1):
        if not np.isnan(V_val):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {V_val:.2f} kmol/h")
    
    # Compositions
    if 'x' in results:
        print(f"\nBENZENE COMPOSITIONS (x1):")
        for j, x1_val in enumerate(results['x'][1], start=1):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {x1_val:.6f}")
        
        print(f"\nTOLUENE COMPOSITIONS (x2):")
        for j, x2_val in enumerate(results['x'][2], start=1):
            stage_type = get_stage_type(j)
            print(f"  Stage {j:2d} ({stage_type:12s}): {x2_val:.6f}")

def get_stage_type(stage):
    """Returns stage type"""
    if stage == 1:
        return "Condenser"
    elif stage == Ns:
        return "Reboiler"
    elif stage == Nf:
        return "Feed Stage"
    else:
        return "Internal"

def create_comparison_dataframe(results_UB, results_LB, aspen_data):
    """Creates DataFrame for comparison between UB, LB and Aspen"""
    
    # Adjusts data for consistency (NaN where not defined)
    L_UB_adj = adjust_liquid_flows(results_UB['L'])
    L_LB_adj = adjust_liquid_flows(results_LB['L'])
    L_aspen_adj = adjust_liquid_flows(aspen_data['L'])
    
    V_UB_adj = adjust_vapor_flows(results_UB['V'])
    V_LB_adj = adjust_vapor_flows(results_LB['V'])
    V_aspen_adj = adjust_vapor_flows(aspen_data['V'])
    
    comparison = pd.DataFrame({
        'Stage': list(range(1, Ns + 1)),
        'Stage_Type': [get_stage_type(j) for j in range(1, Ns + 1)],
        
        # Temperatures
        'T_aspen': aspen_data['T'],
        'T_UB': results_UB['T'],
        'T_LB': results_LB['T'],
        
        # Liquid flows
        'L_aspen': L_aspen_adj,
        'L_UB': L_UB_adj,
        'L_LB': L_LB_adj,
        
        # Vapor flows
        'V_aspen': V_aspen_adj,
        'V_UB': V_UB_adj,
        'V_LB': V_LB_adj,
        
        # Compositions
        'x1_aspen': aspen_data['x1'],
        'x1_UB': results_UB['x'][1],
        'x1_LB': results_LB['x'][1],
        
        'x2_aspen': aspen_data['x2'],
        'x2_UB': results_UB['x'][2],
        'x2_LB': results_LB['x'][2],
    })
    
    return comparison

def adjust_liquid_flows(L_values):
    """Adjusts liquid flows (L not defined in stage Ns)"""
    L_adj = L_values.copy()
    if len(L_adj) >= Ns:
        L_adj[Ns - 1] = float('nan')
    return L_adj

def adjust_vapor_flows(V_values):
    """Adjusts vapor flows (V not defined in stage 1)"""
    V_adj = V_values.copy()
    if len(V_adj) >= 1:
        V_adj[0] = float('nan')
    return V_adj

def plot_comparison_UB_LB_Aspen(results_UB, results_LB, aspen_data):
    """
    Function to plot comparison between UB, LB and Aspen data
    """
    comparison = create_comparison_dataframe(results_UB, results_LB, aspen_data)
    
    # Plot style configuration
    plt.style.use('default')
    colors = {'Aspen': 'red', 'UB': 'blue', 'LB': 'green'}
    markers = {'Aspen': 'o', 'UB': 'x', 'LB': '^'}
    linestyles = {'Aspen': '-', 'UB': '--', 'LB': '--'}
    
    # Variables to plot
    plot_configs = [
        ('Temperature', 'T (K)', 'T_aspen', 'T_UB', 'T_LB', [350, 390], 'temperature'),
        ('Liquid Flow Rate', 'L (kmol/h)', 'L_aspen', 'L_UB', 'L_LB', [50, 200], 'liquid_flow'),
        ('Vapor Flow Rate', 'V (kmol/h)', 'V_aspen', 'V_UB', 'V_LB', [110, 150], 'vapor_flow'),
        ('Benzene Composition', 'Mole Fraction', 'x1_aspen', 'x1_UB', 'x1_LB', [0, 1], 'benzene_composition'),
        ('Toluene Composition', 'Mole Fraction', 'x2_aspen', 'x2_UB', 'x2_LB', [0, 1], 'toluene_composition'),
    ]
    
    for title, ylabel, col_aspen, col_UB, col_LB, ylim, save_suffix in plot_configs:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Aspen data
        ax.plot(comparison['Stage'], comparison[col_aspen], 
                color=colors['Aspen'], marker=markers['Aspen'], linestyle=linestyles['Aspen'],
                label='Aspen', linewidth=2, markersize=6, markevery=1)
        
        # Plot UB data
        ax.plot(comparison['Stage'], comparison[col_UB],
                color=colors['UB'], marker=markers['UB'], linestyle=linestyles['UB'],
                label='UB (Upper Bound)', linewidth=2, markersize=6, markevery=1)
        
        # Plot LB data
        ax.plot(comparison['Stage'], comparison[col_LB],
                color=colors['LB'], marker=markers['LB'], linestyle=linestyles['LB'],
                label='LB (Lower Bound)', linewidth=2, markersize=6, markevery=1)
        
        ax.set_title(f'{title} - UB/LB/Aspen Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(ylim)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, Ns + 1))
        
        # Highlight feed stage
        ax.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(Nf, ax.get_ylim()[1] * 0.95, f'Feed Stage ({Nf})', 
                ha='center', va='top', fontsize=10, color='orange', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        try:
            plt.savefig(f'{save_suffix}_comparison.png', dpi=300, bbox_inches='tight')
            print(f"  - Saved: {save_suffix}_comparison.png")
        except Exception as e:
            print(f"Warning: Could not save {title} plot: {e}")
        
        plt.show()
    
    # Plot heat duties comparison
    plot_heat_duties_comparison(results_UB, results_LB, aspen_data)
    
    return comparison

def plot_heat_duties_comparison(results_UB, results_LB, aspen_data):
    """Plots heat duties comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data for Qr (Reboiler)
    qr_data = {
        'Aspen': aspen_data['Qr'],
        'UB': results_UB['Qr'],
        'LB': results_LB['Qr']
    }
    
    # Data for Qc (Condenser)
    qc_data = {
        'Aspen': aspen_data['Qc'],
        'UB': results_UB['Qc'],
        'LB': results_LB['Qc']
    }
    
    # Plot Qr
    bars1 = ax1.bar(qr_data.keys(), qr_data.values(), 
                   color=['red', 'blue', 'green'], alpha=0.7)
    ax1.set_title('Reboiler Duty (Qr)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('kJ/h', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars1, qr_data.values()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot Qc
    bars2 = ax2.bar(qc_data.keys(), qc_data.values(),
                   color=['red', 'blue', 'green'], alpha=0.7)
    ax2.set_title('Condenser Duty (Qc)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('kJ/h', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars2, qc_data.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    try:
        plt.savefig('heat_duties_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved: heat_duties_comparison.png")
    except Exception as e:
        print(f"Warning: Could not save heat duties plot: {e}")
    plt.show()

def calculate_relative_error(calculated, reference):
    """Calculates percentage relative error"""
    error = []
    for calc, ref in zip(calculated, reference):
        if ref != 0 and not np.isnan(calc) and not np.isnan(ref):
            error.append(100 * (calc - ref) / ref)
        else:
            error.append(np.nan)
    return error

def generate_summary_report(results_UB, results_LB, aspen_data, convergence_history=None):
    """Generates complete summary report"""
    comparison = create_comparison_dataframe(results_UB, results_LB, aspen_data)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    # Convergence statistics
    final_gap = calculate_gap(results_UB['objective_value'], results_LB['objective_value'])
    print(f"\nCONVERGENCE STATISTICS:")
    print(f"  Final Gap: {final_gap:.6f}")
    print(f"  Final UB: {results_UB['objective_value']:.2f} kJ/h")
    print(f"  Final LB: {results_LB['objective_value']:.2f} kJ/h")
    
    # Comparison with Aspen
    print(f"\nCOMPARISON WITH ASPEN:")
    qr_error = 100 * (results_UB['Qr'] - aspen_data['Qr']) / aspen_data['Qr']
    qc_error = 100 * (results_UB['Qc'] - aspen_data['Qc']) / aspen_data['Qc']
    print(f"  Qr Error vs Aspen: {qr_error:.2f}%")
    print(f"  Qc Error vs Aspen: {qc_error:.2f}%")
    
    # Error statistics
    print(f"\nERROR STATISTICS (UB vs Aspen):")
    variables = ['T', 'L', 'V', 'x1', 'x2']
    for var in variables:
        ub_col = f'{var}_UB'
        aspen_col = f'{var}_aspen'
        errors = calculate_relative_error(comparison[ub_col], comparison[aspen_col])
        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            mean_error = np.mean(valid_errors)
            max_error = np.max(np.abs(valid_errors))
            print(f"  {var.upper():5s}: Mean Error = {mean_error:6.2f}%, Max Error = {max_error:6.2f}%")
    
    # Convergence history
    if convergence_history:
        print(f"\nCONVERGENCE HISTORY:")
        print(f"  Total Iterations: {len(convergence_history['iteration'])}")
        if len(convergence_history['gap']) > 1:
            initial_gap = convergence_history['gap'][0]
            improvement = 100 * (initial_gap - final_gap) / initial_gap
            print(f"  Gap Improvement: {improvement:.1f}%")
    
    print("\n" + "="*80)

def save_results_to_excel(results_UB, results_LB, aspen_data, filename="results_comparison.xlsx"):
    """Saves results to Excel file or CSV if openpyxl is not available"""
    comparison = create_comparison_dataframe(results_UB, results_LB, aspen_data)
    
    try:
        # Try to use Excel if openpyxl is available
        import openpyxl
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Comparative data
            comparison.to_excel(writer, sheet_name='Stage_Comparison', index=False)
            
            # Summary
            summary_data = {
                'Metric': ['Qr (kJ/h)', 'Qc (kJ/h)', 'Objective (kJ/h)', 'Final Gap'],
                'Aspen': [aspen_data['Qr'], aspen_data['Qc'], 'N/A', 'N/A'],
                'UB': [results_UB['Qr'], results_UB['Qc'], results_UB['objective_value'], 'N/A'],
                'LB': [results_LB['Qr'], results_LB['Qc'], results_LB['objective_value'], 'N/A'],
                'UB Error %': [
                    100 * (results_UB['Qr'] - aspen_data['Qr']) / aspen_data['Qr'],
                    100 * (results_UB['Qc'] - aspen_data['Qc']) / aspen_data['Qc'],
                    'N/A',
                    calculate_gap(results_UB['objective_value'], results_LB['objective_value']) * 100
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Results saved to {filename}")
        
    except ImportError:
        # Fallback to CSV if openpyxl is not available
        print("openpyxl not available. Saving results to CSV files instead...")
        
        comparison.to_csv("results_comparison.csv", index=False)
        print("  - results_comparison.csv")
        
        summary_data = {
            'Metric': ['Qr (kJ/h)', 'Qc (kJ/h)', 'Objective (kJ/h)', 'Final Gap'],
            'Aspen': [aspen_data['Qr'], aspen_data['Qc'], 'N/A', 'N/A'],
            'UB': [results_UB['Qr'], results_UB['Qc'], results_UB['objective_value'], 'N/A'],
            'LB': [results_LB['Qr'], results_LB['Qc'], results_LB['objective_value'], 'N/A'],
            'UB Error %': [
                100 * (results_UB['Qr'] - aspen_data['Qr']) / aspen_data['Qr'],
                100 * (results_UB['Qc'] - aspen_data['Qc']) / aspen_data['Qc'],
                'N/A',
                calculate_gap(results_UB['objective_value'], results_LB['objective_value']) * 100
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results_summary.csv", index=False)
        print("  - results_summary.csv")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {e}")

def print_iteration_comparison(results_UB, results_LB, iteration):
    """Prints summarized results comparison per iteration"""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} - COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nObjective Values:")
    print(f"  UB: {results_UB['objective_value']:.2f} kJ/h")
    print(f"  LB: {results_LB['objective_value']:.2f} kJ/h")
    print(f"  Gap: {calculate_gap(results_UB['objective_value'], results_LB['objective_value']):.6f}")
    
    print(f"\nHeat Duties:")
    print(f"  Qr UB: {results_UB['Qr']:.2f} kJ/h")
    print(f"  Qr LB: {results_LB['Qr']:.2f} kJ/h")
    print(f"  Qc UB: {results_UB['Qc']:.2f} kJ/h")
    print(f"  Qc LB: {results_LB['Qc']:.2f} kJ/h")

def print_final_comparison_table(results_UB, results_LB, aspen_data):
    """Prints final comparison table"""
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    
    # Table header
    print(f"\n{'Parameter':<25} {'Aspen':<15} {'UB':<15} {'LB':<15} {'UB Error %':<15}")
    print(f"{'-'*85}")
    
    # Table data
    parameters = [
        ('Reboiler Duty (kJ/h)', aspen_data['Qr'], results_UB['Qr'], results_LB['Qr']),
        ('Condenser Duty (kJ/h)', aspen_data['Qc'], results_UB['Qc'], results_LB['Qc']),
        ('Objective (kJ/h)', 'N/A', results_UB['objective_value'], results_LB['objective_value']),
    ]
    
    for param_name, aspen_val, ub_val, lb_val in parameters:
        if aspen_val != 'N/A':
            error = 100 * (ub_val - aspen_val) / aspen_val
            print(f"{param_name:<25} {aspen_val:<15.2f} {ub_val:<15.2f} {lb_val:<15.2f} {error:<15.2f}")
        else:
            print(f"{param_name:<25} {'N/A':<15} {ub_val:<15.2f} {lb_val:<15.2f} {'N/A':<15}")

def print_algorithm_statistics(convergence_history, total_time):
    """Prints algorithm statistics"""
    if not convergence_history or not convergence_history['iteration']:
        return
    
    print(f"\n{'='*60}")
    print("ALGORITHM STATISTICS")
    print(f"{'='*60}")
    
    total_iterations = len(convergence_history['iteration'])
    final_gap = convergence_history['gap'][-1]
    
    print(f"Total Iterations: {total_iterations}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Iteration: {total_time/total_iterations:.2f} seconds")
    print(f"Final Gap: {final_gap:.6f}")
    
    if total_iterations > 1:
        initial_gap = convergence_history['gap'][0]
        improvement = 100 * (initial_gap - final_gap) / initial_gap
        print(f"Gap Improvement: {improvement:.1f}%")
        print(f"Convergence Rate: {improvement/total_iterations:.2f}% per iteration")

