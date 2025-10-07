import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

def solve_mesh_with_scipy(initial_guess):
    """
    Solve MESH equations using scipy.optimize.root
    with specific initial_guess format
    """
    # ... (all function solve_mesh_with_scipy code remains the same) ...
    # Keep all existing code from function solve_mesh_with_scipy
    # I will only add plotting and saving functions at the end

    # =========================================================================
    # FUNCTION solve_mesh_with_scipy CODE (COMPLETE - KEPT ORIGINAL)
    # =========================================================================
    
    Ns=17
    Nf=8
    F=100
    z_benzeno=0.5
    z_tolueno=0.5
    D=50
    B=50
    P=760
    T_feed=365.15
    H_feed=44418.46
    
    # Antoine coefficients
    antoine_benz = [13.985035, 1757.518860, -112.820780]
    antoine_tol = [14.377564, 2126.285144, -109.249431]
    
    # Enthalpy coefficients
    liq_coeffs_benz = [-32980.00, 142.6015, 0.140180, 0.00051393]
    liq_coeffs_tol = [-38420.00, 155.1981, -0.491422, 0.00138480]
    vap_coeffs_benz = [74010.00, -32.7976, 0.541962, -0.00095783]
    vap_coeffs_tol = [82650.00, -36.4012, -0.117715, 0.00015788]
    
    def K_val(component, T):
        """Calculate equilibrium constant"""
        if component == 1:  # Benzene
            A, B, C = antoine_benz
        else:  # Toluene
            A, B, C = antoine_tol
        return np.exp(A - B/(T + C)) / P
    
    def h_liq(component, T):
        """Calculate liquid enthalpy"""
        if component == 1:
            a, b, c, d = liq_coeffs_benz
        else:
            a, b, c, d = liq_coeffs_tol
        return a + b*T + c*T**2 + d*T**3
    
    def h_vap(component, T):
        """Calculate vapor enthalpy"""
        if component == 1:
            a, b, c, d = vap_coeffs_benz
        else:
            a, b, c, d = vap_coeffs_tol
        return a + b*T + c*T**2 + d*T**3
    
    def mesh_equations(X):
        residuals = []
        idx = 0
        
        # Extract variables from vector X
        T = X[idx:idx+Ns]; idx += Ns
        L = X[idx:idx+Ns-1]; idx += Ns-1
        V = X[idx:idx+Ns-1]; idx += Ns-1
        x1 = X[idx:idx+Ns]; idx += Ns
        x2 = X[idx:idx+Ns]; idx += Ns
        Qc = X[idx]; idx += 1
        Qr = X[idx]
        
        # Ensure compositions sum to 1 (except stages with fixed values)
        for j in range(Ns):
            if j != 0 and j != Ns-1:  # Not applied to stages with fixed compositions
                residuals.append(x1[j] + x2[j] - 1.0)
        
        # Boundary conditions for compositions
        residuals.append(x1[0] - 0.98)    # x1 in condenser
        residuals.append(x2[Ns-1] - 0.98) # x2 in reboiler
        
        # Total mass balances
        for j in range(Ns):
            if j == 0:  # Condenser
                residuals.append(V[0] - (L[0] + D))  # V2 = L1 + D
            elif j == Ns-1:  # Reboiler
                residuals.append(L[Ns-2] - (V[Ns-2] + B))  # L_{Ns-1} = V_Ns + B
            else:  # Internal stages
                if j == Nf-1:  # Feed stage (index adjustment)
                    residuals.append(L[j] + V[j-1] - (L[j-1] + V[j] + F))
                else:
                    residuals.append(L[j] + V[j-1] - (L[j-1] + V[j]))
        
        # Component mass balances
        for j in range(Ns):
            # Benzene (component 1)
            if j == 0:  # Condenser
                y2_1 = K_val(1, T[1]) * x1[1]  # y1 in stage 2
                residuals.append(V[0] * y2_1 - (L[0] * x1[0] + D * x1[0]))
            elif j == Ns-1:  # Reboiler
                yNs_1 = K_val(1, T[Ns-1]) * x1[Ns-1]
                residuals.append(L[Ns-2] * x1[Ns-2] - (V[Ns-2] * yNs_1 + B * x1[Ns-1]))
            else:  # Internal stages
                yj_1 = K_val(1, T[j]) * x1[j]
                yj1_1 = K_val(1, T[j+1]) * x1[j+1]
                
                if j == Nf-1:  # Feed stage
                    residuals.append(L[j] * x1[j] + V[j-1] * yj_1 - 
                                   (L[j-1] * x1[j-1] + V[j] * yj1_1 + F * z_benzeno))
                else:
                    residuals.append(L[j] * x1[j] + V[j-1] * yj_1 - 
                                   (L[j-1] * x1[j-1] + V[j] * yj1_1))
            
            # Toluene (component 2)
            if j == 0:  # Condenser
                y2_2 = K_val(2, T[1]) * x2[1]  # y2 in stage 2
                residuals.append(V[0] * y2_2 - (L[0] * x2[0] + D * x2[0]))
            elif j == Ns-1:  # Reboiler
                yNs_2 = K_val(2, T[Ns-1]) * x2[Ns-1]
                residuals.append(L[Ns-2] * x2[Ns-2] - (V[Ns-2] * yNs_2 + B * x2[Ns-1]))
            else:  # Internal stages
                yj_2 = K_val(2, T[j]) * x2[j]
                yj1_2 = K_val(2, T[j+1]) * x2[j+1]
                
                if j == Nf-1:  # Feed stage
                    residuals.append(L[j] * x2[j] + V[j-1] * yj_2 - 
                                   (L[j-1] * x2[j-1] + V[j] * yj1_2 + F * z_tolueno))
                else:
                    residuals.append(L[j] * x2[j] + V[j-1] * yj_2 - 
                                   (L[j-1] * x2[j-1] + V[j] * yj1_2))
        
        # Energy balances
        for j in range(Ns):
            if j == 0:  # Condenser
                # Vapor enthalpy stage 2
                H_vap_2 = (K_val(1, T[1]) * x1[1] * h_vap(1, T[1]) + 
                          K_val(2, T[1]) * x2[1] * h_vap(2, T[1]))
                # Liquid enthalpy stage 1
                H_liq_1 = (x1[0] * h_liq(1, T[0]) + x2[0] * h_liq(2, T[0]))
                
                residuals.append(V[0] * H_vap_2 - (L[0] + D) * H_liq_1 - Qc)
                
            elif j == Ns-1:  # Reboiler
                # Liquid enthalpy stage Ns-1
                H_liq_Ns1 = (x1[Ns-2] * h_liq(1, T[Ns-2]) + 
                            x2[Ns-2] * h_liq(2, T[Ns-2]))
                # Vapor enthalpy stage Ns
                H_vap_Ns = (K_val(1, T[Ns-1]) * x1[Ns-1] * h_vap(1, T[Ns-1]) + 
                           K_val(2, T[Ns-1]) * x2[Ns-1] * h_vap(2, T[Ns-1]))
                # Liquid enthalpy stage Ns
                H_liq_Ns = (x1[Ns-1] * h_liq(1, T[Ns-1]) + 
                           x2[Ns-1] * h_liq(2, T[Ns-1]))
                
                residuals.append(L[Ns-2] * H_liq_Ns1 + Qr - 
                               (V[Ns-2] * H_vap_Ns + B * H_liq_Ns))
                
            else:  # Internal stages
                H_liq_j = x1[j] * h_liq(1, T[j]) + x2[j] * h_liq(2, T[j])
                H_vap_j = (K_val(1, T[j]) * x1[j] * h_vap(1, T[j]) + 
                          K_val(2, T[j]) * x2[j] * h_vap(2, T[j]))
                H_liq_j1 = x1[j-1] * h_liq(1, T[j-1]) + x2[j-1] * h_liq(2, T[j-1])
                H_vap_j1 = (K_val(1, T[j+1]) * x1[j+1] * h_vap(1, T[j+1]) + 
                           K_val(2, T[j+1]) * x2[j+1] * h_vap(2, T[j+1]))
                
                if j == Nf-1:  # Feed stage
                    residuals.append(L[j] * H_liq_j + V[j-1] * H_vap_j - 
                                   (L[j-1] * H_liq_j1 + V[j] * H_vap_j1 + F * H_feed))
                else:
                    residuals.append(L[j] * H_liq_j + V[j-1] * H_vap_j - 
                                   (L[j-1] * H_liq_j1 + V[j] * H_vap_j1))
        
        return np.array(residuals)
    
    # Total number of variables
    n_vars = Ns + (Ns-1) + (Ns-1) + Ns + Ns + 1 + 1  # Total = 5*Ns
    print(f"Total number of variables: {n_vars}")
    
    # Initialize vector X0
    X0 = np.zeros(n_vars)
    idx = 0
    
    # Check if initial_guess was provided
    if initial_guess is None:
        raise ValueError("initial_guess is required. Provide a dictionary with initial values.")
    
    print("Using provided initial points")
    
    # Check and apply initial values
    try:
        # Temperature
        if 'T' in initial_guess and len(initial_guess['T']) == Ns:
            X0[idx:idx+Ns] = initial_guess['T']
        else:
            raise ValueError(f"T must have {Ns} elements")
        idx += Ns
        
        # Liquid flow rates
        if 'L' in initial_guess and len(initial_guess['L']) == Ns-1:
            X0[idx:idx+Ns-1] = initial_guess['L']
        else:
            raise ValueError(f"L must have {Ns-1} elements")
        idx += Ns-1
        
        # Vapor flow rates
        if 'V' in initial_guess and len(initial_guess['V']) == Ns-1:
            X0[idx:idx+Ns-1] = initial_guess['V']
        else:
            raise ValueError(f"V must have {Ns-1} elements")
        idx += Ns-1
        
        # Compositions
        if 'x' in initial_guess and 1 in initial_guess['x'] and 2 in initial_guess['x']:
            if len(initial_guess['x'][1]) == Ns and len(initial_guess['x'][2]) == Ns:
                X0[idx:idx+Ns] = initial_guess['x'][1]  # Benzene
                idx += Ns
                X0[idx:idx+Ns] = initial_guess['x'][2]  # Toluene
                idx += Ns
            else:
                raise ValueError(f"x[1] and x[2] must have {Ns} elements each")
        else:
            raise ValueError("x must be a dictionary with keys 1 and 2")
        
        # Thermal loads
        if 'Qc' in initial_guess:
            X0[idx] = initial_guess['Qc']
        else:
            raise ValueError("Qc must be provided")
        idx += 1
        
        if 'Qr' in initial_guess:
            X0[idx] = initial_guess['Qr']
        else:
            raise ValueError("Qr must be provided")
            
    except Exception as e:
        print(f"Error in initial points: {e}")
        return {'success': False}
    
    # Apply boundary conditions (overwriting if necessary)
    x1_start = Ns + (Ns-1) + (Ns-1)  # Index where x1 starts
    X0[x1_start] = 0.98  # x1 in condenser
    
    x2_start = x1_start + Ns  # Index where x2 starts
    X0[x2_start + Ns - 1] = 0.98  # x2 in reboiler
    
    print(f"Size of vector X0: {len(X0)}")
    print(f"Number of equations: {len(mesh_equations(X0))}")
    
    # Check consistency
    if len(X0) != len(mesh_equations(X0)):
        print(f"ERROR: Number of variables ({len(X0)}) different from number of equations ({len(mesh_equations(X0))})!")
        return {'success': False}
    
    print(f"Solving MESH system with {len(X0)} variables and equations")
    
    # Solve the system
    start_time = time.time()
    
    # Try different methods
    methods = ['hybr', 'lm',  'broyden1']
    solution = None
    
    for method in methods:
        print(f"\nTrying method: {method}")
        try:
            solution = root(mesh_equations, X0, method=method, 
                           options={'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-6})
            
            if solution.success:
                print(f"Convergence achieved with method {method}!")
                break
            else:
                print(f"Method {method} failed: {solution.message}")
        except Exception as e:
            print(f"Error with method {method}: {e}")
    
    end_time = time.time()
    
    if solution and solution.success:
        print(f"\nSolution found in {end_time - start_time:.2f} seconds")
        print(f"Number of function evaluations: {solution.nfev}")
        print(f"Final residual: {np.linalg.norm(solution.fun):.6e}")
        
        # Extract results
        X_sol = solution.x
        idx = 0
        
        T_sol = X_sol[idx:idx+Ns]; idx += Ns
        L_sol = X_sol[idx:idx+Ns-1]; idx += Ns-1
        V_sol = X_sol[idx:idx+Ns-1]; idx += Ns-1
        x1_sol = X_sol[idx:idx+Ns]; idx += Ns
        x2_sol = X_sol[idx:idx+Ns]; idx += Ns
        Qc_sol = X_sol[idx]; idx += 1
        Qr_sol = X_sol[idx]
        
        # Prepare results for comparison
        results = {
            'T': T_sol,
            'L': L_sol,  # Only stages 1-16
            'V': V_sol,  # Only stages 2-17  
            'x1': x1_sol,
            'x2': x2_sol,
            'Qc': Qc_sol,
            'Qr': Qr_sol,
            'success': True,
            'solution_time': end_time - start_time,
            'iterations': solution.nfev,
            'final_residual': np.linalg.norm(solution.fun)
        }
        
        return results
    else:
        print(f"\nConvergence failure after trying all methods")
        if solution:
            print(f"Final message: {solution.message}")
        return {'success': False}


def plot_comparison_scipy_aspen(scipy_result, dados_aspen=None, save_path=None):
    """
    Function to plot Scipy results and compare with Aspen if available
    
    Parameters:
    scipy_result: dictionary with Scipy method results
    dados_aspen: dictionary with Aspen data (optional)
    save_path: path to save images
    """
    
    # Ensure there is a location to save plots
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract Scipy values
    T_scipy = scipy_result['T']  # Stages 1-17
    L_scipy = scipy_result['L']  # Stages 1-16 (L₁₇ doesn't exist)
    V_scipy = scipy_result['V']  # Stages 2-17 (V₁ doesn't exist)
    x1_scipy = scipy_result['x1']  # Stages 1-17
    x2_scipy = scipy_result['x2']  # Stages 1-17
    Qr_scipy = scipy_result['Qr']
    Qc_scipy = scipy_result['Qc']
    
    Ns = 17
    Nf = 8
    
    # Create separate DataFrames for L and V with correct stages
    # Temperature and compositions: stages 1-17
    df_principal = pd.DataFrame({
        'Stage': list(range(1, Ns+1)),
        'T_scipy': T_scipy,
        'x1_scipy': x1_scipy,
        'x2_scipy': x2_scipy,
    })
    
    # L: stages 1-16
    df_L = pd.DataFrame({
        'Stage': list(range(1, Ns)),
        'L_scipy': L_scipy,
    })
    
    # V: stages 2-17  
    df_V = pd.DataFrame({
        'Stage': list(range(2, Ns+1)),
        'V_scipy': V_scipy,
    })
    
    # Add Aspen data if available
    if dados_aspen is not None:
        df_principal['T_aspen'] = dados_aspen['T'][:Ns]
        df_principal['x1_aspen'] = dados_aspen['x1'][:Ns]
        df_principal['x2_aspen'] = dados_aspen['x2'][:Ns]
        
        # Aspen L: stages 1-16
        df_L['L_aspen'] = dados_aspen['L'][:Ns-1]
        
        # Aspen V: stages 2-17
        df_V['V_aspen'] = dados_aspen['V'][1:Ns]  # Skip V₁
    
    # Create plots
    # 1. Temperature plot (stages 1-17)
    plt.figure(figsize=(12, 6))

    if dados_aspen and 'T_aspen' in df_principal.columns:
        plt.plot(df_principal['Stage'], df_principal['T_aspen'], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)
                
    plt.plot(df_principal['Stage'], df_principal['T_scipy'], 'bx--', 
            label='Scipy', linewidth=2, markersize=6)
    
    
    
    plt.title('Temperature - Scipy Solution' + (' vs Aspen' if dados_aspen else ''), 
             fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('T (K)', fontsize=12)
    plt.ylim([350, 390])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, Ns+1))
    plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Feed Stage ({Nf})')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        file_path = os.path.join(save_path, 'temperature_scipy.png')
    else:
        file_path = 'temperature_scipy.png'
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Liquid Flow Rate plot (stages 1-16)
    plt.figure(figsize=(12, 6))
    if dados_aspen and 'L_aspen' in df_L.columns:
        plt.plot(df_L['Stage'], df_L['L_aspen'], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)

    plt.plot(df_L['Stage'], df_L['L_scipy'], 'bx--', 
            label='Scipy', linewidth=2, markersize=6)
    
    
    plt.title('Liquid Flow Rate - Scipy Solution' + (' vs Aspen' if dados_aspen else ''), 
             fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('L (kmol/h)', fontsize=12)
    plt.ylim([50, 200])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, Ns))
    plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Feed Stage ({Nf})')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        file_path = os.path.join(save_path, 'liquid_flow_scipy.png')
    else:
        file_path = 'liquid_flow_scipy.png'
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Vapor Flow Rate plot (stages 2-17)
    plt.figure(figsize=(12, 6))

    if dados_aspen and 'V_aspen' in df_V.columns:
        plt.plot(df_V['Stage'], df_V['V_aspen'], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)
        
    plt.plot(df_V['Stage'], df_V['V_scipy'], 'bx--', 
            label='Scipy', linewidth=2, markersize=6)
    
    
    plt.title('Vapor Flow Rate - Scipy Solution' + (' vs Aspen' if dados_aspen else ''), 
             fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('V (kmol/h)', fontsize=12)
    plt.ylim([110, 150])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2, Ns+1))
    plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Feed Stage ({Nf})')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        file_path = os.path.join(save_path, 'vapor_flow_scipy.png')
    else:
        file_path = 'vapor_flow_scipy.png'
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Individual plot for Benzene
    plt.figure(figsize=(12, 6))
    if dados_aspen and 'x1_aspen' in df_principal.columns:
        plt.plot(df_principal['Stage'], df_principal['x1_aspen'], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)
        
    plt.plot(df_principal['Stage'], df_principal['x1_scipy'], 'bx--', 
            label='Scipy', linewidth=2, markersize=6)
    
    
    plt.title('x1 (Benzene) - Scipy Solution' + (' vs Aspen' if dados_aspen else ''), 
             fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Mole Fraction', fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, Ns+1))
    plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Feed Stage ({Nf})')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        file_path = os.path.join(save_path, 'x1_benzene_scipy.png')
    else:
        file_path = 'x1_benzene_scipy.png'
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Individual plot for Toluene
    plt.figure(figsize=(12, 6))

    if dados_aspen and 'x2_aspen' in df_principal.columns:
        plt.plot(df_principal['Stage'], df_principal['x2_aspen'], 'ro-', 
                label='Aspen', linewidth=2, markersize=6)
        
    plt.plot(df_principal['Stage'], df_principal['x2_scipy'], 'bx--', 
            label='Scipy', linewidth=2, markersize=6)
    
    
    
    plt.title('x2 (Toluene) - Scipy Solution' + (' vs Aspen' if dados_aspen else ''), 
             fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Mole Fraction', fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, Ns+1))
    plt.axvline(x=Nf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Feed Stage ({Nf})')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        file_path = os.path.join(save_path, 'x2_toluene_scipy.png')
    else:
        file_path = 'x2_toluene_scipy.png'
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar plot for thermal loads if Aspen available
    if dados_aspen:
        plt.figure(figsize=(12, 6))
        
        thermal_loads = pd.DataFrame({
            'Type': ['Qr (Reboiler)', 'Qc (Condenser)'],
            'Aspen': [dados_aspen['Qr'], dados_aspen['Qc']],
            'Scipy': [Qr_scipy, Qc_scipy]
        })
        
        x_pos = np.arange(len(thermal_loads))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x_pos - width/2, thermal_loads['Aspen'], width, 
                       label='Aspen', color='red', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, thermal_loads['Scipy'], width, 
                       label='Scipy', color='blue', alpha=0.7)
        
        ax.set_xlabel('Thermal Load', fontsize=12)
        ax.set_ylabel('kJ/h', fontsize=12)
        ax.set_title('Thermal Loads Comparison - Scipy vs Aspen', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(thermal_loads['Type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                       f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        
        if save_path:
            file_path = os.path.join(save_path, 'thermal_loads_scipy_aspen.png')
        else:
            file_path = 'thermal_loads_scipy_aspen.png'
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'principal': df_principal,
        'L': df_L,
        'V': df_V
    }


def save_solution_to_txt(scipy_result, filename="scipy_solution.txt"):
    """
    Saves complete solution to a txt file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MESH SOLUTION - SCIPY METHOD\n")
        f.write("="*70 + "\n\n")
        
        f.write("GENERAL INFORMATION:\n")
        f.write(f"Success: {scipy_result['success']}\n")
        f.write(f"Solution time: {scipy_result['solution_time']:.2f} seconds\n")
        f.write(f"Iterations: {scipy_result['iterations']}\n")
        f.write(f"Final residual: {scipy_result['final_residual']:.6e}\n\n")
        
        f.write("THERMAL LOADS:\n")
        f.write(f"Qc (Condenser): {scipy_result['Qc']:.2f} kJ/h\n")
        f.write(f"Qr (Reboiler): {scipy_result['Qr']:.2f} kJ/h\n\n")
        
        f.write("COMPLETE COLUMN PROFILE:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Stage':<8} {'T [K]':<10} {'L [kmol/h]':<12} {'V [kmol/h]':<12} {'x_Benz':<10} {'x_Tol':<10}\n")
        f.write("-"*100 + "\n")
        
        for j in range(17):
            T_val = scipy_result['T'][j]
            x1_val = scipy_result['x1'][j]
            x2_val = scipy_result['x2'][j]
            
            # L only exists in stages 1-16
            if j < 16:
                L_val = scipy_result['L'][j]
                L_str = f"{L_val:.2f}"
            else:
                L_str = "N/A"
            
            # V only exists in stages 2-17  
            if j > 0:
                V_val = scipy_result['V'][j-1]  # V[0] corresponds to stage 2
                V_str = f"{V_val:.2f}"
            else:
                V_str = "N/A"
            
            f.write(f"{j+1:<8} {T_val:<10.2f} {L_str:<12} {V_str:<12} {x1_val:<10.6f} {x2_val:<10.6f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # Check if compositions sum to 1 (except boundary stages)
        f.write("COMPOSITION VERIFICATION:\n")
        for j in range(17):
            x1_val = scipy_result['x1'][j]
            x2_val = scipy_result['x2'][j]
            sum_val = x1_val + x2_val
            # Stages 1 and 17 have fixed compositions, don't need to sum to 1
            if j == 0 or j == 16:
                status = "✓ (fixed)"
            else:
                status = '✓' if abs(sum_val-1.0) < 1e-6 else '✗'
            f.write(f"Stage {j+1}: {x1_val:.6f} + {x2_val:.6f} = {sum_val:.6f} {status}\n")
    
    print(f"Solution saved in: {filename}")


# Usage example
if __name__ == "__main__":
    print("=== MESH SOLUTION WITH SCIPY ===")
    
    #solucao_tol_0.1_2inter
    #init_guess = {
        #'L': [73.33, 73.33, 70.00, 70.00, 70.00, 70.00, 70.00, 182.00, 182.00, 178.33, 178.33, 186.00, 170.00, 186.00, 170.00, 170.00],  # Estágio 17 sem líquido
        #'V': [123.33, 123.33, 120.00, 120.00, 120.00, 120.00, 120.00, 132.00, 132.00, 128.33, 128.33, 136.00, 120.00, 136.00, 120.00, 120.00],  # Estágio 1 sem vapor
        #'x': {
            #1: [0.980000, 0.943668, 0.884852, 0.790761, 0.703498, 0.791853, 0.786666, 0.588422, 0.486903, 0.400885, 0.374147, 0.263990, 0.168216, 0.122423, 0.073196, 0.042398, 0.020000],
            #2: [0.020000, 0.056332, 0.115148, 0.209239, 0.296502, 0.208147, 0.213334, 0.411578, 0.513097, 0.599115, 0.625853, 0.736010, 0.831784, 0.877577, 0.926804, 0.957602, 0.980000]
        #},
        #'T': [352.19, 352.19, 354.38, 350.00, 358.75, 350.00, 350.00, 350.00, 350.00, 367.50, 367.50, 367.50, 376.25, 376.25, 376.25, 380.62, 382.81],
        #'Qc': 3919767.22,
        #'Qr': 3668800.77
    #}

    #solucao_tol_0.1_4inter
    #init_guess = {
        #'L': [85.53, 84.15, 86.91, 89.68, 70.00, 80.00, 81.38, 177.54, 181.69, 176.67, 170.00, 176.67, 173.39, 173.39, 170.00, 170.00],  # Estágio 17 sem líquido
        #'V': [135.53, 134.15, 136.91, 139.68, 120.00, 130.00, 131.38, 127.54, 131.69, 126.67, 120.00, 126.67, 123.39, 123.39, 120.00, 120.00],  # Estágio 1 sem vapor
       #'x': {
            #1: [0.980000, 0.942010, 0.874283, 0.765755, 0.639711, 0.618535, 0.581759, 0.514616, 0.453207, 0.389832, 0.319892, 0.238939, 0.171425, 0.114154, 0.070431, 0.041116, 0.020000],
            #2: [0.020000, 0.057990, 0.125717, 0.234245, 0.360289, 0.381465, 0.418241, 0.485384, 0.546793, 0.610168, 0.680108, 0.761061, 0.828575, 0.885846, 0.929569, 0.958884, 0.980000]
        #},
        #'T': [351.56, 351.56, 353.63, 355.88, 360.44, 360.46, 358.61, 362.33, 365.79, 367.98, 369.21, 373.16, 374.56, 377.56, 379.81, 380.33, 381.89],
        #'Qc': 3962715.50,
        #'Qr': 3679982.06
    #}

    #solucao_tol_0.1_14inter
    #init_guess = {
        #'L': [90.00, 80.00, 78.57, 70.00, 70.00, 82.86, 80.00, 170.00, 178.57, 184.29, 180.00, 182.86, 172.86, 172.86, 182.86, 171.43],  # Estágio 17 sem líquido
        #'V': [140.00, 130.00, 128.57, 120.00, 120.00, 132.86, 130.00, 120.00, 128.57, 134.29, 130.00, 132.86, 122.86, 122.86, 132.86, 121.43],  # Estágio 1 sem vapor
        #'x': {
            #1: [0.980000, 0.944209, 0.907465, 0.857396, 0.770864, 0.677330, 0.566128, 0.482498, 0.493937, 0.434973, 0.356767, 0.283452, 0.205452, 0.125929, 0.074825, 0.041238, 0.020000],
            #2: [0.020000, 0.055791, 0.092535, 0.142604, 0.229136, 0.322670, 0.433872, 0.517502, 0.506063, 0.565027, 0.643233, 0.716548, 0.794548, 0.874071, 0.925175, 0.958762, 0.980000]
       #},
        #'T': [350.00, 352.50, 352.50, 352.50, 355.00, 357.50, 362.50, 365.00, 362.50, 365.00, 367.50, 370.00, 372.50, 377.50, 380.00, 380.00, 382.50],
        #'Qc': 3922786.67,
        #'Qr': 3670947.31
    #}

    #solucao_tol_0.01_2inter
    #init_guess = {
        #'L': [87.69, 87.69, 76.11, 87.58, 73.41, 83.33, 87.71, 172.78, 176.00, 183.89, 171.93, 176.94, 176.00, 176.00, 173.86, 171.88],  # Estágio 17 sem líquido
        #'V': [137.69, 137.69, 126.11, 137.58, 123.41, 133.33, 137.71, 122.78, 126.00, 133.89, 121.93, 126.94, 126.00, 126.00, 123.86, 121.88],  # Estágio 1 sem vapor
        #'x': {
            #1: [0.980000, 0.945695, 0.889077, 0.813924, 0.721472, 0.621525, 0.536547, 0.505633, 0.471458, 0.385215, 0.279851, 0.221851, 0.156810, 0.106563, 0.067423, 0.038713, 0.020000],
            #2: [0.020000, 0.054305, 0.110923, 0.186076, 0.278528, 0.378475, 0.463453, 0.494367, 0.528542, 0.614785, 0.720149, 0.778149, 0.843190, 0.893437, 0.932577, 0.961287, 0.980000]
        #},
        #'T': [352.29, 353.01, 353.83, 356.02, 354.38, 360.94, 363.12, 363.12, 365.31, 367.50, 371.88, 371.88, 376.25, 378.44, 380.62, 381.96, 382.84],
        #'Qc': 4276540.81,
       #'Qr': 4027660.88
    #}

    #solucao_tol_0.01_4inter
    #init_guess = {
        #'L': [79.30, 88.30, 70.00, 89.68, 70.00, 83.33, 87.22, 181.11, 173.33, 177.02, 181.69, 173.39, 170.00, 181.69, 170.00, 177.02],  # Estágio 17 sem líquido
        #'V': [129.30, 138.30, 120.00, 139.68, 120.00, 133.33, 137.22, 131.11, 123.33, 127.02, 131.69, 123.39, 120.00, 131.69, 120.00, 127.02],  # Estágio 1 sem vapor
        #'x': {
            #1: [0.980000, 0.943544, 0.882902, 0.803762, 0.683455, 0.596575, 0.549750, 0.491782, 0.452276, 0.394661, 0.300576, 0.236695, 0.169995, 0.115535, 0.067052, 0.039677, 0.020000],
            #2: [0.020000, 0.056456, 0.117098, 0.196238, 0.316545, 0.403425, 0.450250, 0.508218, 0.547724, 0.605339, 0.699424, 0.763305, 0.830005, 0.884465, 0.932948, 0.960323, 0.980000]
        #},
        #'T': [352.19, 352.70, 354.27, 356.13, 359.27, 360.81, 362.77, 364.41, 364.58, 367.63, 371.11, 373.16, 375.92, 377.83, 380.44, 381.64, 382.81],
        #'Qc': 4310522.48,
        #'Qr': 4059060.42
    #}

    # Solucao UB
    #init_guess = {
        #'L': [86.42, 88.62, 87.80, 86.75, 85.62, 84.60, 83.84, 173.52, 172.99, 172.34, 171.73, 171.36, 171.31, 171.51, 171.80, 172.08],
        #'V': [136.42, 138.62, 137.80, 136.75, 135.62, 134.60, 133.84, 123.52, 122.99, 122.34, 121.73, 121.36, 121.31, 121.51, 121.80, 122.08],
        #'x': {
            #1: [0.980000, 0.946785, 0.894669, 0.821020, 0.730594, 0.636339, 0.552872, 0.488573, 0.449827, 0.394282, 0.323610, 0.245839, 0.172440, 0.112291, 0.068330, 0.038776, 0.020000],
            #2: [0.020000, 0.053215, 0.105331, 0.178980, 0.269406, 0.363661, 0.447128, 0.511427, 0.550173, 0.605718, 0.676390, 0.754161, 0.827560, 0.887709, 0.931670, 0.961224, 0.980000]
        #},
        #'T': [350.00, 353.01, 354.15, 355.85, 358.06, 360.53, 362.88, 364.81, 366.03, 367.84, 370.28, 373.16, 376.08, 378.63, 380.59, 381.96, 382.85],
        #'Qc': 4365919.87,
        #'Qr': 4067969.34
    #}
    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Ponto Inicial RUIM (Intencionalmente)
    #init_guess = {
        #'L': [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 200.0, 195.0, 190.0, 185.0, 180.0, 175.0, 170.0, 165.0, 160.0], 
        #'V': [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],  
        #'x': {
            #1: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  
            #2: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   
        #},
        #'T': [400.0, 395.0, 390.0, 385.0, 380.0, 375.0, 370.0, 365.0, 360.0, 355.0, 350.0, 345.0, 340.0, 335.0, 330.0, 325.0, 320.0],  
        #'Qc': 6000000,  # Muito alto
        #'Qr': 2000000   # Muito baixo
    #}


    #init_guess = {
        #'L': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        #'V': [130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0, 130.0],
        #'x': {
            #1: [0.98, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74, 0.70, 0.66, 0.62, 0.58, 0.54, 0.50, 0.46, 0.42, 0.38, 0.02],
            #2: [0.02, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94, 0.98]
        #},
        #'T': [355.0, 356.25, 357.5, 358.75, 360.0, 361.25, 362.5, 363.75, 365.0, 366.25, 367.5, 368.75, 370.0, 371.25, 372.5, 373.75, 375.0],
        #'Qc': 4000000.0,
        #'Qr': 4000000.0
    #}

    #init_guess = {
        #'L': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #'V': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #'x': {
            #1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #},
        #'T': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #'Qc': 0.0,
        #'Qr': 0.0
    #}

    # VERY BAD INITIAL GUESS 
    init_guess = {
        'L': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  
        'V': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  
        'x': {
            1: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  
            2: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  
        },
        'T': [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],  
        'Qc': 10000000.0,  
        'Qr': 10000000.0   
    }


    # Run simulation
    results = solve_mesh_with_scipy(initial_guess=init_guess)
    
    if results['success']:
        # Show results
        print("\nMESH SIMULATION RESULTS WITH SCIPY")
        print("="*60)
        print(f"{'Stage':<8} {'T [K]':<10} {'L [kmol/h]':<12} {'V [kmol/h]':<12} {'x_Benz':<10}")
        print("-"*60)
        
        for j in range(17):
            T_val = results['T'][j]
            x1_val = results['x1'][j]
            
            # L only exists in stages 1-16
            if j < 16:
                L_val = results['L'][j]
                L_str = f"{L_val:.2f}"
            else:
                L_str = "N/A"
            
            # V only exists in stages 2-17  
            if j > 0:
                V_val = results['V'][j-1]  # V[0] corresponds to stage 2
                V_str = f"{V_val:.2f}"
            else:
                V_str = "N/A"
            
            print(f"{j+1:<8} {T_val:<10.2f} {L_str:<12} {V_str:<12} {x1_val:<10.6f}")
        
        print(f"\nQc: {results['Qc']:.2f} kJ/h")
        print(f"Qr: {results['Qr']:.2f} kJ/h")
        print(f"Solution time: {results['solution_time']:.2f} seconds")
        print(f"Iterations: {results['iterations']}")
        print(f"Final residual: {results['final_residual']:.6e}")
        
        # Create directory to save results
        save_dir = "RUIM_RUIM"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save solution to txt
        save_solution_to_txt(results, os.path.join(save_dir, "complete_scipy_solution.txt"))
        
        # Try to load Aspen data for comparison
        try:
            from dados_aspen import (T_aspen, L_aspen, V_aspen, 
                                   x_Benz_aspen, x_Tol_aspen, 
                                   Qcond_aspen, Qreb_aspen)
            
            dados_aspen = {
                'T': T_aspen,
                'L': L_aspen, 
                'V': V_aspen,
                'x1': x_Benz_aspen,
                'x2': x_Tol_aspen,
                'Qr': Qreb_aspen,
                'Qc': Qcond_aspen
            }
            
            print("\nAspen data loaded successfully!")
            
        except ImportError:
            print("\nAspen data not available. Plotting only Scipy solution.")
            dados_aspen = None
        
        # Generate plots
        print("Generating plots...")
        dfs_comparacao = plot_comparison_scipy_aspen(results, dados_aspen, save_dir)
        
        # Save data to CSV
        dfs_comparacao['principal'].to_csv(os.path.join(save_dir, 'principal_data_scipy.csv'), index=False)
        dfs_comparacao['L'].to_csv(os.path.join(save_dir, 'L_data_scipy.csv'), index=False)
        dfs_comparacao['V'].to_csv(os.path.join(save_dir, 'V_data_scipy.csv'), index=False)
        
        print(f"\n All results saved in directory: {save_dir}")
        print("   - complete_scipy_solution.txt (detailed solution)")
        print("   - principal_data_scipy.csv (T, x1, x2)")
        print("   - L_data_scipy.csv (liquid flow rates)")
        print("   - V_data_scipy.csv (vapor flow rates)")
        print("   - Various plots (.png)")
        
    else:
        print("Simulation did not converge.")
