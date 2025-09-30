from pyomo.environ import *
import time
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################
def calcula_variaveld(ord_d, Card_D, limites):
    """Calcula o valor da variável para um ponto de discretização específico"""
    return limites[0] + ((limites[1] - limites[0]) / (Card_D - 1)) * (ord_d - 1)
########################################################################################################

########################################################################################################
def solve_LB(variaveis_limites, Card):

    model = ConcreteModel()

    Ns = 17   # Número total de estágios (incluindo condensador e refervedor)
    Nf = 8    # Estágio de alimentação (7 no Aspen corresponde ao estágio 8 no nosso modelo)

    # =============================================================================
    # Conjuntos
    # =============================================================================
    model.components = Set(initialize=[1, 2])       # Componentes (1=Benzeno, 2=Tolueno)
    model.stages = Set(initialize=range(1, Ns+1))   # Ns estágios (1=condensador, Ns=refervedor)


    M_card = Card['T']  # Número de pontos para temperatura
    N_card = Card['L']  # Número de pontos para vazão líquida
    P_card = Card['V']  # Número de pontos para vazão de vapor

    model.M_set = Set(initialize=range(1, M_card+1))  # vai de 1 a M_card
    model.N_set = Set(initialize=range(1, N_card+1))  
    model.P_set = Set(initialize=range(1, P_card+1))      

    model.M_prime = Set(initialize=range(1, M_card)) # vai de 1 a M_card-1
    model.N_prime = Set(initialize=range(1, N_card))
    model.P_prime = Set(initialize=range(1, P_card))

    # =============================================================================
    # Parâmetros
    # =============================================================================
    # Pressão
    model.P = Param(initialize=760)  # mmHg

    # Alimentação
    model.F = Param(initialize=100)  # kmol/h
    model.z = Param(model.components, initialize={1: 0.5, 2: 0.5})  # Fração molar

    model.T_feed = Param(initialize=365.15)  # Temperatura de alimentação K

    # Coeficientes de Antoine (Benzeno e Tolueno)
    model.kk = Param(model.components, [1, 2, 3], initialize={
        (1, 1): 13.985035, (1, 2): 1757.518860, (1, 3): -112.820780,
        (2, 1): 14.377564, (2, 2): 2126.285144, (2, 3): -109.249431
    })


    model.liq_coeffs = Param(model.components, [1, 2, 3, 4], initialize={
            (1, 1): -32980.00, (1, 2): 142.6015, (1, 3): 0.140180, (1, 4): 0.00051393,
            (2, 1): -38420.00, (2, 2): 155.1981, (2, 3): -0.491422, (2, 4): 0.00138480
        })

    model.vap_coeffs = Param(model.components, [1, 2, 3, 4], initialize={
            (1, 1): 74010.00, (1, 2): -32.7976, (1, 3): 0.541962, (1, 4): -0.00095783,
            (2, 1): 82650.00, (2, 2): -36.4012, (2, 3): -0.117715, (2, 4): 0.00015788
        })

    model.D = Param(initialize = 50)  # Destilado
    model.B = Param(initialize = 50)  # Produto de fundo

    model.H_feed = Param(initialize=44418.46) # Entalpia da alimentação (kJ/kmol)


    # =============================================================================
    # Variáveis
    # =============================================================================
    Qcond_upper = 5250000.00
    Qcond_lower = 3364500.00
    Qreb_upper = 5002421.00
    Qreb_lower = 3116921.00
    x_upper = 1
    x_lower = 0

    # Vazões - usando os limites do dicionário variaveis_limites
    model.L = Var(
        model.stages - {Ns},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['L']['lower'][j-1] if j < Ns else None,
            variaveis_limites['L']['upper'][j-1] if j < Ns else None
        )
    )

    model.V = Var(
        model.stages - {1},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['V']['lower'][j-1] if j > 1 else None,
            variaveis_limites['V']['upper'][j-1] if j > 1 else None
        )
    )

    # Composições
    model.x = Var(
        model.components, model.stages,
        domain=NonNegativeReals,
        bounds=(x_lower, x_upper)
    )

    # Temperaturas - usando os limites do dicionário variaveis_limites
    model.T = Var(
        model.stages,
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['T']['lower'][j-1],
            variaveis_limites['T']['upper'][j-1]
        )
    )

    # Cargas térmicas
    model.Qc = Var(
        domain=Reals,
        bounds=(Qcond_lower, Qcond_upper)
    )
    model.Qr = Var(
        domain=Reals,
        bounds=(Qreb_lower, Qreb_upper)
    )


    # Variáveis auxiliares para linearização
    model.phi = Var(model.components, model.stages, domain=NonNegativeReals)
    model.lx = Var(model.components, model.stages - {Ns}, domain=NonNegativeReals)
    model.vy = Var(model.components, model.stages - {1}, domain=NonNegativeReals)
    model.hl = Var(model.components, model.stages - {Ns}, domain=Reals)
    model.hv = Var(model.components, model.stages - {1}, domain=Reals)
    model.hx = Var(model.components, model.stages, domain=Reals)    

    # Variáveis binárias para discretização
    model.lambda_var = Var(model.stages, model.M_prime, domain=Binary)
    model.omega_var = Var(model.stages - {Ns}, model.N_prime, domain=Binary)
    model.rho_var = Var(model.stages - {1}, model.P_prime, domain=Binary)    

    # Variáveis auxiliares para produtos
    model.a = Var(model.components, model.stages, model.M_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.b = Var(model.components, model.stages - {Ns}, model.N_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.c = Var(model.components, model.stages - {1}, model.M_prime, model.P_prime, domain=NonNegativeReals, bounds=(0, x_upper))
    model.d = Var(model.components, model.stages - {Ns}, model.M_prime, model.N_prime, domain=NonNegativeReals, bounds=(0, x_upper))    

    # Variáveis binárias auxiliares para produtos
    model.alpha = Var(model.stages - {1}, model.M_prime, model.P_prime, domain=Binary)
    model.sigma = Var(model.stages - {Ns}, model.M_prime, model.N_prime, domain=Binary)



 
    # =============================================================================
    # Print discretized variables
    # =============================================================================
    def init_T_hat(model, j, m):
        return calcula_variaveld(m, M_card, [variaveis_limites['T']['lower'][j-1], variaveis_limites['T']['upper'][j-1]])

    def init_L_hat(model, j, n):
        if j < Ns:  # L só é definido para estágios 1 a 16
            return calcula_variaveld(n, N_card, [variaveis_limites['L']['lower'][j-1], variaveis_limites['L']['upper'][j-1]])
        else:
            return 0  # Valor padrão para estágio 17 (não usado)

    def init_V_hat(model, j, p):
        if j > 1:  # V só é definido para estágios 2 a 17
            return calcula_variaveld(p, P_card, [variaveis_limites['V']['lower'][j-1], variaveis_limites['V']['upper'][j-1]])
        else:
            return 0  # Valor padrão para estágio 1 (não usado)

    model.T_hat = Param(model.stages, model.M_set, initialize=init_T_hat)
    model.L_hat = Param(model.stages, model.N_set, initialize=init_L_hat)
    model.V_hat = Param(model.stages, model.P_set, initialize=init_V_hat)

    # =============================================================================
    # Print discretized variables
    # =============================================================================
    #print("Discretized Temperature Values (T_hat):")
    #for j in model.stages:
        #print(f"Stage {j}: {[model.T_hat[j, m] for m in model.M_set]}")
#
    #print("\nDiscretized Liquid Flow Values (L_hat):")
    #for j in model.stages:
        #if j < Ns:  # L is only defined for stages 1 to 16
            #print(f"Stage {j}: {[model.L_hat[j, n] for n in model.N_set]}")
#
    #print("\nDiscretized Vapor Flow Values (V_hat):")
    #for j in model.stages:
        #if j > 1:  # V is only defined for stages 2 to 17
            #print(f"Stage {j}: {[model.V_hat[j, p] for p in model.P_set]}")



    # Fixar composições conforme especificado
    model.x[1, 1].fix(0.98)  # x11 não é variável (fixo em 0.98)
    model.x[2, Ns].fix(0.98) # x2_Ns não é variável (fixo em 0.98)

    # =============================================================================
    # Restrições
    # =============================================================================

    # Restrições de discretização para T, L, V
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


    # Normalização das composições (exclui estágios 1 e Ns devido às fixações)
    def norm_x_rule(model, j):
        if j == 1 or j == Ns:
            return Constraint.Skip
        return sum(model.x[i, j] for i in model.components) == 1
    model.norm_x_eq = Constraint(model.stages, rule=norm_x_rule)



    def norm_phi_rule(model, j):
        return (1/model.P) * sum(model.phi[i, j] for i in model.components) == 1
    model.norm_phi_eq = Constraint(model.stages, rule=norm_phi_rule)

    # Restrições para φ
    def phi_lower_rule(model, i, j):
        return sum(exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m])) * model.a[i, j, m] 
                    for m in model.M_prime) <= model.phi[i, j]
        
    def phi_upper_rule(model, i, j):
        return model.phi[i, j] <= sum(exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m+1])) * model.a[i, j, m] 
                    for m in model.M_prime)
        

    model.phi_lower = Constraint(model.components, model.stages, rule=phi_lower_rule)
    model.phi_upper = Constraint(model.components, model.stages, rule=phi_upper_rule)


    # Balanços materiais
    def mass_balance_total(model, j):
        if j == 1:  # Condensador total
            return model.V[j+1] == model.L[j] + model.D
        elif j == Ns:  # Refervedor parcial
            return model.L[j-1] == model.V[j] + model.B
        else:  # Estágios internos
            if j == Nf:  # Estágio de alimentação
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1] + model.F
            else:
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1]
    model.mass_balance_total_eq = Constraint(model.stages, rule=mass_balance_total)

    # Balanços por componente
    def mass_balance_component(model, i, j):
        if j == 1:  # Condensador
            return (1/model.P) * model.vy[i, j+1] - model.lx[i, j] - model.D * model.x[i, j] == 0
        elif j == Ns:  # Refervedor
            return model.lx[i, j-1] == (1/model.P) * model.vy[i, j] + model.B * model.x[i, j]
        else:  # Estágios internos
            if j == Nf:
                return (model.lx[i, j] + (1/model.P) * model.vy[i, j] - model.lx[i, j-1] - 
                        (1/model.P) * model.vy[i, j+1] - model.F * model.z[i] == 0)
            else:
                return (model.lx[i, j] + (1/model.P) * model.vy[i, j] - model.lx[i, j-1] - 
                        (1/model.P) * model.vy[i, j+1] == 0)
    model.mass_balance_component_eq = Constraint(model.components, model.stages, rule=mass_balance_component)

    # Restrições para lx
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
        
    # Restrições para vy 
    def vy_lower_rule(model, i, j):
            return sum(model.V_hat[j, p] * exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m])) * 
                model.c[i, j, m, p] for m in model.M_prime for p in model.P_prime) <= model.vy[i, j]
        
    def vy_upper_rule(model, i, j):
            return model.vy[i, j] <= sum(model.V_hat[j, p+1] * exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m+1])) * 
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
    # Balanços de energia 
    # =============================================================================
    def energy_balance(model, j):
        if j == 1:  # Condensador
            return ((1/model.P) * sum(model.hv[i, 2] for i in model.components) - 
                        sum(model.hl[i, 1] for i in model.components) - 
                        model.D * sum(model.hx[i, 1] for i in model.components) - 
                        model.Qc == 0)
        elif j == Ns:  # Refervedor
            return (sum(model.hl[i, Ns-1] for i in model.components) + model.Qr - 
                        (1/model.P) * sum(model.hv[i, Ns] for i in model.components) - 
                        model.B * sum(model.hx[i, Ns] for i in model.components) == 0)
        else:  # Estágios internos
            if j == Nf:
                return (sum(model.hl[i, j] for i in model.components) + 
                        (1/model.P) * sum(model.hv[i, j] for i in model.components) - 
                            sum(model.hl[i, j-1] for i in model.components) - 
                            (1/model.P) * sum(model.hv[i, j+1] for i in model.components) - 
                            model.F * model.H_feed == 0)
            else:
                return (sum(model.hl[i, j] for i in model.components) + 
                            (1/model.P) * sum(model.hv[i, j] for i in model.components) - 
                            sum(model.hl[i, j-1] for i in model.components) - 
                            (1/model.P) * sum(model.hv[i, j+1] for i in model.components) == 0)
    model.energy_balance_eq = Constraint(model.stages, rule=energy_balance)
        
    # Funções para cálculo de entalpia
    def h0_liq_expr(model, i, T_val):
            return (model.liq_coeffs[i, 1] + 
                    model.liq_coeffs[i, 2] * T_val + 
                    model.liq_coeffs[i, 3] * T_val**2 + 
                    model.liq_coeffs[i, 4] * T_val**3)
        
    def h0_vap_expr(model, i, T_val):
            return (model.vap_coeffs[i, 1] + 
                    model.vap_coeffs[i, 2] * T_val + 
                    model.vap_coeffs[i, 3] * T_val**2 + 
                    model.vap_coeffs[i, 4] * T_val**3)
        
    # Restrições para hl
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
        
    # Restrições para hv 
    def hv_lower_rule(model, i, j):
        return sum(exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m])) * 
                    h0_vap_expr(model, i, model.T_hat[j, m]) * model.V_hat[j, p] * model.c[i, j, m, p] 
                for m in model.M_prime for p in model.P_prime) <= model.hv[i, j]
        
    def hv_upper_rule(model, i, j):
        return model.hv[i, j] <= sum(exp(model.kk[i, 1] - model.kk[i, 2]/(model.kk[i, 3] + model.T_hat[j, m+1])) * 
                    h0_vap_expr(model, i, model.T_hat[j, m+1]) * model.V_hat[j, p+1] * model.c[i, j, m, p] 
                for m in model.M_prime for p in model.P_prime)
        
    model.hv_lower = Constraint(model.components, model.stages - {1}, rule=hv_lower_rule)
    model.hv_upper = Constraint(model.components, model.stages - {1}, rule=hv_upper_rule)
        
    # Restrições para hx
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
    # Balanço Total de Energia
    # =============================================================================
    def total_energy_balance(model):
            return (model.F * model.H_feed + model.Qr == 
                model.D * (sum(model.hx[i, 1] for i in model.components)) + 
                model.B * (sum(model.hx[i, Ns] for i in model.components)) + 
                model.Qc)
    model.total_energy_eq = Constraint(rule=total_energy_balance)

    # Função objetivo
    def objective_rule(model):
        return model.Qr
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # =============================================================================
    # Resolver o modelo
    # =============================================================================
    solver = SolverFactory('gams')
    solver.options['solver'] = 'cplex'
    results = solver.solve(model)
    
    # =============================================================================
    # Coletar e retornar resultados
    # =============================================================================
    # Verificar se a solução foi encontrada
    if results.solver.termination_condition == TerminationCondition.optimal:
        # Coletar valores das variáveis principais
        termination_condition = results.solver.termination_condition
        Qr_val = value(model.Qr)
        Qc_val = value(model.Qc)
        
        # Coletar valores por estágio
        T_values = [value(model.T[j]) for j in model.stages]
        L_values = [value(model.L[j]) if j in model.L else float('nan') for j in model.stages]
        V_values = [value(model.V[j]) if j in model.V else float('nan') for j in model.stages]
        
        # Coletar composições
        x_values = {}
        for i in model.components:
            x_values[i] = [value(model.x[i, j]) for j in model.stages]
        
        # Coletar variáveis auxiliares
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
        
        # Coletar variáveis binárias
        lambda_values = {}
        for j in model.stages:
            lambda_values[j] = [value(model.lambda_var[j, m]) for m in model.M_prime]
        
        omega_values = {}
        for j in model.stages - {Ns}:
            omega_values[j] = [value(model.omega_var[j, n]) for n in model.N_prime]
        
        rho_values = {}
        for j in model.stages - {1}:
            rho_values[j] = [value(model.rho_var[j, p]) for p in model.P_prime]
        
        # Coletar variáveis de produtos
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
        
        # Coletar variáveis binárias auxiliares
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
        
        # Retornar dicionário com todos os resultados
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
            'model': model,  # Opcional: retornar o modelo completo se necessário
            'results': results  # Opcional: retornar os resultados completos
        }
 

        return results_dict
        
    else:
        #print(f"Solver não convergiu. Condição de término: {results.solver.termination_condition}")
        return {
            'termination_condition': results.solver.termination_condition,
            'error': 'Solver não convergiu para uma solução ótima'
        }
########################################################################################################

########################################################################################################
def solve_UB(variaveis_limites, UB_init):

    model = ConcreteModel()

    Ns = 17   # Número total de estágios (incluindo condensador e refervedor)
    Nf = 8    # Estágio de alimentação (7 no Aspen corresponde ao estágio 8 no nosso modelo)

    # =============================================================================
    # Conjuntos
    # =============================================================================
    model.components = Set(initialize=[1, 2])       # Componentes (1=Benzeno, 2=Tolueno)
    model.stages = Set(initialize=range(1, Ns+1))   # Ns estágios (1=condensador, Ns=refervedor)


    # =============================================================================
    # Parâmetros
    # =============================================================================
    # Pressão
    model.P = Param(initialize=760)  # mmHg

    # Alimentação
    model.F = Param(initialize=100)  # kmol/h
    model.z = Param(model.components, initialize={1: 0.5, 2: 0.5})  # Fração molar

    model.T_feed = Param(initialize=365.15)  # Temperatura de alimentação K

    # Coeficientes de Antoine (Benzeno e Tolueno)
    model.kk = Param(model.components, [1, 2, 3], initialize={
        (1, 1): 13.985035, (1, 2): 1757.518860, (1, 3): -112.820780,
        (2, 1): 14.377564, (2, 2): 2126.285144, (2, 3): -109.249431
    })

    model.liq_coeffs = Param(model.components, [1, 2, 3, 4], initialize={
            (1, 1): -32980.00, (1, 2): 142.6015, (1, 3): 0.140180, (1, 4): 0.00051393,
            (2, 1): -38420.00, (2, 2): 155.1981, (2, 3): -0.491422, (2, 4): 0.00138480
        })

    model.vap_coeffs = Param(model.components, [1, 2, 3, 4], initialize={
            (1, 1): 74010.00, (1, 2): -32.7976, (1, 3): 0.541962, (1, 4): -0.00095783,
            (2, 1): 82650.00, (2, 2): -36.4012, (2, 3): -0.117715, (2, 4): 0.00015788
        })

    model.H_feed = Param(initialize=44418.46) # Entalpia da alimentação (kJ/kmol)

    model.D = Param(initialize = 50)  # Destilado
    model.B = Param(initialize = 50)  # Produto de fundo


    # =============================================================================
    # Variáveis
    # =============================================================================
    Qcond_upper = 5250000.00
    Qcond_lower = 3364500.00
    Qreb_upper = 5002421.00
    Qreb_lower = 3116921.00
    x_upper = 1
    x_lower = 0

    # Vazões - usando os limites do dicionário variaveis_limites
    model.L = Var(
        model.stages - {Ns},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['L']['lower'][j-1] if j < Ns else None,
            variaveis_limites['L']['upper'][j-1] if j < Ns else None
        )
    )

     # Inicializar vazões líquidas (apenas estágios 1 a 16)
    for j in model.stages:
        if j < Ns:  # Estágios 1 a 16
            model.L[j].value = UB_init['L'][j-1]

    model.V = Var(
        model.stages - {1},
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['V']['lower'][j-1] if j > 1 else None,
            variaveis_limites['V']['upper'][j-1] if j > 1 else None
        )
    )

    # Inicializar vazões de vapor (apenas estágios 2 a 17)
    for j in model.stages:
        if j > 1:  # Estágios 2 a 17
            model.V[j].value = UB_init['V'][j-1]

    # Composições
    model.x = Var(
        model.components, model.stages,
        domain=NonNegativeReals,
        bounds=(x_lower, x_upper)
    )

    # Inicializar composições
    for i in model.components:
        for j in model.stages:
            model.x[i, j].value = UB_init['x'][i][j-1]

    # Temperaturas - usando os limites do dicionário variaveis_limites
    model.T = Var(
        model.stages,
        domain=NonNegativeReals,
        bounds=lambda model, j: (
            variaveis_limites['T']['lower'][j-1],
            variaveis_limites['T']['upper'][j-1]
        )
    )

    if UB_init is not None:
        # Inicializar temperaturas
        for j in model.stages:
            model.T[j].value = UB_init['T'][j-1]

    # Cargas térmicas
    model.Qc = Var(
        domain=Reals,
        bounds=(Qcond_lower, Qcond_upper)
    )

    model.Qr = Var(
        domain=Reals,
        bounds=(Qreb_lower, Qreb_upper)
    )

    # Inicializar cargas térmicas
    model.Qc.value = UB_init['Qc']
    model.Qr.value = UB_init['Qr']

    # Fixar composições conforme especificado
    model.x[1, 1].fix(0.98)  # x11 não é variável (fixo em 0.98)
    model.x[2, Ns].fix(0.98) # x2_Ns não é variável (fixo em 0.98)

    # =============================================================================
    # Restrições
    # =============================================================================

    # Normalização das composições (exclui estágios 1 e Ns devido às fixações)
    def norm_x_rule(model, j):
        if j == 1 or j == Ns:
            return Constraint.Skip
        return sum(model.x[i, j] for i in model.components) == 1
    model.norm_x_eq = Constraint(model.stages, rule=norm_x_rule)

    # Balanços materiais
    def mass_balance_total(model, j):
        if j == 1:  # Condensador total
            return model.V[j+1] == model.L[j] + model.D
        elif j == Ns:  # Refervedor parcial
            return model.L[j-1] == model.V[j] + model.B
        else:  # Estágios internos
            if j == Nf:  # Estágio de alimentação
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1] + model.F
            else:
                return model.L[j] + model.V[j] == model.L[j-1] + model.V[j+1]
    model.mass_balance_total_eq = Constraint(model.stages, rule=mass_balance_total)

    # Balanços por componente
    def mass_balance_component(model, i, j):
        if j == 1:  # Condensador
            return (model.V[j+1] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]) 
                    == model.L[j] * model.x[i, j] + model.D * model.x[i, j])
        elif j == Ns:  # Refervedor
            return (model.L[j-1] * model.x[i, j-1] 
                    == model.V[j] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                    + model.B * model.x[i, j])
        else:  # Estágios internos
            if j == Nf:
                return (model.L[j] * model.x[i, j] 
                        + model.V[j] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                        == model.L[j-1] * model.x[i, j-1] 
                        + model.V[j+1] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]) + 
                        model.F * model.z[i])
            else:
                return (model.L[j] * model.x[i, j] 
                        + model.V[j] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                        == model.L[j-1] * model.x[i, j-1] 
                        + model.V[j+1] * (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]))
    model.mass_balance_component_eq = Constraint(model.components, model.stages, rule=mass_balance_component)

    # =============================================================================
    # Balanços de energia 
    # =============================================================================

    def h0_liq_expr(model, i, T_val):
            return (model.liq_coeffs[i, 1] + 
                    model.liq_coeffs[i, 2] * T_val + 
                    model.liq_coeffs[i, 3] * T_val**2 + 
                    model.liq_coeffs[i, 4] * T_val**3)        

    def h0_vap_expr(model, i, T_val):
            return (model.vap_coeffs[i, 1] + 
                    model.vap_coeffs[i, 2] * T_val + 
                    model.vap_coeffs[i, 3] * T_val**2 + 
                    model.vap_coeffs[i, 4] * T_val**3)
    # Balanços de energia

    def energy_balance(model, j):
        if j == 1:  # Condensador
            return model.V[j+1] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ) == (model.L[j] + model.D) * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.Qc
        
        elif j == Ns:  # Refervedor
            return model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.Qr == model.V[j] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) + model.B * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        )
        
        else:  # Estágios internos
            if j == Nf:
                return (model.L[j] * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.V[j] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) == 
                        model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.V[j+1] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ) +  model.F * model.H_feed)
            
            else:
                return (model.L[j] * sum(
        model.x[i, j] * h0_liq_expr(model, i, model.T[j]) for i in model.components
        ) + model.V[j] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j] + model.kk[i, 3])))/model.P) * model.x[i, j]) 
                * h0_vap_expr(model, i, model.T[j]) for i in model.components
            ) == 
                        model.L[j-1] * sum(
        model.x[i, j-1] * h0_liq_expr(model, i, model.T[j-1]) for i in model.components
        ) + model.V[j+1] * sum(
                (((exp(model.kk[i, 1] - model.kk[i, 2]/(model.T[j+1] + model.kk[i, 3])))/model.P) * model.x[i, j+1]) 
                * h0_vap_expr(model, i, model.T[j+1]) for i in model.components
            ))
    model.energy_balance = Constraint(model.stages, rule=energy_balance)

    # =============================================================================
    # Balanço Total de Energia
    # =============================================================================
    #def total_energy_balance(model):
    #    return (model.F * model.H_feed + model.Qr == 
    #            model.D * sum(
    #   model.x[i, 1] * h0_liq_expr(model, i, model.T[1]) for i in model.components
    #   ) + 
    #           model.B * sum(
    #   model.x[i, Ns] * h0_liq_expr(model, i, model.T[Ns]) for i in model.components
    #   ) + 
    #           model.Qc)
    #model.total_energy_eq = Constraint(rule=total_energy_balance)

    # Função objetivo
    def objective_rule(model):
        return model.Qr
    model.obj = Objective(rule=objective_rule, sense=minimize)


    # =============================================================================
    # Resolver o modelo
    # =============================================================================
    solver = SolverFactory('gams')
    solver.options['solver'] = 'conopt'
    results = solver.solve(model)


    # =============================================================================
    # Coletar e retornar resultados
    # =============================================================================
    # Verificar se a solução foi encontrada
    if results.solver.termination_condition == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
        # Coletar valores das variáveis principais
        termination_condition = results.solver.termination_condition
        Qr_val = value(model.Qr)
        Qc_val = value(model.Qc)
        
        # Coletar valores por estágio
        T_values = [value(model.T[j]) for j in model.stages]
        L_values = [value(model.L[j]) if j in model.L else float('nan') for j in model.stages]
        V_values = [value(model.V[j]) if j in model.V else float('nan') for j in model.stages]
        
        # Coletar composições
        x_values = {}
        for i in model.components:
            x_values[i] = [value(model.x[i, j]) for j in model.stages]
            
        # Retornar dicionário com todos os resultados
        results_dict = {
            'termination_condition': termination_condition,
            'objective_value': Qr_val,
            'Qr': Qr_val,
            'Qc': Qc_val,
            'T': T_values,
            'L': L_values,
            'V': V_values,
            'x': x_values,
            'model': model,  # Opcional: retornar o modelo completo se necessário
            'results': results  # Opcional: retornar os resultados completos
        }
 

        return results_dict
        
    else:
        #print(f"Solver não convergiu. Condição de término: {results.solver.termination_condition}")
        return {
            'termination_condition': results.solver.termination_condition,
            'error': 'Solver não convergiu para uma solução ótima'
        }
########################################################################################################

########################################################################################################
# Define o arquivo de saída
output_file = open('teste.txt', 'w+', encoding='utf-8')
start_time = time.time()  # Calcula o inicio do tempo
########################################################################################################

#########################
#       Paramêtros      #
#########################

Ns = 17   # Número total de estágios (incluindo condensador e refervedor)
Nf = 8    # Estágio de alimentação (7 no Aspen corresponde ao estágio 8 no nosso modelo)

# Número inicial de partições para cada variável
Card_L = 3
Card_V = 3
Card_T = 3

# Tolerância 
Tol = 0.1

#epsilon = 1e-3
epsilon = 0

min_interval_width = 1e-2  # Valor mínimo permitido para a diferença entre upper e lower


# Solução Upper Bound 
T_upper = [385] * Ns
T_lower = [350] * Ns
L_upper = [190] * Ns        
L_lower = [50] * Ns  
V_upper = [140] * Ns   
V_lower = [120] * Ns 
 

variaveis_limites = {
    'L': {'upper': L_upper, 'lower': L_lower},
    'V': {'upper': V_upper, 'lower': V_lower},
    'T': {'upper': T_upper, 'lower': T_lower},
}

# Inicializa os limites atuais para contração
limites_atuais = {
    'L': [list(L_lower), list(L_upper)],
    'V': [list(V_lower), list(V_upper)],
    'T': [list(T_lower), list(T_upper)],
}

# Dicionário para armazenar o número de partições por variável
Card_var = {'L': Card_L, 'V': Card_V, 'T': Card_T}
########################################################################################################

########################################################################################################

def is_optimal_flag(flag):
    """Compatibilidade: aceita string ou TerminationCondition enum."""
    try:
        return (flag == TerminationCondition.optimal) or (flag == TerminationCondition.locallyOptimal) or (str(flag).lower().find('optimal')!=-1)
    except:
        return str(flag).lower() in ('optimal', 'locallyoptimal', 'locallyoptimal')

def is_infeasible_flag(flag):
    try:
        return (flag == TerminationCondition.infeasible) or (str(flag).lower().find('infeasible')!=-1)
    except:
        return str(flag).lower() in ('infeasible', 'locallyinfeasible')



###########################################################
#       Algoritmo de Bound Contraction Particionado       #
###########################################################  

###################################################
iter = 1

print(" ", file = output_file)
print("+" * 100, file = output_file)
print(f"Iteracao: {iter}", file = output_file)  
print("+" * 100, file = output_file)
print(" ", file = output_file)
##################################################

print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Particione todas as variaveis", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)

# --------------------------------------------------------------------------------------------------- # 
for var_name in variaveis_limites.keys():
    print(f"Variável {var_name}:", file=output_file)
    print("-" * 50, file=output_file)
    
    # Check if this variable has the expected structure
    if 'lower' not in variaveis_limites[var_name] or 'upper' not in variaveis_limites[var_name]:
        print(f"  Estrutura de dados incompleta para {var_name}", file=output_file)
        continue
    
    # Determine how many stages this variable actually has
    num_stages = min(len(variaveis_limites[var_name]['lower']), 
                     len(variaveis_limites[var_name]['upper']))
    
    for estagio in range(num_stages):
        # Skip specific invalid combinations if needed
        if (var_name == 'V' and estagio == 0) or (var_name == 'L' and estagio == Ns - 1):
            continue
            
        try:
            # Obtém os limites para este estágio
            lower = variaveis_limites[var_name]['lower'][estagio]
            upper = variaveis_limites[var_name]['upper'][estagio]
            
            # Obtém o número de partições para esta variável
            card = Card_var[var_name]
            
            # Calcula os pontos de discretização
            pontos = []
            for i in range(1, card + 1):
                valor = calcula_variaveld(i, card, [lower, upper])
                pontos.append(valor)
            
            # Formata os pontos para impressão
            pontos_formatados = [f"{p:.7f}" for p in pontos]
            
            # Imprime os resultados
            print(f"Estágio {estagio+1}: [{', '.join(pontos_formatados)}]", file=output_file)
        except (IndexError, KeyError, TypeError) as e:
            print(f"  Erro ao processar {var_name} no estágio {estagio+1}: {str(e)}", file=output_file)
            continue
    
    print("", file=output_file)  # Linha em branco entre variáveis
# ------------------------------------------------------------------------------------------------------ #

print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Resolver o modelo de limite inferior (LB)", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)

resultados_LB = solve_LB(variaveis_limites, Card_var)
Fobj_LB = resultados_LB['objective_value']

# Verificar se a solução foi ótima
if resultados_LB['termination_condition'] == TerminationCondition.optimal:
    print("="*50, file = output_file)
    print("RESULTADOS LB", file = output_file)
    print("="*50, file = output_file)

    print(" ", file = output_file)    
    print(f"status = {resultados_LB['termination_condition']}", file = output_file)
    print(" ", file = output_file)
    print(f"Fobj_LB = {Fobj_LB:.6f}", file = output_file)
    print(" ", file = output_file)
        
    # Imprimir objetivo e cargas térmicas
    print(f"\nCarga do refervedor (Qr): {resultados_LB['Qr']:.2f} kJ/h", file = output_file)
    print(f"Carga do condensador (Qc): {resultados_LB['Qc']:.2f} kJ/h", file = output_file)
    
    # Imprimir temperaturas
    print("\nTemperaturas por estágio:", file = output_file)
    for j, T_val in enumerate(resultados_LB['T'], start=1):
        print(f"Estágio {j}: {T_val:.2f} K", file = output_file)
    
    # Imprimir vazões líquidas
    print("\nVazões líquidas por estágio:", file = output_file)
    for j, L_val in enumerate(resultados_LB['L'], start=1):
        if not np.isnan(L_val):
            print(f"Estágio {j}: {L_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Estágio {j}: Não definido", file = output_file)
    
    # Imprimir vazões de vapor
    print("\nVazões de vapor por estágio:", file = output_file)
    for j, V_val in enumerate(resultados_LB['V'], start=1):
        if not np.isnan(V_val):
            print(f"Estágio {j}: {V_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Estágio {j}: Não definido", file = output_file)
    
    # Imprimir composições
    print("\nComposições de Benzeno (x1) por estágio:", file = output_file)
    for j, x1_val in enumerate(resultados_LB['x'][1], start=1):
        print(f"Estágio {j}: {x1_val:.6f}", file = output_file)
    
    print("\nComposições de Tolueno (x2) por estágio:", file = output_file)
    for j, x2_val in enumerate(resultados_LB['x'][2], start=1):
        print(f"Estágio {j}: {x2_val:.6f}", file = output_file)

    print("\nVariáveis binárias lambda:", file = output_file)
    for j in resultados_LB['lambda']:
         print(f"Estágio {j}: {resultados_LB['lambda'][j]}", file = output_file)

    print("\nVariáveis binárias omega:", file = output_file)
    for j in resultados_LB['omega']:
         print(f"Estágio {j}: {resultados_LB['omega'][j]}", file = output_file)   
    
    print("\nVariáveis binárias rho:", file = output_file)
    for j in resultados_LB['rho']:
         print(f"Estágio {j}: {resultados_LB['rho'][j]}", file = output_file)    
    
else:
    print("O solver não convergiu para uma solução ótima.", file = output_file)
    print(f"Condição de término: {resultados_LB['termination_condition']}", file = output_file)



print(" ", file = output_file)
print("-" * 100, file = output_file)
print(f"Resolver o modelo UB", file = output_file)
print("-" * 100, file = output_file)
print(" ", file = output_file)


# Criar o vetor de inicialização para UB
UB_init = {
    'T': resultados_LB['T'],  # Lista de temperaturas por estágio
    'L': resultados_LB['L'],  # Lista de vazões líquidas por estágio
    'V': resultados_LB['V'],  # Lista de vazões de vapor por estágio
    'x': resultados_LB['x'],  # Dicionário de composições por componente
    'Qc': resultados_LB['Qc'],  # Valor da carga do condensador
    'Qr': resultados_LB['Qr']   # Valor da carga do refervedor
}


resultados_UB = solve_UB(variaveis_limites, UB_init)

Fobj_UB = resultados_UB['objective_value']

# Verificar se a solução foi ótima
if resultados_UB['termination_condition'] == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
    
    print("="*50, file = output_file)
    print("RESULTADOS UB", file = output_file)
    print("="*50, file = output_file)
    
    # Imprimir objetivo e cargas térmicas
    print(f"\nCarga do refervedor (Qr): {resultados_UB['Qr']:.2f} kJ/h", file = output_file)
    print(f"Carga do condensador (Qc): {resultados_UB['Qc']:.2f} kJ/h", file = output_file)
    
    # Imprimir temperaturas
    print("\nTemperaturas por estágio:", file = output_file)
    for j, T_val in enumerate(resultados_UB['T'], start=1):
        print(f"Estágio {j}: {T_val:.2f} K", file = output_file)
    
    # Imprimir vazões líquidas
    print("\nVazões líquidas por estágio:", file = output_file)
    for j, L_val in enumerate(resultados_UB['L'], start=1):
        if not np.isnan(L_val):
            print(f"Estágio {j}: {L_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Estágio {j}: Não definido", file = output_file)
    
    # Imprimir vazões de vapor
    print("\nVazões de vapor por estágio:", file = output_file)
    for j, V_val in enumerate(resultados_UB['V'], start=1):
        if not np.isnan(V_val):
            print(f"Estágio {j}: {V_val:.2f} kmol/h", file = output_file)
        else:
            print(f"Estágio {j}: Não definido", file = output_file)
    
    # Imprimir composições
    print("\nComposições de Benzeno (x1) por estágio:", file = output_file)
    for j, x1_val in enumerate(resultados_UB['x'][1], start=1):
        print(f"Estágio {j}: {x1_val:.6f}", file = output_file)
    
    print("\nComposições de Tolueno (x2) por estágio:", file = output_file)
    for j, x2_val in enumerate(resultados_UB['x'][2], start=1):
        print(f"Estágio {j}: {x2_val:.6f}", file = output_file)

else:
    print("O solver não convergiu para uma solução ótima.", file = output_file)
    print(f"Condição de término: {resultados_UB['termination_condition']}", file = output_file)


var_UB = {
    'T': resultados_UB['T'],
    'L': resultados_UB['L'],
    'V': resultados_UB['V'],
}

# Variáveis binárias do LB
binarias_values = {
    'T': resultados_LB['lambda'],
    'L': resultados_LB['omega'],
    'V': resultados_LB['rho']
}

##################################################
# Verificar convergência
##################################################
gap = abs((Fobj_UB - Fobj_LB) / Fobj_UB) if Fobj_UB != 0 else float('inf')
print(" ", file=output_file)
print("=" * 100, file=output_file)
print(f"ITERAÇÃO {iter} - RESULTADOS", file=output_file)
print("=" * 100, file=output_file)
print(f"LB: {Fobj_LB:.6f}", file=output_file)
print(f"UB: {Fobj_UB:.6f}", file=output_file)
print(f"Gap: {gap:.6f} (Tol: {Tol})", file=output_file)
print("=" * 100, file=output_file)

while (abs((Fobj_UB-Fobj_LB)/Fobj_UB) >= Tol):
    # Dicionário para contar quantos estágios não sofreram contração para cada variável
    aux_var = {'L': 0, 'V': 0, 'T': 0}

    for var_name in ['L', 'V', 'T']:

        if var_name not in variaveis_limites or var_name not in var_UB:
            continue  # Skip this variable if it doesn't exist

        print(" ", file = output_file)
        print("-" * 100, file = output_file)
        print(f"Selecione uma variavel do conjunto de variaveis concavas.", file = output_file)
        print("-" * 100, file = output_file)

         # Determine valid stages for this variable
        valid_stages = min(
            len(variaveis_limites[var_name]['lower']),
            len(variaveis_limites[var_name]['upper']),
            len(var_UB[var_name])
        )

        for estagio in range(valid_stages):

            if (var_name == 'V' and estagio == 0) or (var_name == 'L' and estagio == Ns - 1):
                continue  # Skip specifically defined invalid combinations
            

            # Verifique se o intervalo já está muito estreito ANTES de tentar contrair
            current_width = variaveis_limites[var_name]['upper'][estagio] - variaveis_limites[var_name]['lower'][estagio]
            if current_width < min_interval_width:
                print(f"Intervalo para {var_name} no estágio {estagio+1} já está muito estreito ({current_width:.6f}). Não contrair.", file=output_file)
                continue  # Pula para o próximo estágio sem contrair        



            print(f"\nVariavel escolhida:  {var_name} no estágio {estagio+1}", file=output_file)
            
            # obtém particionamento do estágio atual
            lower = variaveis_limites[var_name]['lower'][estagio]
            upper = variaveis_limites[var_name]['upper'][estagio]
            
            # Obtém o número de partições para esta variável
            card = Card_var[var_name]
            pontos = [calcula_variaveld(i, card, [lower, upper]) for i in range(1, card+1)]

            # recupera vetor binário para esse estágio (se existir)
            bin_vec = binarias_values.get(var_name, {})
            # em sua implementação, provavelmente bin_vec é dicionário com índice de estágio começando em 1
            lambda_stage_value = None
            if isinstance(bin_vec, dict):
                # tentamos indexação por estagio+1
                lambda_stage_value = bin_vec.get(estagio+1, None)

            # Se não houver valor binário, pulamos (não temos informação de qual intervalo é proibido)
            if lambda_stage_value is None:
                continue

            # indice_proibido_lista: todos os indices cujo valor binário difere de 1 por Tol
            # Se sua função LB devolve vetor com valores reals, use a checagem com Tol
            # Entretanto aqui assumimos lambda_stage_value é uma lista/array ou um único valor (dependendo da sua estrutura).
            # Para compatibilidade: se for número, vou assumir que o índice proibido foi retornado em outra estrutura.
            # Tentaremos obter um índice proibido a partir do vetor completo (se disponível em binarias_values[var_name])
            indice_proibido_idx = None
            # Se bin_vec[estagio+1] for uma lista (p.ex. lista de lambdas por partição), detectamos o índice com valor ~=1
            if isinstance(bin_vec.get(estagio+1, None), (list, tuple, np.ndarray)):
                vec = list(bin_vec[estagio+1])
                # encontra posições onde lambda != 1 (com tol)
                indices = [i+1 for i, v in enumerate(vec) if (v >= 1+Tol) or (v <= 1-Tol)]
                if len(indices) > 0:
                    indice_proibido_idx = indices[0]
            else:
                # Caso LB retorne apenas um inteiro indicando a partição ativa:
                # Se a estrutura armazenou o indice proibido diretamente
                try:
                    # se for inteiro (1..Card), consideramos
                    v = int(bin_vec.get(estagio+1))
                    if 1 <= v <= card:
                        indice_proibido_idx = v
                except:
                    indice_proibido_idx = None

            if indice_proibido_idx is None:
                # sem informação suficiente, não processa este estágio
                continue

            P_inicial = pontos[0]
            P_final = pontos[-1]
            # valor UB atual para essa variável/estágio
            ub_val = var_UB[var_name][estagio]

            # se ub_val for None, pula
            if ub_val is None:
                continue

            dist_inicial = abs(P_inicial - ub_val)
            dist_final = abs(P_final - ub_val)

            aux_L = None
            aux_U = None

            # salva original para restaurar se necessário
            orig_lower = variaveis_limites[var_name]['lower'][estagio]
            orig_upper = variaveis_limites[var_name]['upper'][estagio]

            if dist_inicial <= dist_final:
                # eliminar extremos próximos ao início: movemos lower para ponto correspondente a D_linha-ésima partição
                aux_L = orig_lower
                # results[D_linha-1] (0-based)
                if (card-2) < len(pontos) and (card-2) >= 0:
                    variaveis_limites[var_name]['lower'][estagio] = pontos[card-2]
                else:
                    # fallback: mover para o segundo ponto
                    variaveis_limites[var_name]['lower'][estagio] = pontos[1] if len(pontos) > 1 else pontos[0]
            else:
                # eliminar extremos próximos ao final
                aux_U = orig_upper
                # results[1] (0-based index 1 é o segundo ponto)
                variaveis_limites[var_name]['upper'][estagio] = pontos[1] if len(pontos) > 1 else pontos[-1]

            
            print(" ", file = output_file)
            print("-" * 100, file=output_file)
            print("Execute o modelo LB, proibindo o intervalo que contem", file=output_file)
            print("a solucao de UB da variavel escolhida no passo 3.1. (e seus adjacentes)", file=output_file)
            print("-" * 100, file=output_file)
            print(" ", file=output_file)

            print(f"\nExecutar LB no intervalo {var_name} estágio {estagio+1} = [{variaveis_limites[var_name]['lower'][estagio]:.6f}, {variaveis_limites[var_name]['upper'][estagio]:.6f}]", file=output_file)
            print(f"Solução UB: {ub_val :.6f}", file=output_file)
            
            resultados_LB = solve_LB(variaveis_limites, Card_var)
            Fobj_LB = resultados_LB.get('objective_value', None)
            term_LB = resultados_LB.get('termination_condition', None)

            if is_optimal_flag(term_LB):
                print(f"Fobj_LB = {Fobj_LB:.7f}", file=output_file)
                print(f"(Fobj_LB > UB) = {Fobj_LB > Fobj_UB}", file=output_file)

            print(f"status LB: {term_LB}", file=output_file)

            if Fobj_LB is not None and Fobj_UB is not None and (Fobj_LB < Fobj_UB):
                # Não faz contração, apenas mantém os limites
                print("-"*100, file=output_file)
                print("LB é viável e LB < UB → NÃO CONTRAIR INTERVALOS.", file=output_file)
                print(f"Intervalo preservado = [{orig_lower:.8f}, {orig_upper:.8f}]", file=output_file)
                print("-"*100, file=output_file)

                # restaura os limites originais
                variaveis_limites[var_name]['lower'][estagio] = orig_lower
                variaveis_limites[var_name]['upper'][estagio] = orig_upper
                
                # Incrementa o contador para esta variável (não houve contração)
                aux_var[var_name] += 1
            
            if is_infeasible_flag(term_LB) or (Fobj_LB is not None and Fobj_UB is not None and (Fobj_LB > Fobj_UB)):
                # reverte o lado que alteramos e mantém o intervalo proibido (somente esse não eliminado)
                if aux_L is not None:
                    # restaurar lower e setar upper para o ponto D_linha (mantendo só intervalo proibido)
                    variaveis_limites[var_name]['lower'][estagio] = orig_lower
                    # upper -> ponto D_linha-1
                    if (card-2) < len(pontos):
                        variaveis_limites[var_name]['upper'][estagio] = pontos[card-2]
                else:
                    variaveis_limites[var_name]['upper'][estagio] = orig_upper
                    if 1 < len(pontos):
                        variaveis_limites[var_name]['lower'][estagio] = pontos[1]
                print("-"*100, file=output_file)
                print("Se LB e inviavel ou LB e viavel, mas LB > UB.", file=output_file)
                print("Elimine todos os intervalos, exceto o intervalo proibido", file=output_file)
                print(f"Intervalo que NAO foi eliminado = [{variaveis_limites[var_name]['lower'][estagio]:.8f}, {variaveis_limites[var_name]['upper'][estagio]:.8f}]", file=output_file)
                print("-"*100, file=output_file)


            print("*" * 50, file = output_file)

    # Verifica se precisa aumentar o número de partições para cada variável
    for var_name in ['L', 'V', 'T']:
        # Número de estágios válidos para esta variável
        num_estagios_validos = Ns - 1 if var_name in ['L', 'V'] else Ns
        
        if aux_var[var_name] == num_estagios_validos:
            print(" ", file = output_file)
            print("-" * 100, file = output_file)
            print(f"Nenhum intervalo foi eliminado para {var_name} e o criterio de parada nao foi atendido", file = output_file)
            print(f"deve-se aumentar o numero de intervalos para {var_name} e recomecar", file = output_file)
                            
            Card_var[var_name] += 1
            print(f"Card_{var_name} = {Card_var[var_name]}", file = output_file)
            print("-" * 100, file = output_file)  

    ####################################################
    iter += 1

    print(" ", file = output_file)
    print("+" * 100, file = output_file)
    print(f"Iteracao: {iter}", file = output_file)  
    print("+" * 100, file = output_file)
    print(" ", file = output_file)

    ###################################################

    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Particione todas as variaveis", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)

    # --------------------------------------------------------------------------------------------------- # 
    for var_name in variaveis_limites.keys():
        print(f"Variável {var_name}:", file=output_file)
        print("-" * 50, file=output_file)
        
        # Check if this variable has the expected structure
        if 'lower' not in variaveis_limites[var_name] or 'upper' not in variaveis_limites[var_name]:
            print(f"  Estrutura de dados incompleta para {var_name}", file=output_file)
            continue
        
        # Determine how many stages this variable actually has
        num_stages = min(len(variaveis_limites[var_name]['lower']), 
                        len(variaveis_limites[var_name]['upper']))
        
        for estagio in range(num_stages):
            # Skip specific invalid combinations if needed
            if (var_name == 'V' and estagio == 0) or (var_name == 'L' and estagio == Ns - 1):
                continue
                
            try:
                # Obtém os limites para este estágio
                lower = variaveis_limites[var_name]['lower'][estagio]
                upper = variaveis_limites[var_name]['upper'][estagio]
                
                # Obtém o número de partições para esta variável
                card = Card_var[var_name]
                
                # Calcula os pontos de discretização
                pontos = []
                for i in range(1, card + 1):
                    valor = calcula_variaveld(i, card, [lower, upper])
                    pontos.append(valor)
                
                # Formata os pontos para impressão
                pontos_formatados = [f"{p:.7f}" for p in pontos]
                
                # Imprime os resultados
                print(f"Estágio {estagio+1}: [{', '.join(pontos_formatados)}]", file=output_file)
            except (IndexError, KeyError, TypeError) as e:
                print(f"  Erro ao processar {var_name} no estágio {estagio+1}: {str(e)}", file=output_file)
                continue
        
        print("", file=output_file)  # Linha em blanco entre variáveis
    # ------------------------------------------------------------------------------------------------------ #

    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Resolver o modelo de limite inferior (LB)", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)


    resultados_LB = solve_LB(variaveis_limites, Card_var)
    Fobj_LB = resultados_LB['objective_value']

    # Verificar se a solução foi ótima
    if resultados_LB['termination_condition'] == TerminationCondition.optimal:
        print("="*50, file = output_file)
        print("RESULTADOS LB", file = output_file)
        print("="*50, file = output_file)

        print(" ", file = output_file)    
        print(f"status = {resultados_LB['termination_condition']}", file = output_file)
        print(" ", file = output_file)
        print(f"Fobj_LB = {Fobj_LB:.6f}", file = output_file)
        print(" ", file = output_file)
            
        # Imprimir objetivo e cargas térmicas
        print(f"\nCarga do refervedor (Qr): {resultados_LB['Qr']:.2f} kJ/h", file = output_file)
        print(f"Carga do condensador (Qc): {resultados_LB['Qc']:.2f} kJ/h", file = output_file)
        
        # Imprimir temperaturas
        print("\nTemperaturas por estágio:", file = output_file)
        for j, T_val in enumerate(resultados_LB['T'], start=1):
            print(f"Estágio {j}: {T_val:.2f} K", file = output_file)
        
        # Imprimir vazões líquidas
        print("\nVazões líquidas por estágio:", file = output_file)
        for j, L_val in enumerate(resultados_LB['L'], start=1):
            if not np.isnan(L_val):
                print(f"Estágio {j}: {L_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Estágio {j}: Não definido", file = output_file)
        
        # Imprimir vazões de vapor
        print("\nVazões de vapor por estágio:", file = output_file)
        for j, V_val in enumerate(resultados_LB['V'], start=1):
            if not np.isnan(V_val):
                print(f"Estágio {j}: {V_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Estágio {j}: Não definido", file = output_file)
        
        # Imprimir composições
        print("\nComposições de Benzeno (x1) por estágio:", file = output_file)
        for j, x1_val in enumerate(resultados_LB['x'][1], start=1):
            print(f"Estágio {j}: {x1_val:.6f}", file = output_file)
        
        print("\nComposições de Tolueno (x2) por estágio:", file = output_file)
        for j, x2_val in enumerate(resultados_LB['x'][2], start=1):
            print(f"Estágio {j}: {x2_val:.6f}", file = output_file)

        print("\nVariáveis binárias lambda:", file = output_file)
        for j in resultados_LB['lambda']:
            print(f"Estágio {j}: {resultados_LB['lambda'][j]}", file = output_file)

        print("\nVariáveis binárias omega:", file = output_file)
        for j in resultados_LB['omega']:
            print(f"Estágio {j}: {resultados_LB['omega'][j]}", file = output_file)   
        
        print("\nVariáveis binárias rho:", file = output_file)
        for j in resultados_LB['rho']:
            print(f"Estágio {j}: {resultados_LB['rho'][j]}", file = output_file)    
        
    else:
        print("O solver não convergiu para uma solução ótima.", file = output_file)
        print(f"Condição de término: {resultados_LB['termination_condition']}", file = output_file)



    print(" ", file = output_file)
    print("-" * 100, file = output_file)
    print(f"Resolver o modelo UB", file = output_file)
    print("-" * 100, file = output_file)
    print(" ", file = output_file)


    # Criar o vetor de inicialização para UB
    UB_init = {
        'T': resultados_LB['T'],  # Lista de temperaturas por estágio
        'L': resultados_LB['L'],  # Lista de vazões líquidas por estágio
        'V': resultados_LB['V'],  # Lista de vazões de vapor por estágio
        'x': resultados_LB['x'],  # Dicionário de composições por componente
        'Qc': resultados_LB['Qc'],  # Valor da carga do condensador
        'Qr': resultados_LB['Qr']   # Valor da carga do refervedor
    }


    resultados_UB = solve_UB(variaveis_limites, UB_init)

    Fobj_UB = resultados_UB['objective_value']

    # Verificar se a solução foi ótima
    if resultados_UB['termination_condition'] == (TerminationCondition.optimal) or (TerminationCondition.locallyOptimal):
        
        print("="*50, file = output_file)
        print("RESULTADOS UB", file = output_file)
        print("="*50, file = output_file)
        
        # Imprimir objetivo e cargas térmicas
        print(f"\nCarga do refervedor (Qr): {resultados_UB['Qr']:.2f} kJ/h", file = output_file)
        print(f"Carga do condensador (Qc): {resultados_UB['Qc']:.2f} kJ/h", file = output_file)
        
        # Imprimir temperaturas
        print("\nTemperaturas por estágio:", file = output_file)
        for j, T_val in enumerate(resultados_UB['T'], start=1):
            print(f"Estágio {j}: {T_val:.2f} K", file = output_file)
        
        # Imprimir vazões líquidas
        print("\nVazões líquidas por estágio:", file = output_file)
        for j, L_val in enumerate(resultados_UB['L'], start=1):
            if not np.isnan(L_val):
                print(f"Estágio {j}: {L_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Estágio {j}: Não definido", file = output_file)
        
        # Imprimir vazões de vapor
        print("\nVazões de vapor por estágio:", file = output_file)
        for j, V_val in enumerate(resultados_UB['V'], start=1):
            if not np.isnan(V_val):
                print(f"Estágio {j}: {V_val:.2f} kmol/h", file = output_file)
            else:
                print(f"Estágio {j}: Não definido", file = output_file)
        
        # Imprimir composições
        print("\nComposições de Benzeno (x1) por estágio:", file = output_file)
        for j, x1_val in enumerate(resultados_UB['x'][1], start=1):
            print(f"Estágio {j}: {x1_val:.6f}", file = output_file)
        
        print("\nComposições de Tolueno (x2) por estágio:", file = output_file)
        for j, x2_val in enumerate(resultados_UB['x'][2], start=1):
            print(f"Estágio {j}: {x2_val:.6f}", file = output_file)

    else:
        print("O solver não convergiu para uma solução ótima.", file = output_file)
        print(f"Condição de término: {resultados_UB['termination_condition']}", file = output_file)


    var_UB = {
        'T': resultados_UB['T'],
        'L': resultados_UB['L'],
        'V': resultados_UB['V'],
    }

    # Variáveis binárias do LB
    binarias_values = {
        'T': resultados_LB['lambda'],
        'L': resultados_LB['omega'],
        'V': resultados_LB['rho']
    }

    ##################################################
    # Verificar convergência
    ##################################################
    gap = abs((Fobj_UB - Fobj_LB) / Fobj_UB) if Fobj_UB != 0 else float('inf')
    print(" ", file=output_file)
    print("=" * 100, file=output_file)
    print(f"ITERAÇÃO {iter} - RESULTADOS", file=output_file)
    print("=" * 100, file=output_file)
    print(f"LB: {Fobj_LB:.6f}", file=output_file)
    print(f"UB: {Fobj_UB:.6f}", file=output_file)
    print(f"Gap: {gap:.6f} (Tol: {Tol})", file=output_file)
    print("=" * 100, file=output_file)



print(f"Solution Time: {time.time() - start_time:.2f} seconds", file=output_file)
# Fecha o arquivo de saída
output_file.close()




