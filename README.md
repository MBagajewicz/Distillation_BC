This repository contains a code that solves MESH equations for distillation columns using a combination of Bound Contraction and a Newton solver


# Bound Contraction Algorithm for Distillation Column Optimization

## Project Structure

### Main Files:

- `main.py`: Main file that implements the Bound Contraction algorithm.
- `resolve_LB.py`: Solves the Lower Bound (LB) model using a MILP model.
- `resolve_UB.py`: Solves the Upper Bound (UB) model using an NLP model.
- `problem_data.py`: Physical parameters and specifications of the distillation column.
- `aspen_data.py`: Reference data from the Aspen Plus simulation.
- `calculate_chapel_variable.py`: Variable discretization function.
- `Bound_contraction.txt`: Description of the algorithm's methodology.

## Column Specifications

- Number of stages (Ns): 17 (including condenser and reboiler)
- Feed stage (Nf): 8
- Pressure: 760 mmHg
- Components: Benzene (1) and Toluene (2)

## Optimization Variables

- Temperatures (T): 350K to 385K in each stage
- Liquid flow rates (L): 50 to 190 kmol/h
- Vapor flow rates (V): 120 to 140 kmol/h

## Mathematical Models

### Lower Bound (LB) Model

- MILP (Mixed-Integer Linear Programming) model that underestimates the original objective function.
- Uses discretization and binary variables to represent the partitions.
- Solved with the CPLEX solver via GAMS.

### Upper Bound (UB) Model

- NLP (Nonlinear Programming) model that represents the original problem.
- Uses exact nonlinear expressions for vapor-liquid equilibrium and energy balances.
- Solved with the CONOPT solver via GAMS.

## How to Run

### Prerequisites

- Python 3.x
- Pyomo
- GAMS with CPLEX and CONOPT solvers

### Running

1. Clone the repository or download the files.
2. Make sure the GAMS solvers (CPLEX and CONOPT) are installed and licensed.

### Configuração
Modifique o arquivo problem_data.py para ajustar:

1. Número de intervalos de discretização (Card_L, Card_V, Card_T)
2. Tolerância de convergência (Tol)



  
