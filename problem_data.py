
########################## Parameters that can be modified ###########################

# Initial number of partitions for each variable
# The number of intervals is Card_L - 1
Card_L = 3
Card_V = 3
Card_T = 3

Tol = 0.1 # Tolerance
########################################################################################


############################### Parametros do problema ###############################

Pr = 760  # Pressure mmHg

Feed = 100  # Feed kmol/h
z_feed = 0.5 # It is the same for both components

T_feed = 365.15  # Feed temperature K

H_feed = 44418.46 # Feed enthalpy (kJ/kmol)

# Antoine coefficients (Benzene and Toluene)
kk = {
        (1, 1): 13.985035, (1, 2): 1757.518860, (1, 3): -112.820780,
        (2, 1): 14.377564, (2, 2): 2126.285144, (2, 3): -109.249431
    }

liq_coeffs = {
            (1, 1): -32980.00, (1, 2): 142.6015, (1, 3): 0.140180, (1, 4): 0.00051393,
            (2, 1): -38420.00, (2, 2): 155.1981, (2, 3): -0.491422, (2, 4): 0.00138480
        }

vap_coeffs = {
            (1, 1): 74010.00, (1, 2): -32.7976, (1, 3): 0.541962, (1, 4): -0.00095783,
            (2, 1): 82650.00, (2, 2): -36.4012, (2, 3): -0.117715, (2, 4): 0.00015788
}

Dist = 50 # Distillate
Bott = 50  # Bottom product

Ns = 17   # Total number of stages (including condenser and reboiler)
Nf = 8    # Feeding stage (7 on the Aspen corresponds to stage 8 on our model)


Qcond_upper = 5250000.00
Qcond_lower = 3364500.00

Qreb_upper = 5002421.00
Qreb_lower = 3116921.00

x_upper = 1
x_lower = 0
########################################################################################




min_interval_width = 1e-2  # Minimum value allowed for the difference between upper and lower


