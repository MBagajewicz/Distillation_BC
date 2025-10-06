
"""

    Calculates the value of the variable for a specific discretization point

"""

def chapel_variable(ord_d, Card_D, bounds):
    return bounds[0] + ((bounds[1] - bounds[0]) / (Card_D - 1)) * (ord_d - 1)
