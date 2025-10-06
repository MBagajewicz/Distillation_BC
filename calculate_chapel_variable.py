
"""
    CHAPEL VARIABLE DISCRETIZATION FUNCTION
  
    Parameters:
    -----------
    ord_d : int
        Ordinal position of the discretization point (1-indexed).
        Must satisfy: 1 ≤ ord_d ≤ Card_D
    
    Card_D : int
        Total number of discretization points (cardinality of the set).
        Must be ≥ 2 to create a valid interval.
    
    bounds : list or tuple
        Lower and upper bounds of the continuous interval [lower, upper].
        The interval must be valid (upper > lower).
    
    Returns:
    --------
    float
        The discrete value at the specified ordinal position within the bounds.
    
        

    Mathematical Formulation:
    -------------------------
    The Chapel discretization uses linear interpolation:
        value = lower_bound + (upper_bound - lower_bound) / (Card_D - 1) * (ord_d - 1)
    
    This creates a uniform grid where:
        - When ord_d = 1: value = lower_bound
        - When ord_d = Card_D: value = upper_bound
        - Intermediate points are equally spaced
    """

def chapel_variable(ord_d, Card_D, bounds):
    return bounds[0] + ((bounds[1] - bounds[0]) / (Card_D - 1)) * (ord_d - 1)
