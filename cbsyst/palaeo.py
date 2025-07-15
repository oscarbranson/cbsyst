import numpy as np
from .boron_isotopes import (
    get_epsilonB,
    d11_to_A11, A11_to_d11,
    epsilon_to_alpha, alpha_to_epsilon,
    calculate_H, calculate_KB,
    calculate_ABT, calculate_ABO4,
    calculate_alpha_ABO4
)

# Wrapper functions using delta values
def calculate_pH(Ks, d11BT, d11B4, epsilonB=get_epsilonB()):
    """
    Calculates pH on the total scale

    Parameters
    ----------
    Ks : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units
    d11B4 : float or array-like
        isotope ratio 11B/10B in BO4 - delta units, in ‰
    epsilon : float or array-like
        fractionation factor between BO3 and BO4, in ‰

    Returns
    ----------
    array-like
        pH on the total scale
    """
    ABO4 = d11_to_A11(d11B4)
    ABT = d11_to_A11(d11BT)
    alphaB = epsilon_to_alpha(epsilonB)

    return -np.log10(calculate_H(Ks,alphaB,ABT,ABO4))

def calculate_pKB(pH, d11BT, d11B4, epsilonB=get_epsilonB()):
    """
    Calculate stoichiometric equilibrium constant for boron with delta inputs

    Parameters
    ----------
    pH : array-like
        pH on the total scale
    d11BT : array-like
        The isotope ratio of 11B in total B in delta units, in ‰
    d11B4 : array-like
        The isotope ratio of 11B in borate ion (B(OH)4) in delta units, in ‰
    epsilonB : array-like
        The fractionation factor between B(OH)3 and B(OH)4- as delta units, in ‰

    Returns
    -------
    array-like
        The stoichiometric equilibrium constant for boron (KB)
    """
    ABO4 = d11_to_A11(d11B4)
    ABT = d11_to_A11(d11BT)
    H = 10.0**-pH

    alphaB = epsilon_to_alpha(epsilonB)

    return -np.log10(calculate_KB(H,alphaB,ABT,ABO4))

def calculate_d11BT(pH, KB, d11B4, epsilonB=get_epsilonB()):
    """
    Calcluates the isotope ratio of total boron in delta units

    Parameters
    ----------
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    d11B4 : float or array-like
        isotope ratio 11B/10B in BO4 - delta units, in ‰
    epsilonB : float or array-like
        fractionation factor between BO3 and BO4, units of ‰

    Returns
    -------
    array-like
        The isotope ratio 11B/10B in BT - delta units (d11BT), in ‰
    """
    ABO4 = d11_to_A11(d11B4)
    alphaB = epsilon_to_alpha(epsilonB)
    H = 10.0**-pH
    return A11_to_d11(calculate_ABT(H,KB,alphaB,ABO4))

def calculate_d11B4(pH, KB, d11BT, epsilonB=get_epsilonB()):
    """
    Calculates the isotope ratio of borate ion in delta units

    Parameters
    ----------
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units, in ‰
    epsilonB : float or array-like
        fractionation factor between BO3 and BO4, units of ‰
    
    Returns
    -------
    array-like
        The isotope ratio 11B/10B in BO4 - delta units, in ‰
    """
    ABOT = d11_to_A11(d11BT)
    alphaB = epsilon_to_alpha(epsilonB)

    return A11_to_d11(calculate_ABO4(10.0**-pH,KB,ABOT,alphaB))

def calculate_epsilon(pH, KB, d11BT, d11B4):
    """
    Returns isotope ratio of borate ion in delta units

    Parameters
    ----------
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units, in ‰
    d11B4 : float or array-like
        isotope ratio 11B/10B in borate ion (B(OH)4) - delta units, in ‰
        
    Returns
    -------
    array-like
        fractionation factor between BO3 and BO4 in delta units (epsilon, in ‰)
    """
    ABO4 = d11_to_A11(d11B4)
    ABT = d11_to_A11(d11BT)
    H = 10.0**-pH

    alphaB = calculate_alpha_ABO4(H,KB,ABT,ABO4)

    return alpha_to_epsilon(alphaB)
