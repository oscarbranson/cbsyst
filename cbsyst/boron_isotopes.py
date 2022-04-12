# B isotope fns

import numpy as np
from cbsyst.helpers import ch,Bunch
from .boron import chiB_calc

def get_alphaB():
    """
    Klochko alpha for B fractionation
    """
    return 1.0272
def get_epsilonB():
    """
    Klochko epsilon for B fractionation
    """
    return alpha_2_epsilon(get_alphaB())

def alpha_2_epsilon(alphaB):
    """
    Convert alpha to epsilon (which is alpha in delta space)
    """
    return (alphaB-1)*1000
def epsilon_2_alpha(epsilonB):
    """
    Convert epsilon to alpha
    """
    return (epsilonB/1000)+1

# Isotope Unit Converters
def A11_2_d11(A11, SRM_ratio=4.04367):
    """
    Convert fractional abundance (A11) to delta notation (d11).

    Parameters
    ----------
    A11 : array-like
        The fractional abundance of 11B: 11B / (11B + 10B).
    SRM_ratio : float, optional
        The 11B/10B of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
        A11 expressed in delta notation (d11).
    """
    return ((A11 / (1 - A11)) / SRM_ratio - 1) * 1000
def A11_2_R11(A11):
    """
    Convert fractional abundance (A11) to isotope ratio (R11).

    Parameters
    ----------
    A11 : array-like
        The fractional abundance of 11B: 11B / (11B + 10B).

    Returns
    -------
    array-like
        A11 expressed as an isotope ratio (R11).
    """
    return A11 / (1 - A11)
def d11_2_A11(d11, SRM_ratio=4.04367):
    """
    Convert delta notation (d11) to fractional abundance (A11).

    Parameters
    ----------
    d11 : array-like
        The isotope ratio expressed in delta notation.
    SRM_ratio : float, optional
        The 11B/10B of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
       Delta notation (d11) expressed as fractional abundance (A11).
    """
    return SRM_ratio * (d11 / 1000 + 1) / (SRM_ratio * (d11 / 1000 + 1) + 1)
def d11_2_R11(d11, SRM_ratio=4.04367):
    """
    Convert delta notation (d11) to isotope ratio (R11).

    Parameters
    ----------
    d11 : array-like
        The isotope ratio expressed in delta notation.
    SRM_ratio : float, optional
        The 11B/10B of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
       Delta notation (d11) expressed as isotope ratio (R11).
    """
    return (d11 / 1000 + 1) * SRM_ratio
def R11_2_d11(R11, SRM_ratio=4.04367):
    """
    Convert isotope ratio (R11) to delta notation (d11).

    Parameters
    ----------
    R11 : array-like
        The isotope ratio (11B/10B).
    SRM_ratio : float, optional
        The 11B/10B of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
        R11 expressed in delta notation (d11).
    """
    return (R11 / SRM_ratio - 1) * 1000
def R11_2_A11(R11):
    """
    Convert isotope ratio (R11) to fractional abundance (A11).

    Parameters
    ----------
    R11 : array-like
        The isotope ratio (11B/10B).

    Returns
    -------
    array-like
        R11 expressed as fractional abundance (A11).
    """
    return R11 / (1 + R11)

# Alpha Converters
def ABO3_to_ABO4(ABO3,alphaB):
    """
    Converts isotope fractional abundance of boric acid to isotope fraction abundance of borate ion
    """
    return (1 / ((alphaB / ABO3) - alphaB + 1) )
def ABO3_or_ABO4(ABO3,ABO4,alphaB):
    """
    Helper function to determine ABO4 where necessary
    """
    if (ABO4 is None and ABO3 is None) or (ABO4 is not None and ABO3 is not None):
        raise(ValueError("Either ABO4 or ABO3 must be specified"))
    elif ABO4 is None and ABO3 is not None:
        ABO4 = ABO3_to_ABO4(ABO3,alphaB)
    return ABO4

# Base Functions
# Calculate total boron isotope fractional abundance using borate ion (B(OH)4)
def calculate_ABT(Ks, pH, alphaB, ABO4=None, ABO3=None):
    """
    Calculates ABT from pH (total scale) and ABO4.

    Parameters
    ----------
    pH : array-like
        pH on the Total scale
    ABO4 : array-like
        The fractional abundance of 11B in B(OH)3.
    Ks : dict
        A dictionary of stoichiometric equilibrium constants.
    alphaB : array-like
        The fractionation factor between B(OH)3 and B(OH)4-

    Returns
    -------
    array-like
        The fractional abundance of 11B in total B (ABT).
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

    H = ch(pH)
    chiB = chiB_calc(H, Ks)
    return (
        ABO4
        * (
            -ABO4 * alphaB * chiB
            + ABO4 * alphaB
            + ABO4 * chiB
            - ABO4
            + alphaB * chiB
            - chiB
            + 1
        )
        / (ABO4 * alphaB - ABO4 + 1))

# Calculate pH using isotope fractional abundance of borate ion (B(OH)4)
def calculate_ApH(Ks, alphaB, ABT, ABO4=None, ABO3=None):
    """
    Calculates pHtot from ABO4 and ABT. 

    Parameters
    ----------
    ABO4 : float or array-like
        fractional abundance of 11B in B(OH)4-
    ABT : float or array-like
        fractional abundance of 11B in total B
    Ks : dict
        dictionary of speciation constants
    alphaB : float or array-like
        fractionation factor between B(OH)3 and B(OH)4-
        
    Returns
    -------
    array-like
        pH on the Total scale.
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

    return -np.log10(Ks.KB / ((alphaB / (1 - ABO4 + alphaB * ABO4) - 1) / (ABT / ABO4 - 1) - 1))

# Calculate isotope fractional abundance of boric acid (B(OH)3)
def calculate_ABO3(H, ABT, Ks, alphaB):
    """
    Calculate ABO3 from H and ABT

    Parameters
    ----------
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
    Ks : dict
        A dictionary of stoichiometric equilibrium constants.
    alphaB : array-like
        The fractionation factor between B(OH)3 and B(OH)4-

    Returns
    -------
    array-like
        The fractional abundance of 11B in B(OH)3.
    """
    chiB = chiB_calc(H, Ks)
    return (
        ABT * alphaB
        - ABT
        + alphaB * chiB
        - chiB
        - np.sqrt(
            ABT ** 2 * alphaB ** 2
            - 2 * ABT ** 2 * alphaB
            + ABT ** 2
            - 2 * ABT * alphaB ** 2 * chiB
            + 2 * ABT * alphaB
            + 2 * ABT * chiB
            - 2 * ABT
            + alphaB ** 2 * chiB ** 2
            - 2 * alphaB * chiB ** 2
            + 2 * alphaB * chiB
            + chiB ** 2
            - 2 * chiB
            + 1
        )
        + 1
    ) / (2 * chiB * (alphaB - 1))

# Calculate isotope fractional abundance of borate ion (B(OH)4)
def calculate_ABO4(H, ABT, Ks, alphaB):
    """
    Calculate ABO4 from H and ABT

    Parameters
    ----------
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
    Ks : dict
        Dictionary of stoichiometric equilibrium constants.
    alphaB : array-like
        The fractionation factor between B(OH)3 and B(OH)4-

    Returns
    -------
    array-like
        The fractional abundance of 11B in B(OH)4-
    """
    chiB = chiB_calc(H, Ks)
    return -(
        ABT * alphaB
        - ABT
        - alphaB * chiB
        + chiB
        + np.sqrt(
            ABT ** 2 * alphaB ** 2
            - 2 * ABT ** 2 * alphaB
            + ABT ** 2
            - 2 * ABT * alphaB ** 2 * chiB
            + 2 * ABT * alphaB
            + 2 * ABT * chiB
            - 2 * ABT
            + alphaB ** 2 * chiB ** 2
            - 2 * alphaB * chiB ** 2
            + 2 * alphaB * chiB
            + chiB ** 2
            - 2 * chiB
            + 1
        )
        - 1
    ) / (2 * alphaB * chiB - 2 * alphaB - 2 * chiB + 2)

# Calculate alpha using isotope fractional abundance of boric acid (B(OH)3)
def calculate_alpha_ABO3(Ks,H,ABT,ABO3):
    return ( (1
            / ((H/Ks.KB) * (ABT - ABO3) + ABT)) 
            / (ABO3 -1))

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_alpha_ABO4(Ks,H,ABT,ABO4):
    return ( (1 - ABO4)
            / (ABO4 / (ABT - ((ABO4-ABT)/(H/Ks.KB))) -ABO4) )

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_AKB(H,alphaB,ABT,ABO4=None,ABO3=None):
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)
    return (H
            / ((ABO4 - ABT)
            / ( ABT 
            - 1 / ( (1/alphaB) * (1/ABO4 -1) + 1) )))

# Wrapper functions using delta values
def calculate_pH(Ks,d11BT,d11B4=None,d11B3=None,epsilon=get_epsilonB()):
    """
    Returns pH on the total scale

    Parameters
    ----------
    Ks : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units
    d11B4 : float or array-like
        isotope ratio 11B/10B in BO4 - delta units
    epsilon : float or array-like
        fractionation factor between BO3 and BO4
    """
    ABO4 = d11_2_A11(d11B4)
    ABT = d11_2_A11(d11BT)
    alpha = epsilon_2_alpha(epsilon)

    return calculate_ApH(Ks,alpha,ABT,ABO4)
def calculate_KB(d11B4,d11BT,pH,epsilon=get_epsilonB()):
    ABO4 = d11_2_A11(d11B4)
    ABT = d11_2_A11(d11BT)
    H = ch(pH)

    alphaB = epsilon_2_alpha(epsilon)

    return calculate_AKB(H,alphaB,ABT,ABO4)
def calculate_d11BT(d11B4,pH,KB,epsilon=get_epsilonB()):
    """
    Returns isotope ratio of total boron in delta units

    Parameters
    ----------
    d11B4 : float or array-like
        isotope ratio 11B/10B in BO4 - delta units
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    epsilon : float or array-like
        fractionation factor between BO3 and BO4
    """
    ABO4 = d11_2_A11(d11B4)
    alpha = epsilon_2_alpha(epsilon)
    return A11_2_d11(calculate_ABT(KB,pH,alpha,ABO4))
def calculate_d11B4(d11BT,pH,KB,epsilon=get_epsilonB()):
    """
    Returns isotope ratio of borate ion in delta units

    Parameters
    ----------
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    epsilon : float or array-like
        fractionation factor between BO3 and BO4
    """
    ABOT = d11_2_A11(d11BT)
    alpha = epsilon_2_alpha(epsilon)

    return A11_2_d11(calculate_ABO4(ch(pH),ABOT,KB,alpha))
def calculate_epsilon(d11B4,d11BT,pH,KB):
    """
    Returns isotope ratio of borate ion in delta units

    Parameters
    ----------
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units
    pH : float or array-like
        pH on the total scale
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    epsilon : float or array-like
        fractionation factor between BO3 and BO4
    """
    ABO4 = d11_2_A11(d11B4)
    ABT = d11_2_A11(d11BT)
    H = ch(pH)

    alpha = calculate_alpha_ABO4(KB,H,ABT,ABO4)

    return alpha_2_epsilon(alpha)
