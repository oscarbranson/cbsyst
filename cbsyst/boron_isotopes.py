# B isotope fns

import numpy as np
from cbsyst.helpers import ch, cp, NnotNone, Bunch
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
    return alpha_to_epsilon(get_alphaB())

def alpha_to_epsilon(alphaB):
    """
    Convert alpha to epsilon (which is alpha in delta space)

    Parameters
    ----------
    alphaB : array-like
        The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns
    -------
    array-like
        alphaB expressed in delta notation (AKA epsilonB).
    """
    return (alphaB-1)*1000

def epsilon_to_alpha(epsilonB):
    """
    Convert epsilon to alpha

    Parameters
    ----------
    epsilonB : array-like
        The isotope fractionation factor (11/10 BO3)/(11/10 BO4) expressed in delta notation.

    Returns
    -------
    array-like
        The isotope fractionation factor (11/10 BO3)/(11/10 BO4).
    """
    return (epsilonB/1000)+1


# Isotope Unit Converters
def A11_to_d11(A11, SRM_ratio=4.04367):
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

def A11_to_R11(A11):
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

def d11_to_A11(d11, SRM_ratio=4.04367):
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

def d11_to_R11(d11, SRM_ratio=4.04367):
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

def R11_to_d11(R11, SRM_ratio=4.04367):
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

def R11_to_A11(R11):
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

    Parameters
    ----------
    ABO3 : array-like
        The fractional abundance of boric acid (B(OH)3)
    alphaB : array-like
        The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns
    -------
    array-like
        ABO4 - the fractional abundance of borate ion (B(OH)4)
    """
    return (1 / ((alphaB / ABO3) - alphaB + 1) )

def ABO3_or_ABO4(ABO3,ABO4,alphaB):
    """
    Helper function to determine ABO4 if ABO3 is None

    Parameters
    ----------
    ABO3 : array-like
        The fractional abundance of boric acid (B(OH)3)
    ABO4 : array-like
        The fractional abundance of borate ion (B(OH)4)
    alphaB : array-like
        The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns
    -------
    array-like
        ABO4 - the fractional abundance of borate ion (B(OH)4)
    """
    if NnotNone(ABO3, ABO4) < 1:
        raise(ValueError("Either ABO4 or ABO3 must be specified"))
    elif ABO4 is None:
        ABO4 = ABO3_to_ABO4(ABO3,alphaB)
    return ABO4


# Base Functions
# Calculate total boron isotope fractional abundance using borate ion (B(OH)4)
def calculate_ABT(H, Ks, alphaB, ABO4=None, ABO3=None):
    """
    Calculate ABT from pH (total scale) and ABO4 or ABO3.

    Parameters
    ----------    
    Ks : dict
        A dictionary of stoichiometric equilibrium constants.
    pH : array-like
        pH on the Total scale
    alphaB : array-like
        The fractionation factor between B(OH)3 and B(OH)4-
    ABO4 : array-like
        The fractional abundance of 11B in B(OH)4.
    ABO3 : array-like
        The fractional abundance of 11B in B(OH)3.

    Returns
    -------
    array-like
        The fractional abundance of 11B in total B (ABT).
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

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
def calculate_H(Ks, alphaB, ABT, ABO4=None, ABO3=None):
    """
    Calculate H from ABO4 or ABO3 and ABT. 

    Parameters
    ----------
    Ks : dict
        dictionary of speciation constants
    alphaB : float or array-like
        fractionation factor between B(OH)3 and B(OH)4-
    ABT : float or array-like
        fractional abundance of 11B in total B
    ABO4 : float or array-like
        fractional abundance of 11B in B(OH)4-
        
    Returns
    -------
    array-like
        pH on the total scale.
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

    return (Ks.KB / ((alphaB / (1 - ABO4 + alphaB * ABO4) - 1) / (ABT / ABO4 - 1) - 1))

# Calculate isotope fractional abundance of boric acid (B(OH)3)
def calculate_ABO3(H, Ks, ABT, alphaB):
    """
    Calculate ABO3 from H and ABT

    Parameters
    ----------
    Ks : dict
        A dictionary of stoichiometric equilibrium constants.
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
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
def calculate_ABO4(H, Ks, ABT, alphaB):
    """
    Calculate ABO4 from H and ABT

    Parameters
    ----------
    Ks : dict
        Dictionary of stoichiometric equilibrium constants.
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
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
def calculate_alpha_ABO3(H, Ks, ABT, ABO3):
    """
    Calculate fractionation factor (alpha) from the fractional abundance of 11B in B(OH)3 (ABO3)

    Parameters
    ----------
    Ks : dict
        Dictionary of stoichiometric equilibrium constants.
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
    ABO3 : array-like
        The fractional abundance of 11B in boric acid (B(OH)3).

    Returns
    -------
    array-like
        The fractionation factor between B(OH)3 and B(OH)4- (alpha)
    """
    return ( (1
            / ((H/Ks.KB) * (ABT - ABO3) + ABT)) 
            / (ABO3 -1))

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_alpha_ABO4(H, Ks, ABT, ABO4):
    """
    Calculate fractionation factor (alpha) from the fractional abundance of 11B in B(OH)3 (ABO3)

    Parameters
    ----------
    Ks : dict
        Dictionary of stoichiometric equilibrium constants.
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    ABT : array-like
        The fractional abundance of 11B in total B.
    ABO4 : array-like
        The fractional abundance of 11B in borate ion (B(OH)4).

    Returns
    -------
    array-like
        The fractionation factor between B(OH)3 and B(OH)4- (alpha)
    """
    return ( (1/ABO4 - 1)
            / (1 / (ABT - ((ABO4-ABT)/(H/Ks.KB))) -1) )

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_KB(H, alphaB, ABT, ABO4=None, ABO3=None):
    """
    Calculate stoichiometric equilibrium constant for boron

    Parameters
    ----------
    H : array-like
        The activity of Hydrogen ions in mol kg-1
    alphaB : array-like
        The fractionation factor between B(OH)3 and B(OH)4-
    ABT : array-like
        The fractional abundance of 11B in total B.
    ABO4 : array-like
        The fractional abundance of 11B in borate ion (B(OH)4).
    ABO3 : array-like
        The fractional abundance of 11B in boric acid (B(OH)3).

    Returns
    -------
    array-like
        The stoichiometric equilibrium constant for boron (KB)
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)
    return (H
            / ((ABO4 - ABT)
            / ( ABT 
            - 1 / ( (1/alphaB) * (1/ABO4 -1) + 1) )))

def calc_B_isotopes(pHtot=None, ABT=None, ABO3=None, ABO4=None, alphaB=None, Ks=None, **kwargs):
    # determine pH and ABT
    if pHtot is not None:  # pH is known
        H = ch(pHtot)
        if ABT is None:
            ABT = calculate_ABT(H=H, Ks=Ks, alphaB=alphaB, ABO3=ABO3, ABO4=ABO4)
    else:  # pH is not known
        if ABT is not None:
            H = calculate_H(Ks=Ks, alphaB=alphaB, ABT=ABT, ABO3=ABO3, ABO4=ABO4)
            pHtot = cp(H)
        else:
            raise ValueError('ABT and one of ABO3 or ABO4 must be specified if pH is missing.')
    
    if ABO3 is None:
        ABO3 = calculate_ABO3(H=H, Ks=Ks, ABT=ABT, alphaB=alphaB)
    if ABO4 is None:
        ABO4 = calculate_ABO4(H=H, Ks=Ks, ABT=ABT, alphaB=alphaB)
    
    return Bunch({
        'pHtot': pHtot,
        'ABT': ABT,
        'ABO4': ABO4,
        'ABO3': ABO3,
        'H': H
    })

# Wrapper functions using delta values
def calculate_pH(Ks, d11BT, d11B4, epsilon=get_epsilonB()):
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
    alphaB = epsilon_to_alpha(epsilon)

    return cp(calculate_H(Ks,alphaB,ABT,ABO4))

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
    H = ch(pH)

    alphaB = epsilon_to_alpha(epsilonB)

    return cp(calculate_KB(H,alphaB,ABT,ABO4))

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
    H = ch(pH)
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

    return A11_to_d11(calculate_ABO4(ch(pH),KB,ABOT,alphaB))

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
    H = ch(pH)

    alphaB = calculate_alpha_ABO4(H,KB,ABT,ABO4)

    return alpha_to_epsilon(alphaB)
