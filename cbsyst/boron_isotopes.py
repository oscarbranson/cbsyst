# B isotope fns

import numpy as np
from cbsyst.helpers import ch
from .boron import chiB_calc



def alphaB_calc(**kwargs):
    """
    Klochko alpha for B fractionation
    """
    return 1.0272


# def alphaB_calc(TempC):
#     """
#     Temperature-sensitive alpha from Honisch et al, 2008
#     """    
    # return 1.0293 - 0.000082 * TempC


# pH_ABO3 - ABT
def pH_ABO3(pH, ABO3, Ks, alphaB):
    """
    Calculates ABT from pHtot and ABO3.

    Parameters
    ----------
    pH : array-like
        pH on the Total scale
    ABO3 : arra-like
        The fractional abundance of 11B in B(OH)3.
    Ks : dict
        A dictionary of stoichiometric equilibrium constants.
    alphaB : array-like
        The fractionation factor between BO3 and BO4.

    Returns
    -------
    array-like
        The fractional abundance of 11B in total B (ABT).
    """
    H = ch(pH)
    chiB = chiB_calc(H, Ks)
    return (
        ABO3
        * (-ABO3 * alphaB * chiB + ABO3 * chiB + alphaB * chiB - chiB + 1)
        / (-ABO3 * alphaB + ABO3 + alphaB)
    )


# pH_ABO4 - ABT
def pH_ABO4(pH, ABO4, Ks, alphaB):
    """
    Calculates ABT from pHtot and ABO4.

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
        / (ABO4 * alphaB - ABO4 + 1)
    )

# ABO4_ABT - pH
def ABO4_ABT(ABO4, ABT, Ks, alphaB):
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
    return -np.log10(Ks.KB / ((alphaB / (1 - ABO4 + alphaB * ABO4) - 1) / (ABT / ABO4 - 1) - 1))

def cABO3(H, ABT, Ks, alphaB):
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


def cABO4(H, ABT, Ks, alphaB):
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


# Isotope Unit Converters
def A11_2_d11(A11, SRM_ratio=4.04367):
    """
    Convert Abundance to Delta notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return ((A11 / (1 - A11)) / SRM_ratio - 1) * 1000


def A11_2_R11(A11):
    """
    Convert Abundance to Ratio notation.
    """
    return A11 / (1 - A11)


def d11_2_A11(d11, SRM_ratio=4.04367):
    """
    Convert Delta to Abundance notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return SRM_ratio * (d11 / 1000 + 1) / (SRM_ratio * (d11 / 1000 + 1) + 1)


def d11_2_R11(d11, SRM_ratio=4.04367):
    """
    Convert Delta to Ratio notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return (d11 / 1000 + 1) * SRM_ratio


def R11_2_d11(R11, SRM_ratio=4.04367):
    """
    Convert Ratio to Delta notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return (R11 / SRM_ratio - 1) * 1000


def R11_2_A11(R11):
    """
    Convert Ratio to Abundance notation.
    """
    return R11 / (1 + R11)