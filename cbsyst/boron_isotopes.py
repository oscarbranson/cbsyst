# B isotope fns

import numpy as np
from cbsyst.helpers import ch,Bunch
from .boron import chiB_calc



def alphaB_calc(**kwargs):
    """
    Klochko alpha for B
    """
    return 1.0272

# pH_ABO3 - ABT
def pH_ABO3(pH, ABO3, Ks, alphaB):
    """
    Returns ABT
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
    Returns ABT
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
    Returns pHtot

    Parameters
    ----------
    ABO4 : float or array-like
        fractional abundance of 11B in BO4
    ABT : float or array-like
        fractional abundance of 11B in total B
    Ks : dict
        dictionary of speciation constants
    alphaB : float or array-like
        fractionation factor between BO3 and BO4
    """
    return -np.log10(Ks.KB / ((alphaB / (1 - ABO4 + alphaB * ABO4) - 1) / (ABT / ABO4 - 1) - 1))

def cABO3(H, ABT, Ks, alphaB):
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
def calpha(ABO4,ABT,H,Ks):
    return ( (H*ABT*(ABO4+1) + (Ks.KB*(ABO4-ABT))) / ((ABO4**2 * (H+Ks.KB)) + (ABO4*(H-(ABT*Ks.KB)))))


def calculate_pH(d11B4,d11BT,KB,epsilon=27.2):
    """
    Returns pH on the total scale

    Parameters
    ----------
    d11B4 : float or array-like
        isotope ratio 11B/10B in BO4 - delta units
    d11BT : float or array-like
        isotope ratio 11B/10B in total boron - delta units
    KB : Bunch (dictionary with . access)
        bunch containing the boron speciation constant KB
    epsilon : float or array-like
        fractionation factor between BO3 and BO4
    """
    ABO4 = d11_2_A11(d11B4)
    ABT = d11_2_A11(d11BT)
    alpha = epsilon_2_alpha(epsilon)

    return ABO4_ABT(ABO4,ABT,KB,alpha)

def calculate_d11BT(d11B4,pH,KB,epsilon=27.2):
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

    return A11_2_d11(pH_ABO4(pH,ABO4,KB,alpha))

def calculate_d11B4(d11BT,pH,KB,epsilon=27.2):
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

    return A11_2_d11(cABO4(ch(pH),ABOT,KB,alpha))

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
    ABOT = d11_2_A11(d11BT)
    H = ch(pH)

    alpha = calpha(ABO4,ABOT,H,KB)

    return alpha_2_epsilon(alpha)

    

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

def alpha_2_epsilon(alpha=None):
    if alpha is None:
        alpha = alphaB_calc()
    return (alpha-1)*1000
def epsilon_2_alpha(epsilon):
    return (epsilon/1000)+1