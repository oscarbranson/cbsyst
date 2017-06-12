import numpy as np
from cbsyst.helpers import ch


# B conc fns
def BT_BO3(BT, BO3, Ks):
    """
    Returns H
    """
    return Ks.KB / (BT / BO3 - 1)


def BT_BO4(BT, BO4, Ks):
    """
    Returns H
    """
    return Ks.KB * (BT / BO4 - 1)


def pH_BO3(pH, BO3, Ks):
    """
    Returns BT
    """
    H = ch(pH)
    return BO3 * (1 + Ks.KB / H)


def pH_BO4(pH, BO4, Ks):
    """
    Returns BT
    """
    H = ch(pH)
    return BO4 * (1 + H / Ks.KB)


def cBO4(BT, H, Ks):
    return BT / (1 + H / Ks.KB)


def cBO3(BT, H, Ks):
    return BT / (1 + Ks.KB / H)


# B isotope fns
def chiB_calc(H, Ks):
    return 1 / (1 + Ks.KB / H)


def alphaB_calc(TempC):
    """
    Temperature-sensitive alpha from Honisch et al, 2008
    """
    return 1.0293 - 0.000082 * TempC


# pH_ABO3 - ABT
def pH_ABO3(pH, ABO3, Ks, alphaB):
    """
    Returns ABT
    """
    H = ch(pH)
    chiB = chiB_calc(H, Ks)
    return (ABO3 * (-ABO3 * alphaB * chiB +
                    ABO3 * chiB + alphaB *
                    chiB - chiB + 1) /
            (-ABO3 * alphaB + ABO3 + alphaB))


# pH_ABO4 - ABT
def pH_ABO4(pH, ABO4, Ks, alphaB):
    """
    Returns ABT
    """
    H = ch(pH)
    chiB = chiB_calc(H, Ks)
    return (ABO4 * (-ABO4 * alphaB * chiB +
                    ABO4 * alphaB + ABO4 *
                    chiB - ABO4 + alphaB *
                    chiB - chiB + 1) /
            (ABO4 * alphaB - ABO4 + 1))


def cABO3(H, ABT, Ks, alphaB):
    chiB = chiB_calc(H, Ks)
    return ((ABT * alphaB - ABT + alphaB *
            chiB - chiB -
            np.sqrt(ABT ** 2 * alphaB ** 2 - 2 *
                    ABT ** 2 * alphaB + ABT ** 2 -
                    2 * ABT * alphaB ** 2 * chiB +
                    2 * ABT * alphaB + 2 * ABT *
                    chiB - 2 * ABT + alphaB ** 2 *
                    chiB ** 2 - 2 * alphaB *
                    chiB ** 2 + 2 * alphaB * chiB +
                    chiB ** 2 - 2 * chiB + 1) + 1) /
            (2 * chiB * (alphaB - 1)))


def cABO4(H, ABT, Ks, alphaB):
    chiB = chiB_calc(H, Ks)
    return (-(ABT * alphaB - ABT - alphaB *
            chiB + chiB +
            np.sqrt(ABT ** 2 * alphaB ** 2 - 2 *
                    ABT ** 2 * alphaB + ABT ** 2 -
                    2 * ABT * alphaB ** 2 * chiB +
                    2 * ABT * alphaB + 2 * ABT *
                    chiB - 2 * ABT + alphaB ** 2 *
                    chiB ** 2 - 2 * alphaB *
                    chiB ** 2 + 2 * alphaB *
                    chiB + chiB ** 2 - 2 * chiB +
                    1) - 1) / (2 * alphaB * chiB - 2 *
                               alphaB - 2 * chiB + 2))


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
