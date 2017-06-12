import scipy.optimize as opt
import numpy as np
from cbsyst.helpers import ch, maxL, noms


# Zero-finders: 2-5, 10-15
# Algebraid: 1, 6-9

# Zeebe & Wolf-Gladrow, Appendix B
# 1. CO2 and pH given
def CO2_pH(CO2, pH, Ks):
    """
    Returns DIC
    """
    h = ch(pH)
    return CO2 * (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h**2)


# 2. CO2 and HCO3 given
def CO2_HCO3(CO2, HCO3, Ks):
    """
    Returns H
    """
    L = maxL(CO2, HCO3)  # find length of longest input
    CO2, HCO3 = noms(CO2, HCO3)  # get nominal values of inputs

    return opt.fsolve(zero_CO2_HCO3, [1] * L, args=(CO2, HCO3, Ks), xtol=1e-12)


def zero_CO2_HCO3(h, CO2, HCO3, Ks):
    # Roots: two negative, one positive - use positive.
    LH = CO2 * (h**2 + Ks.K1 * h + Ks.K1 * Ks.K2)
    RH = HCO3 * (h**2 + h**3 / Ks.K1 + Ks.K2 * h)
    return LH - RH


# 3. CO2 and CO3
def CO2_CO3(CO2, CO3, Ks):
    """
    Returns H
    """
    L = maxL(CO2, CO3)
    CO2, CO3 = noms(CO2, CO3)

    return opt.fsolve(zero_CO2_CO3, [1] * L, args=(CO2, CO3, Ks), xtol=1e-12)


def zero_CO2_CO3(h, CO2, CO3, Ks):
    # Roots: one positive, three negative. Use positive.
    LH = CO2 * (h**2 + Ks.K1 * h + Ks.K1 * Ks.K2)
    RH = CO3 * (h**2 + h**3 / Ks.K2 + h**4 / (Ks.K1 * Ks.K2))
    return LH - RH


# 4. CO2 and TA
def CO2_TA(CO2, TA, BT, Ks):
    """
    Returns H
    """
    L = maxL(CO2, TA, BT)
    CO2, TA, BT = noms(CO2, TA, BT)

    return opt.fsolve(zero_CO2_TA, [1] * L, args=(CO2, TA, BT, Ks), xtol=1e-12)


def zero_CO2_TA(h, CO2, TA, BT, Ks):
    # Roots: one pos, one neg, 2 conj. complex. Use positive
    LH = TA * h**2 * (Ks.KB + h)
    RH = (CO2 * (Ks.KB + h) * (Ks.K1 * h + 2 * Ks.K1 * Ks.K2) +
          h**2 * Ks.KB * BT + (Ks.KB + h) * (Ks.KW * h - h**3))
    return LH - RH


# 5. CO2 and DIC
def CO2_DIC(CO2, DIC, Ks):
    """
    Returns H
    """
    L = maxL(CO2, DIC)
    CO2, DIC = noms(CO2, DIC)

    return opt.fsolve(zero_CO2_DIC, [1] * L, args=(CO2, DIC, Ks), xtol=1e-12)


def zero_CO2_DIC(h, CO2, DIC, Ks):
    # Roots: one positive, one negative. Use positive.
    LH = DIC * h**2
    RH = CO2 * (h**2 + Ks.K1 * h + Ks.K1 * Ks.K2)
    return LH - RH


# 6. pH and HCO3
def pH_HCO3(pH, HCO3, Ks):
    """
    Returns DIC
    """
    h = ch(pH)
    return HCO3 * (1 + h / Ks.K1 + Ks.K2 / h)


# 7. pH and CO3
def pH_CO3(pH, CO3, Ks):
    """
    Returns DIC
    """
    h = ch(pH)
    return CO3 * (1 + h / Ks.K2 + h**2 / (Ks.K1 * Ks.K2))


# 8. pH and TA
def pH_TA(pH, TA, BT, Ks):
    """
    Returns CO2
    """
    h = ch(pH)
    return ((TA - Ks.KB * BT / (Ks.KB + h) - Ks.KW / h + h) /
            (Ks.K1 / h + 2 * Ks.K1 * Ks.K2 / h**2))


# 9. pH and DIC
def pH_DIC(pH, DIC, Ks):
    """
    Returns CO2
    """
    h = ch(pH)
    return DIC / (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h**2)


# 10. HCO3 and CO3
def HCO3_CO3(HCO3, CO3, Ks):
    """
    Returns H
    """
    L = maxL(HCO3, CO3)
    HCO3, CO3 = noms(HCO3, CO3)

    return opt.fsolve(zero_HCO3_CO3, [1] * L, args=(HCO3, CO3, Ks), xtol=1e-12)


def zero_HCO3_CO3(h, HCO3, CO3, Ks):
    # Roots: one pos, two neg. Use pos.
    LH = HCO3 * (h + h**2 / Ks.K1 + Ks.K2)
    RH = CO3 * (h + h**2 / Ks.K2 + h**3 / (Ks.K1 * Ks.K2))
    return LH - RH


# 11. HCO3 and TA
def HCO3_TA(HCO3, TA, BT, Ks):
    """
    Returns H
    """
    L = maxL(HCO3, TA, BT)
    HCO3, TA, BT = noms(HCO3, TA, BT)

    return opt.fsolve(zero_HCO3_TA, [1] * L, args=(HCO3, TA, BT, Ks), xtol=1e-12)


def zero_HCO3_TA(h, HCO3, TA, BT, Ks):
    # Roots: one pos, four neg. Use pos.
    LH = TA * (Ks.KB + h) * (h**3 + Ks.K1 * h**2 + Ks.K1 * Ks.K2 * h)
    RH = ((HCO3 * (h + h**2 / Ks.K1 + Ks.K2) *
           ((Ks.KB + 2 * Ks.K2) * Ks.K1 * h +
            2 * Ks.KB * Ks.K1 * Ks.K2 + Ks.K1 * h**2)) +
          ((h**2 + Ks.K1 * h + Ks.K1 * Ks.K2) *
           (Ks.KB * BT * h + Ks.KW * Ks.KB + Ks.KW * h - Ks.KB * h**2 - h**3)))
    return LH - RH


# 12. HCO3 amd DIC
def HCO3_DIC(HCO3, DIC, Ks):
    """
    Returns H
    """
    L = maxL(HCO3, DIC)
    HCO3, DIC = noms(HCO3, DIC)

    return opt.fsolve(zero_HCO3_DIC, [0] * L, args=(HCO3, DIC, Ks), xtol=1e-12)


def zero_HCO3_DIC(h, HCO3, DIC, Ks):
    # Roots: two pos. Use smaller.
    LH = HCO3 * (h + h**2 / Ks.K1 + Ks.K2)
    RH = h * DIC
    return LH - RH


# 13. CO3 and TA
def CO3_TA(CO3, TA, BT, Ks):
    """
    Returns H
    """
    L = maxL(CO3, TA, BT)
    CO3, TA, BT = noms(CO3, TA, BT)

    return opt.fsolve(zero_CO3_TA, [1] * L, args=(CO3, TA, BT, Ks), xtol=1e-12)


def zero_CO3_TA(h, CO3, TA, BT, Ks):
    # Roots: three neg, two pos. Use larger pos.
    LH = TA * (Ks.KB + h) * (h**3 + Ks.K1 * h**2 + Ks.K1 * Ks.K2 * h)
    RH = ((CO3 * (h + h**2 / Ks.K2 + h**3 / (Ks.K1 * Ks.K2)) *
           (Ks.K1 * h**2 + Ks.K1 * h * (Ks.KB + 2 * Ks.K2) + 2 * Ks.KB * Ks.K1 * Ks.K2)) +
          ((h**2 + Ks.K1 * h + Ks.K1 * Ks.K2) *
           (Ks.KB * BT * h + Ks.KW * Ks.KB + Ks.KW * h - Ks.KB * h**2 - h**3)))
    return LH - RH


# 14. CO3 and DIC
def CO3_DIC(CO3, DIC, Ks):
    """
    Returns H
    """
    L = maxL(CO3, DIC)
    CO3, DIC = noms(CO3, DIC)

    return opt.fsolve(zero_CO3_DIC, [1] * L, args=(CO3, DIC, Ks), xtol=1e-12)


def zero_CO3_DIC(h, CO3, DIC, Ks):
    # Roots: one pos, one neg. Use neg.
    LH = CO3 * (1 + h / Ks.K2 + h**2 / (Ks.K1 * Ks.K2))
    RH = DIC
    return LH - RH


# 15. TA and DIC
def TA_DIC(TA, DIC, BT, Ks):
    """
    Returns H
    """
    L = maxL(TA, DIC, BT)
    TA, DIC, BT = noms(TA, DIC, BT)

    return opt.fsolve(zero_TA_DIC, [1] * L, args=(TA, DIC, BT, Ks), xtol=1e-12)
    # iterate through each row individually - use cast_array and map?


def zero_TA_DIC(h, TA, DIC, BT, Ks):
    # Roots: one pos, four neg. Use pos.
    LH = DIC * (Ks.KB + h) * (Ks.K1 * h**2 + 2 * Ks.K1 * Ks.K2 * h)
    RH = ((TA * (Ks.KB + h) * h -
           Ks.KB * BT * h -
           Ks.KW * (Ks.KB + h) +
           (Ks.KB + h) * h**2) *
          (h**2 +
           Ks.K1 * h +
           Ks.K1 * Ks.K2))
    return LH - RH


# 1.1.9
def cCO2(H, DIC, Ks):
    """
    Returns CO2
    """
    return DIC / (1 + Ks.K1 / H + Ks.K1 * Ks.K2 / H**2)


# 1.1.10
def cHCO3(H, DIC, Ks):
    """
    Returns HCO3
    """
    return DIC / (1 + H / Ks.K1 + Ks.K2 / H)


# 1.1.11
def cCO3(H, DIC, Ks):
    """
    Returns CO3
    """
    return DIC / (1 + H / Ks.K2 + H**2 / (Ks.K1 * Ks.K2))


# 1.5.80
def cTA(CO2, H, BT, Ks):
    """
    Returns TA
    """
    return (CO2 * (Ks.K1 / H + 2 * Ks.K1 * Ks.K2 / H**2) +
            BT * Ks.KB / (Ks.KB + H) + Ks.KW / H - H)


# C.4.14
def fCO2_to_CO2(fCO2, Ks):
    """
    Calculate CO2 from fCO2
    """
    return fCO2 * Ks.K0


# C.4.14
def CO2_to_fCO2(CO2, Ks):
    """
    Calculate fCO2 from CO2
    """
    return CO2 / Ks.K0


# 1.4.65
def pCO2_to_fCO2(pCO2, Tc, atm=1):
    """
    Calculate fCO2 from pCO2
    """
    Tk = Tc + 273.15
    p = atm * 101325.  # convert atm to Pa
    R = 8.314  # Gas constant
    B = (-1636.75 + 12.040 * Tk - 3.27957e-2 * Tk**2 + 3.16528e-5 * Tk**3) * 1e-6
    delta = (57.7 - 0.118 * Tk) * 1e-6

    return pCO2 * np.exp(p * (B + 2 * delta) / (R * Tk))


# 1.4.65
def fCO2_to_pCO2(fCO2, Tc, atm=1):
    """
    Calculate pCO2 from fCO2
    """
    Tk = Tc + 273.15
    p = atm * 101325.  # convert atm to Pa
    R = 8.314  # Gas constant
    B = (-1636.75 + 12.040 * Tk - 3.27957e-2 * Tk**2 + 3.16528e-5 * Tk**3) * 1e-6
    delta = (57.7 - 0.118 * Tk) * 1e-6

    return fCO2 / np.exp(p * (B + 2 * delta) / (R * Tk))

