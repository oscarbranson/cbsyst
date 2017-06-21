import scipy.optimize as opt
import numpy as np
from cbsyst.helpers import ch, noms, cast_array, maxL
# from cbsyst.boron_fns import cBO4


def _zero_wrapper(ps, fn, bounds=(10**-14, 10**-1)):
    """
    Wrapper to handle zero finders.
    """
    try:
        return opt.brentq(fn, *bounds, args=tuple(ps), xtol=1e-16)
        # brentq is ~100 times faster.
    except ValueError:
        return opt.fsolve(fn, 1, args=tuple(ps))[0]
        # but can be fragile if limits aren't right.


# Function types
# Zero-finders: 2-5, 10-15
# Algebraic: 1, 6-9


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
    CO2, HCO3 = noms(CO2, HCO3)  # get nominal values of inputs
    par = cast_array(CO2, HCO3, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_HCO3)


def zero_CO2_HCO3(h, CO2, HCO3, K1, K2):
    # Roots: two negative, one positive - use positive.
    LH = CO2 * (h**2 + K1 * h + K1 * K2)
    RH = HCO3 * (h**2 + h**3 / K1 + K2 * h)
    return LH - RH


# 3. CO2 and CO3
def CO2_CO3(CO2, CO3, Ks):
    """
    Returns H
    """
    CO2, CO3 = noms(CO2, CO3)
    par = cast_array(CO2, CO3, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_CO3)


def zero_CO2_CO3(h, CO2, CO3, K1, K2):
    # Roots: one positive, three negative. Use positive.
    LH = CO2 * (h**2 + K1 * h + K1 * K2)
    RH = CO3 * (h**2 + h**3 / K2 + h**4 / (K1 * K2))
    return LH - RH


# 4. CO2 and TA
# def CO2_TA(CO2, TA, BT, Ks):
#     """
#     Returns H
#     """
#     CO2, TA, BT = noms(CO2, TA, BT)  # get nominal values of inputs
#     par = cast_array(CO2, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # cast parameters into array

#     return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_TA)
def CO2_TA(CO2, TA, BT, TP, TSi, TS, TF, Ks):
    """
    Returns pH

    Taken from matlab CO2SYS
    """
    fCO2 = CO2 / Ks.K0
    L = maxL(TA, CO2, BT, TP, TSi, TS, TF, Ks.K1)
    pHguess = 8.
    pHtol = 0.0000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while any(abs(deltapH) > pHtol):
        H = 10**-pHx
        HCO3 = Ks.K0 * Ks.K1 * fCO2 / H
        CO3 = Ks.K0 * Ks.K1 * Ks.K2 * fCO2 / H**2
        CAlk = HCO3 + 2 * CO3
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H**3
        PhosBot = H**3 + Ks.KP1 * H**2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        PAlk = TP * PhosTop / PhosBot
        SiAlk = TSi * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + TS / Ks.KSO4)
        HSO4 = TS / (1 + Ks.KSO4 / Hfree)
        HF = TF / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF
        Slope = ln10 * (HCO3 + 4. * CO3 + BAlk * H / (Ks.KB + H) + OH + H)
        deltapH = Residual / Slope

        while any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx


def zero_CO2_TA(h, CO2, TA, BT, K1, K2, KB, KW):
    # Roots: one pos, one neg, 2 conj. complex. Use positive
    LH = TA * h**2 * (KB + h)
    RH = (CO2 * (KB + h) * (K1 * h + 2 * K1 * K2) +
          h**2 * KB * BT + (KB + h) * (KW * h - h**3))
    return LH - RH


# 5. CO2 and DIC
def CO2_DIC(CO2, DIC, Ks):
    """
    Returns H
    """
    CO2, DIC = noms(CO2, DIC)  # get nominal values of inputs
    par = cast_array(CO2, DIC, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_DIC)


def zero_CO2_DIC(h, CO2, DIC, K1, K2):
    # Roots: one positive, one negative. Use positive.
    LH = DIC * h**2
    RH = CO2 * (h**2 + K1 * h + K1 * K2)
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
# def pH_TA(pH, TA, BT, Ks):
#     """
#     Returns CO2
#     """
#     h = ch(pH)
#     return ((TA - Ks.KB * BT / (Ks.KB + h) - Ks.KW / h + h) /
#             (Ks.K1 / h + 2 * Ks.K1 * Ks.K2 / h**2))
def pH_TA(pH, TA, BT, TP, TSi, TS, TF, Ks):
    """
    Returns DIC

    Taken directly from MATLAB CO2SYS.
    """
    H = 10**-pH
    # negative alk
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H**3
    PhosBot = H**3 + Ks.KP1 * H**2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = TP * PhosTop / PhosBot
    SiAlk = TSi * Ks.KSi / (Ks.KSi + H)
    # positive alk
    Hfree = H / (1 + TS / Ks.KSO4)
    HSO4 = TS / (1 + Ks.KSO4 / Hfree)
    HF = TF / (1 + Ks.KF / Hfree)
    CAlk = TA - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

    return CAlk * (H**2 + Ks.K1 * H + Ks.K1 * Ks.K2) / (Ks.K1 * (H + 2. * Ks.K2))


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
    HCO3, CO3 = noms(HCO3, CO3)  # get nominal values of inputs
    par = cast_array(HCO3, CO3, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_CO3)


def zero_HCO3_CO3(h, HCO3, CO3, K1, K2):
    # Roots: one pos, two neg. Use pos.
    LH = HCO3 * (h + h**2 / K1 + K2)
    RH = CO3 * (h + h**2 / K2 + h**3 / (K1 * K2))
    return LH - RH


# 11. HCO3 and TA
def HCO3_TA(HCO3, TA, BT, Ks):
    """
    Returns H
    """
    HCO3, TA, BT = noms(HCO3, TA, BT)  # get nominal values of inputs
    par = cast_array(HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_TA)


def zero_HCO3_TA(h, HCO3, TA, BT, K1, K2, KB, KW):
    # Roots: one pos, four neg. Use pos.
    LH = TA * (KB + h) * (h**3 + K1 * h**2 + K1 * K2 * h)
    RH = ((HCO3 * (h + h**2 / K1 + K2) *
           ((KB + 2 * K2) * K1 * h +
            2 * KB * K1 * K2 + K1 * h**2)) +
          ((h**2 + K1 * h + K1 * K2) *
           (KB * BT * h + KW * KB + KW * h - KB * h**2 - h**3)))
    return LH - RH


# 12. HCO3 amd DIC
def HCO3_DIC(HCO3, DIC, Ks):
    """
    Returns H
    """
    HCO3, DIC = noms(HCO3, DIC)  # get nominal values of inputs
    par = cast_array(HCO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_DIC)


def zero_HCO3_DIC(h, HCO3, DIC, K1, K2):
    # Roots: two pos. Use smaller.
    LH = HCO3 * (h + h**2 / K1 + K2)
    RH = h * DIC
    return LH - RH


# 13. CO3 and TA
def CO3_TA(CO3, TA, BT, Ks):
    """
    Returns H
    """
    CO3, TA, BT = noms(CO3, TA, BT)  # get nominal values of inputs
    par = cast_array(CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_TA)


def zero_CO3_TA(h, CO3, TA, BT, K1, K2, KB, KW):
    # Roots: three neg, two pos. Use larger pos.
    LH = TA * (KB + h) * (h**3 + K1 * h**2 + K1 * K2 * h)
    RH = ((CO3 * (h + h**2 / K2 + h**3 / (K1 * K2)) *
           (K1 * h**2 + K1 * h * (KB + 2 * K2) + 2 * KB * K1 * K2)) +
          ((h**2 + K1 * h + K1 * K2) *
           (KB * BT * h + KW * KB + KW * h - KB * h**2 - h**3)))
    return LH - RH


# 14. CO3 and DIC
def CO3_DIC(CO3, DIC, Ks):
    """
    Returns H
    """
    CO3, DIC = noms(CO3, DIC)  # get nominal values of inputs
    par = cast_array(CO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array

    return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_DIC)


def zero_CO3_DIC(h, CO3, DIC, K1, K2):
    # Roots: one pos, one neg. Use neg.
    LH = CO3 * (1 + h / K2 + h**2 / (K1 * K2))
    RH = DIC
    return LH - RH


# 15. TA and DIC
def TA_DIC(TA, DIC, BT, TP, TSi, TS, TF, Ks):
    """
    Returns pH

    Taken directly from MATLAB CO2SYS.
    """
    L = maxL(TA, DIC, BT, TP, TSi, TS, TF, Ks.K1)
    pHguess = 7.
    pHtol = 0.00000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while any(abs(deltapH) > pHtol):
        H = 10**-pHx
        # negative
        Denom = H**2 + Ks.K1 * H + Ks.K1 * Ks.K2
        CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H**3
        PhosBot = H**3 + Ks.KP1 * H**2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        PAlk = TP * PhosTop / PhosBot
        SiAlk = TSi * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + TS / Ks.KSO4)
        HSO4 = TS / (1 + Ks.KSO4 / Hfree)
        HF = TF / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

        Slope = ln10 * (DIC * Ks.K1 * H *
                        (H**2 +
                         Ks.K1 * Ks.K2 +
                         4 * H * Ks.K2) /
                        Denom / Denom +
                        BAlk * H / (Ks.KB + H) + OH + H)
        deltapH = Residual / Slope

        while any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx

# def TA_DIC(TA, DIC, BT, Ks):
#     """
#     Returns H
#     """
#     TA, DIC, BT = noms(TA, DIC, BT)  # get nominal values of inputs
#     par = cast_array(TA, DIC, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # cast parameters into array

#     return np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_TA_DIC)


def zero_TA_DIC(h, TA, DIC, BT, K1, K2, KB, KW):
    # Roots: one pos, four neg. Use pos.
    LH = DIC * (KB + h) * (K1 * h**2 + 2 * K1 * K2 * h)
    RH = ((TA * (KB + h) * h - KB * BT * h - KW * (KB + h) +
           (KB + h) * h**2) * (h**2 + K1 * h + K1 * K2))
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
# def cTA(CO2, H, BT, Ks, unit=1e6):
#     """
#     Returns TA
#     """
#     return (CO2 * (Ks.K1 / H + 2 * Ks.K1 * Ks.K2 / H**2) +
#             BT * Ks.KB / (Ks.KB + H) + unit * Ks.KW / H - H * unit)
def cTA(H, DIC, BT, TP, TSi, TS, TF, Ks, mode='multi'):
    # negative
    Denom = H**2 + Ks.K1 * H + Ks.K1 * Ks.K2
    CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H**3
    PhosBot = H**3 + Ks.KP1 * H**2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = TP * PhosTop / PhosBot
    SiAlk = TSi * Ks.KSi / (Ks.KSi + H)
    # positive
    Hfree = H / (1 + TS / Ks.KSO4)
    HSO4 = TS / (1 + Ks.KSO4 / Hfree)
    HF = TF / (1 + Ks.KF / Hfree)

    TA = CAlk + BAlk + OH + PAlk + SiAlk - Hfree - HSO4 - HF

    if mode == 'multi':
        return TA, CAlk, PAlk, SiAlk, OH
    else:
        return TA

# # 1.2.28
# def cTA(HCO3, CO3, BT, H, Ks):
#     """
#     Total Alkalinity
#     """
#     OH = Ks.KW / H
#     return HCO3 + 2 * CO3 + cBO4(BT, H, Ks) + OH - H


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

