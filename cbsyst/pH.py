import numpy as np
from .uncertainties import negative_log10_preserve_type

pH_scales = ['pHtot', 'pHNBS', 'pHsws', 'pHfree']

def n_given(params):
    return sum(params.get(p) is not None for p in pH_scales)

def calc_fH(TempK, Sal):
    # Same as CO2SYS
    # Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    # v. 3, 1982 (p. 80)

    a, b, c, d = (1.2948, -2.036e-3, 4.607e-4, -1.475e-6)
    return a + b * TempK + (c + d * TempK) * Sal ** 2

def convert_scales(params):
    params.FREEtoTOT = negative_log10_preserve_type((1 + params.ST / params.Ks.KS))
    params.SWStoTOT = negative_log10_preserve_type((1 + params.ST / params.Ks.KS) / (1 + params.ST / params.Ks.KS + params.FT / params.Ks.KF))
    params.fH = calc_fH(params.T_in + 273.15, params.S_in)

    pH_scales = ['pHtot', 'pHfree', 'pHNBS', 'pHsws']
    pH_present = [ph for ph in pH_scales if params[ph] is not None]

    if len(pH_present) == 0:
        return

    match pH_present[0]:
        case 'pHtot':
            params.pHfree = params.pHtot - params.FREEtoTOT
            params.pHsws = params.pHtot - params.SWStoTOT
            params.pHNBS = params.pHsws - np.log10(params.fH)
        case 'pHsws':
            params.pHNBS = params.pHsws - np.log10(params.fH)
            params.pHtot = params.pHsws + params.SWStoTOT
            params.pHfree = params.pHtot - params.FREEtoTOT
        case 'pHfree':
            params.pHtot = params.pHfree + params.FREEtoTOT
            params.pHsws = params.pHtot - params.SWStoTOT
            params.pHNBS = params.pHsws - np.log10(params.fH)
        case 'pHNBS':
            params.pHsws = params.pHNBS + np.log10(params.fH)
            params.pHtot = params.pHsws + params.SWStoTOT
            params.pHfree = params.pHtot - params.FREEtoTOT