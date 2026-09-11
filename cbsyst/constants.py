import kgen
from .dataclasses import KValues, CBsystData
from .helpers import isnone
from . import dataclasses as dc


def calc_Ks(params: CBsystData) -> KValues:
    Ks = kgen.calc_Ks(temp_c=params.T_in, sal=params.S_in, p_bar=params.P_in, magnesium=params.Mg, calcium=params.Ca, sulphate=params.ST, fluorine=params.FT, MyAMI_mode=params.MyAMI_mode)

    return KValues(**Ks)

def calc_ST(Sal):
    """
    Calculate total Sulphur in mol/kg-SW- lifted directly from CO2SYS.m

    From Dickson et al., 2007, Table 2
    Note: Sal / 1.80655 = Chlorinity
    """
    return 0.14 * Sal / 1.80655 / 96.062 # mol/kg-SW


def calc_FT(Sal):
    """
    Calculate total Fluorine in mol/kg-SW

    From Dickson et al., 2007, Table 2
    Note: Sal / 1.80655 = Chlorinity
    """
    return 6.7e-5 * Sal / 1.80655 / 18.9984 # mol/kg-SW


# def calc_BT(Sal):
#     """
#     Calculate total Boron

#     Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.
#     Geochimica Et Cosmochimica Acta 74 (6): 1801-1811
#     """
#     a, b = (0.0004326, 35.)
#     return a * Sal / b


def calc_BT(Sal):
    """
    Calculate total Boron in mol/kg-SW - lifted directly from CO2SYS.m

    Directly from CO2SYS:
    Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    this is 0.000416 * Sal/35. = 0.0000119 * Sal
    TB(FF) = (0.000232 / 10.811) * (Sal / 1.80655) in mol/kg-SW
    """
    a, b = (0.0004157, 35.0)
    return a * Sal / b  # mol/kg-SW


def calc_fH(TempK, Sal):
    # Same as CO2SYS
    # Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    # v. 3, 1982 (p. 80)

    a, b, c, d = (1.2948, -2.036e-3, 4.607e-4, -1.475e-6)
    return a + b * TempK + (c + d * TempK) * Sal ** 2


def calc_conservative_composition(params: CBsystData):
    if isnone(params.ST):
        params.ST = calc_ST(params.S_in)
    if isnone(params.FT):
        params.FT = calc_FT(params.S_in)
    
    if isinstance(params, (dc.BoronParams, dc.BoronSystemParams, dc.CarbonBoronParams, dc.CarbonBoronIsotopeParams)):
        if isnone(params.BT) and (isnone(params.BO3) and isnone(params.BO4)):
            params.BT = calc_BT(params.S_in)
    else:
        if isnone(params.BT):
            params.BT = calc_BT(params.S_in)
