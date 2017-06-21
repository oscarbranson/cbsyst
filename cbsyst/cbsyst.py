import numpy as np
from cbsyst.helpers import Bunch, maxL
from cbsyst.MyAMI_V2 import MyAMI_K_calc, MyAMI_K_calc_multi
from cbsyst.carbon_fns import *
from cbsyst.boron_fns import *
from cbsyst.helpers import ch, cp, NnotNone, calc_TF, calc_TS
from cbsyst.non_MyAMI_constants import *


# Helper functions
# ----------------
def get_Ks(ps):
    """
    Helper function to calculate Ks.

    If ps.Ks is a dict, those Ks are used
    transparrently, with no pressure modification.
    """
    if isinstance(ps.Ks, dict):
        Ks = Bunch(ps.Ks)
    else:
        if maxL(ps.T, ps.S, ps.P, ps.Mg, ps.Ca) == 1:
            if ps.Mg is None:
                ps.Mg = 0.0528171
            if ps.Ca is None:
                ps.Ca = 0.0102821
                Ks = MyAMI_K_calc(ps.T, ps.S, P=ps.P)
        else:
            # if only Ca or Mg provided, fill in other with modern
            if ps.Mg is None:
                ps.Mg = 0.0528171
            if ps.Ca is None:
                ps.Ca = 0.0102821
            # calculate Ca and Mg specific Ks
            Ks = MyAMI_K_calc_multi(ps.T, ps.S, ps.Ca, ps.Mg, ps.P)

    # non-MyAMI Constants
    Ks.update(calc_KPs(ps.T, ps.S, ps.P))
    Ks.update(calc_KF(ps.T, ps.S, ps.P))
    Ks.update(calc_KSi(ps.T, ps.S, ps.P))

    return Ks


# C Speciation
# ------------
def Csys(pH=None, DIC=None, CO2=None,
         HCO3=None, CO3=None, TA=None,
         fCO2=None, pCO2=None,
         BT=433., Ca=None, Mg=None,
         T=25., S=35., P=None,
         TP=0., TSi=0.,
         Ks=None, pdict=None, unit='umol'):
    """
    Calculate the carbon chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    pH is Total scale.

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations, but:

    **WARNING** Error propagation NOT IMPLEMENTED for carbon system calculations
    with zero-finders (i.e. when pH is not given; cases 2-5 and 10-15).

    Parameters
    ----------
    pH, DIC, CO2, HCO3, CO3, TA : array-like
        Carbon system parameters. Two of these must be provided.
    BT : array-like
        Total B, used in Alkalinity calculations.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    unit : str
        Concentration units of C and B parameters (all must be in
        the same units).
        Can be 'mol', 'mmol', 'umol', 'nmol', 'pmol' or 'fmol'.
        Used in calculating Alkalinity. Default is 'umol'.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # convert unit to multiplier
    udict = {'mol': 1.,
             'mmol': 1.e3,
             'umol': 1.e6,
             'µmol': 1.e6,
             'nmol': 1.e9,
             'pmol': 1.e12,
             'fmol': 1.e15}
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]

    if ps.unit != 1:
        upar = ['DIC', 'TA', 'CO2', 'HCO3', 'CO3',
                'BT', 'fCO2', 'pCO2', 'TP', 'TSi']
        for p in upar:
            if ps[p] is not None:
                ps[p] = np.divide(ps[p], ps.unit)  # convert to molar

    ps.Ks = get_Ks(ps)

    # Conserved seawater chemistry
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S)

    # if fCO2 is given but CO2 is not, calculate CO2
    if ps.CO2 is None:
        if ps.fCO2 is not None:
            ps.CO2 = fCO2_to_CO2(ps.fCO2, ps.Ks)
        elif ps.pCO2 is not None:
            ps.CO2 = fCO2_to_CO2(pCO2_to_fCO2(ps.pCO2, ps.T), ps.Ks)

    # Carbon System Calculations (from Zeebe & Wolf-Gladrow, Appendix B)
    # 1. CO2 and pH
    if ps.CO2 is not None and ps.pH is not None:
        ps.H = ch(ps.pH)
        ps.DIC = CO2_pH(ps.CO2, ps.pH, ps.Ks)
    # 2. ps.CO2 and ps.HCO3
    elif ps.CO2 is not None and ps.HCO3 is not None:
        ps.H = CO2_HCO3(ps.CO2, ps.HCO3, ps.Ks)
        ps.DIC = CO2_pH(ps.CO2, cp(ps.H), ps.Ks)
    # 3. ps.CO2 and ps.CO3
    elif ps.CO2 is not None and ps.CO3 is not None:
        ps.H = CO2_CO3(ps.CO2, ps.CO3, ps.Ks)
        ps.DIC = CO2_pH(ps.CO2, cp(ps.H), ps.Ks)
    # 4. ps.CO2 and ps.TA
    elif ps.CO2 is not None and ps.TA is not None:
        # unit conversion because OH and H wrapped
        # up in TA fns - all need to be in same units.
        print('CO2_TA')
        ps.pH = CO2_TA(CO2=ps.CO2,
                       TA=ps.TA,
                       BT=ps.BT,
                       TP=ps.TP,
                       TSi=ps.TSi,
                       TS=ps.TS,
                       TF=ps.TF,
                       Ks=ps.Ks)
        ps.H = ch(ps.pH)
        ps.DIC = CO2_pH(ps.CO2, ps.pH, ps.Ks)
    # 5. ps.CO2 and ps.DIC
    elif ps.CO2 is not None and ps.DIC is not None:
        ps.H = CO2_DIC(ps.CO2, ps.DIC, ps.Ks)
    # 6. ps.pH and ps.HCO3
    elif ps.pH is not None and ps.HCO3 is not None:
        ps.H = ch(ps.pH)
        ps.DIC = pH_HCO3(ps.pH, ps.HCO3, ps.Ks)
    # 7. ps.pH and ps.CO3
    elif ps.pH is not None and ps.CO3 is not None:
        ps.H = ch(ps.pH)
        ps.DIC = pH_CO3(ps.pH, ps.CO3, ps.Ks)
    # 8. ps.pH and ps.TA
    elif ps.pH is not None and ps.TA is not None:
        ps.H = ch(ps.pH)
        ps.DIC = pH_TA(pH=ps.pH,
                       TA=ps.TA,
                       BT=ps.BT,
                       TP=ps.TP,
                       TSi=ps.TSi,
                       TS=ps.TS,
                       TF=ps.TF,
                       Ks=ps.Ks)
    # 9. ps.pH and ps.DIC
    elif ps.pH is not None and ps.DIC is not None:
        ps.H = ch(ps.pH)
    # 10. ps.HCO3 and ps.CO3
    elif ps.HCO3 is not None and ps.CO3 is not None:
        ps.H = HCO3_CO3(ps.HCO3, ps.CO3, ps.Ks)
        ps.DIC = pH_CO3(cp(ps.H), ps.CO3, ps.Ks)
    # 11. ps.HCO3 and ps.TA
    elif ps.HCO3 is not None and ps.TA is not None:
        Warning('Nutrient alkalinity not implemented for this input combination.\nCalculations use only C and B alkalinity.')
        ps.H = HCO3_TA(ps.HCO3,
                       ps.TA,
                       ps.BT,
                       ps.Ks)
        ps.DIC = pH_HCO3(cp(ps.H), ps.HCO3, ps.Ks)
    # 12. ps.HCO3 amd ps.DIC
    elif ps.HCO3 is not None and ps.DIC is not None:
        ps.H = HCO3_DIC(ps.HCO3, ps.DIC, ps.Ks)
    # 13. ps.CO3 and ps.TA
    elif ps.CO3 is not None and ps.TA is not None:
        Warning('Nutrient alkalinity not implemented for this input combination.\nCalculations use only C and B alkalinity.')
        ps.H = CO3_TA(ps.CO3,
                      ps.TA,
                      ps.BT,
                      ps.Ks)
        ps.DIC = pH_CO3(cp(ps.H), ps.CO3, ps.Ks)
    # 14. ps.CO3 and ps.DIC
    elif ps.CO3 is not None and ps.DIC is not None:
        ps.H = CO3_DIC(ps.CO3, ps.DIC, ps.Ks)
    # 15. ps.TA and ps.DIC
    elif ps.TA is not None and ps.DIC is not None:
        ps.pH = TA_DIC(TA=ps.TA,
                       DIC=ps.DIC,
                       BT=ps.BT,
                       TP=ps.TP,
                       TSi=ps.TSi,
                       TS=ps.TS,
                       TF=ps.TF,
                       Ks=ps.Ks)
        ps.H = ch(ps.pH)

    # The above makes sure that DIC and H are known,
    # this next bit calculates all the missing species
    # from DIC and H.
    if ps.CO2 is None:
        ps.CO2 = cCO2(ps.H, ps.DIC, ps.Ks)
    if ps.fCO2 is None:
        ps.fCO2 = CO2_to_fCO2(ps.CO2, ps.Ks)
    if ps.pCO2 is None:
        ps.pCO2 = fCO2_to_pCO2(ps.fCO2, ps.T)
    if ps.HCO3 is None:
        ps.HCO3 = cHCO3(ps.H, ps.DIC, ps.Ks)
    if ps.CO3 is None:
        ps.CO3 = cCO3(ps.H, ps.DIC, ps.Ks)
    # Always calculate elements of alkalinity
    try:
        # necessary for use with CBsyst in special cases
        # where BT is not known before Csys is run.
        ps.TA, ps.CAlk, ps.PAlk, ps.SiAlk, ps.OH = cTA(H=ps.H,
                                                       DIC=ps.DIC,
                                                       BT=ps.BT,
                                                       TP=ps.TP,
                                                       TSi=ps.TSi,
                                                       TS=ps.TS,
                                                       TF=ps.TF,
                                                       Ks=ps.Ks, mode='multi')
    except TypeError:
        pass
    if ps.pH is None:
        ps.pH = cp(ps.H)

    # clean up for output
    if 'pdict' in ps:
        del ps.pdict  # remove pdict, for clarity
    for k in ['BT', 'CO2', 'CO3', 'Ca', 'DIC', 'H',
              'HCO3', 'Mg', 'S', 'T', 'TA', 'pH',
              'CAlk', 'PAlk', 'SiAlk', 'OH']:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)
    if ps.unit != 1:
        for p in upar + ['CAlk', 'PAlk', 'SiAlk', 'OH']:
            ps[p] *= ps.unit  # convert back to input units

    return ps


# B Speciation
# ------------
def Bsys(pH=None, BT=None, BO3=None, BO4=None,
         ABT=None, ABO3=None, ABO4=None,
         dBT=None, dBO3=None, dBO4=None,
         alphaB=None,
         T=25., S=35., P=None,
         Ca=None, Mg=None,
         Ks=None, pdict=None):
    """
    Calculate the boron chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    pH is Total scale.

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations.

    Parameters
    ----------
    pH, BT, BO3, BO4 : array-like
        Boron system parameters. Two of these must be provided.
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # if neither Ca nor Mg provided, use default Ks
    ps.Ks = get_Ks(ps)

    # B system calculations
    if ps.pH is not None and ps.BT is not None:
        ps.H = ch(ps.pH)
    elif ps.BT is not None and ps.BO3 is not None:
        ps.H = BT_BO3(ps.BT, ps.BO3, ps.Ks)
    elif ps.BT is not None and ps.BO4 is not None:
        ps.H = BT_BO4(ps.BT, ps.BO4, ps.Ks)
    elif ps.BO3 is not None and ps.BO4 is not None:
        ps.BT = ps.BO3 + ps.BO3
        ps.H = BT_BO3(ps.BT, ps.BO3, ps.Ks)
    elif ps.pH is not None and ps.BO3 is not None:
        ps.H = ch(ps.pH)
        ps.BT = pH_BO3(ps.pH, ps.BO3, ps.Ks)
    elif ps.pH is not None and ps.BO4 is not None:
        ps.H = ch(ps.pH)
        ps.BT = pH_BO4(ps.pH, ps.BO4, ps.Ks)

    # The above makes sure that BT and H are known,
    # this next bit calculates all the missing species
    # from BT and H.
    if ps.BO3 is None:
        ps.BO3 = cBO3(ps.BT, ps.H, ps.Ks)
    if ps.BO4 is None:
        ps.BO4 = cBO4(ps.BT, ps.H, ps.Ks)
    if ps.pH is None:
        ps.pH = cp(ps.H)

    if NnotNone(ps.ABT, ps.ABO3, ps.ABO4, ps.dBT, ps.dBO3, ps.dBO4) != 0:
        ps.update(ABsys(pdict=ps))

    if 'pdict' in ps:
        del ps.pdict  # remove pdict, for clarity
    for k in ['BT', 'H', 'pH', 'BO3', 'BO4',
              'Ca', 'Mg', 'S', 'T', ]:
        # convert all outputs to (min) 1D numpy arrays.
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    return ps


# B Isotopes
# ----------
def ABsys(pH=None,
          ABT=None, ABO3=None, ABO4=None,
          dBT=None, dBO3=None, dBO4=None,
          alphaB=None,
          T=25., S=35., P=None,
          Ca=None, Mg=None,
          Ks=None, pdict=None):
    """
    Calculate the boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    pH is Total scale.

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations.

    Parameters
    ----------
    pH, ABT, ABO3, ABO4, dBT, dBO3, dBO4 : array-like
        Boron isotope system parameters. pH and one other
        parameter must be provided.
    alphaB : array-like
        Alpha value describing B fractionation (1.0XXX).
        If missing, it's calculated using the temperature
        sensitive formulation of Hönisch et al (2008)
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # if neither Ca nor Mg provided, use default Ks
    ps.Ks = get_Ks(ps)

    # if deltas provided, calculate corresponding As
    if ps.dBT is not None:
        ps.ABT = d11_2_A11(ps.dBT)
    if ps.dBO3 is not None:
        ps.ABO3 = d11_2_A11(ps.dBO3)
    if ps.dBO4 is not None:
        ps.ABO4 = d11_2_A11(ps.dBO4)

    # calculate alpha
    ps.alphaB = alphaB_calc(ps.T)

    if ps.pH is not None and ps.ABT is not None:
        ps.H = ch(ps.pH)
    elif ps.pH is not None and ps.ABO3 is not None:
        ps.ABT = pH_ABO3(ps.pH, ps.ABO3, ps.Ks, ps.alphaB)
    elif ps.pH is not None and ps.ABO4 is not None:
        ps.ABT = pH_ABO3(ps.pH, ps.ABO4, ps.Ks, ps.alphaB)
    else:
        raise ValueError('pH must be determined to calculate isotopes.')

    if ps.ABO3 is None:
        ps.ABO3 = cABO3(ps.H, ps.ABT, ps.Ks, ps.alphaB)
    if ps.ABO4 is None:
        ps.ABO4 = cABO4(ps.H, ps.ABT, ps.Ks, ps.alphaB)

    if ps.dBT is None:
        ps.dBT = A11_2_d11(ps.ABT)
    if ps.dBO3 is None:
        ps.dBO3 = A11_2_d11(ps.ABO3)
    if ps.dBO4 is None:
        ps.dBO4 = A11_2_d11(ps.ABO4)

    if 'pdict' in ps:
        del ps.pdict  # remove pdict, for clarity
    for k in ['ABO3', 'ABO4', 'ABT', 'Ca',
              'H', 'Mg', 'S', 'T', 'alphaB',
              'dBO3', 'dBO4', 'dBT', 'pH']:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    return ps


# Whole C-B-Isotope System
# ------------------------
def CBsys(pH=None, DIC=None, CO2=None, HCO3=None, CO3=None, TA=None, fCO2=None, pCO2=None,
          BT=None, BO3=None, BO4=None,
          ABT=None, ABO3=None, ABO4=None, dBT=None, dBO3=None, dBO4=None,
          alphaB=None,
          T=25., S=35., P=None,
          Ca=None, Mg=None, TP=0., TSi=0.,
          Ks=None, pdict=None, unit='umol'):
    """
    Calculate carbon, boron and boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    pH is Total scale.

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Note: Special Case! If pH is not known, you must provide either:
      - Two of [DIC, CO2, HCO3, CO3], and one of [BT, BO3, BO4]
      - One of [DIC, CO2, HCO3, CO3], and TA and BT
      - Two of [BT, BO3, BO4] and one of [DIC, CO2, HCO3, CO3]

    Isotopes will only be calculated if one of [ABT, ABO3, ABO4, dBT, dBO3, dBO4]
    is provided.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations, but:

    **WARNING** Error propagation NOT IMPLEMENTED for carbon system calculations
    with zero-finders (i.e. when pH is not given; cases 2-5 and 10-15).

    Parameters
    ----------
    pH, DIC, CO2, HCO3, CO3, TA : array-like
        Carbon system parameters. Two of these must be provided.
        If TA is specified, a B species must also be specified.
    pH, BT, BO3, BO4 : array-like
        Boron system parameters. Two of these must be provided.
    pH, ABT, ABO3, ABO4, dBT, dBO3, dBO4 : array-like
        Boron isotope system parameters. pH and one other
        parameter must be provided.
    alphaB : array-like
        Alpha value describing B fractionation (1.0XXX).
        If missing, it's calculated using the temperature
        sensitive formulation of Hönisch et al (2008)
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    unit : str
        Concentration units of C and B parameters (all must be in
        the same units).
        Can be 'mol', 'mmol', 'umol', 'nmol', 'pmol' or 'fmol'.
        Used in calculating Alkalinity. Default is 'umol'.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, * in mol / kg *.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # convert unit to multiplier
    udict = {'mol': 1.,
             'mmol': 1.e3,
             'umol': 1.e6,
             'µmol': 1.e6,
             'nmol': 1.e9,
             'pmol': 1.e12,
             'fmol': 1.e15}
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]
    elif isinstance(ps.unit, (int, float)):
        ps.unit = unit

    upar = ['DIC', 'CO2', 'HCO3', 'CO3', 'TA', 'fCO2', 'pCO2',
            'BT', 'BO3', 'BO4', 'TP', 'TSi']
    for p in upar:
        if ps[p] is not None:
            ps[p] = np.divide(ps[p], ps.unit)  # convert to molar

    # reassign unit, so conversions aren't repeated by Csys
    orig_unit = ps.unit
    ps.unit = 1.

    # determin max lengths
    kexcl = ['Ks', 'pdict', 'unit']
    ks = [k for k in ps.keys() if k not in kexcl]
    L = maxL(*[ps[k] for k in ks])
    # make inputs same length
    for k in ks:
        if ps[k] is not None:
            if isinstance(ps[k], (int, float)):
                ps[k] = np.full(L, ps[k])

    # Calculate Ks
    # if neither Ca nor Mg provided, use MyAMI Ks for modern SW
    ps.Ks = get_Ks(ps)

    # Conserved seawater chemistry
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S)

    # if fCO2 is given but CO2 is not, calculate CO2
    if ps.CO2 is None:
        if ps.fCO2 is not None:
            ps.CO2 = fCO2_to_CO2(ps.fCO2, ps.Ks)
        elif ps.pCO2 is not None:
            ps.CO2 = fCO2_to_CO2(pCO2_to_fCO2(ps.pCO2, ps.T), ps.Ks)

    # if no B info provided, assume modern conc.
    nBspec = NnotNone(ps.BT, ps.BO3, ps.BO4)
    if nBspec == 0:
        ps.BT = 433.e-6

    # This section works out the order that things should be calculated in.
    # Special case: if pH is missing, must have:
    #   a) two C
    #   b) two B
    #   c) one pH-dependent B, one pH-dependent C... But that's cray...
    #      (c not implemented!)
    
    if ps.pH is None:
        nCspec = NnotNone(ps.DIC, ps.CO2, ps.HCO3, ps.CO3)
        # a) if there are 2 C species, or one C species and TA and BT
        if ((nCspec == 2) | ((nCspec == 1) & (NnotNone(ps.TA, ps.BT) == 2))):
            ps.update(Csys(pdict=ps))  # calculate C first
            ps.update(Bsys(pdict=ps))  # then B
            # Note on the peculiar syntax here:
            #  ps is a dict of parameters, where
            #  everything that needs to be calculated
            #  is None.
            #  We give this to the [N]sys function
            #  as pdict, which passes the paramters from THIS
            #  function to [N]sys.
            #  As the output of Csys is also a dict (Bunch)
            #  with exactly the same form as ps, we can then
            #  use the .update attribute of ps to update
            #  all the paramters that were calculated by
            #  Csyst.
            #  Thus, all calculation is incremental, working
            #  with the same parameter set. As dicts are
            #  mutable, this has the added benefit of the
            #  parameters only being stored in memory once.
            if ps.TA is None:
                ps.TA, ps.CAlk, ps.PAlk, ps.SiAlk, ps.OH = cTA(H=ps.H,
                                                               DIC=ps.DIC,
                                                               BT=ps.BT,
                                                               TP=ps.TP,
                                                               TSi=ps.TSi,
                                                               TS=ps.TS,
                                                               TF=ps.TF,
                                                               Ks=ps.Ks,
                                                               mode='multi')
                # necessary becayse TA in Csys fails if there's no BT
        # b) if there are 2 B species
        elif nBspec == 2:
            ps.update(Bsys(pdict=ps))  # calculate B first
            ps.update(Csys(pdict=ps))  # then C
        else:  # if neither condition is met, throw an error
            raise ValueError(("Impossible! You haven't provided enough parameters.\n" +
                              "If you don't know pH, you must provide either:\n" +
                              "  - Two of [DIC, CO2, HCO3, CO3], and one of [BT, BO3, BO4]\n" +
                              "  - One of [DIC, CO2, HCO3, CO3], and TA and BT\n" +
                              "  - Two of [BT, BO3, BO4] and one of [DIC, CO2, HCO3, CO3]"))

    else:  # if we DO have pH, it's dead easy!
        ps.update(Bsys(pdict=ps))  # calculate B first
        ps.update(Csys(pdict=ps))  # then C

    for p in upar + ['CAlk', 'PAlk', 'SiAlk', 'OH']:
        ps[p] *= orig_unit  # convert back to input units

    return ps
