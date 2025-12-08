"""
Functions for calculating the carbon and boron chemistry of seawater.
"""
import inspect
import numpy as np
from cbsyst.uncertainties import remove_negatives

from .dataclasses import CBsystData, KValues, create_dataclass
from . import utils, units, constants, carbon, boron, boron_isotopes, pH

from typing import Optional, Union, overload

DEBUG = False

def Csys(
    # Carbon System Parameters
    pHtot: Optional[Union[float, np.ndarray]] = None,
    pHsws: Optional[Union[float, np.ndarray]] = None,
    pHfree: Optional[Union[float, np.ndarray]] = None,
    pHNBS: Optional[Union[float, np.ndarray]] = None,
    DIC: Optional[Union[float, np.ndarray]] = None,
    TA: Optional[Union[float, np.ndarray]] = None,
    CO2: Optional[Union[float, np.ndarray]] = None,
    HCO3: Optional[Union[float, np.ndarray]] = None,
    CO3: Optional[Union[float, np.ndarray]] = None,
    pCO2: Optional[Union[float, np.ndarray]] = None,
    fCO2: Optional[Union[float, np.ndarray]] = None,
    OmegaC: Optional[Union[float, np.ndarray]] = None,
    OmegaA: Optional[Union[float, np.ndarray]] = None,
    # Boron System Parameters
    BT: Optional[Union[float, np.ndarray]] = None,
    BO3: Optional[Union[float, np.ndarray]] = None,
    BO4: Optional[Union[float, np.ndarray]] = None,
    # Boron Isotope Parameters
    dBT: Optional[Union[float, np.ndarray]] = 39.61,
    dBO3: Optional[Union[float, np.ndarray]] = None,
    dBO4: Optional[Union[float, np.ndarray]] = None,
    ABT: Optional[Union[float, np.ndarray]] = None,
    ABO3: Optional[Union[float, np.ndarray]] = None,
    ABO4: Optional[Union[float, np.ndarray]] = None,
    alphaB:  Optional[Union[float, np.ndarray]] = 1.0272,
    # Conservative Ions
    Ca: Union[float, np.ndarray] = 0.0102821,
    Mg: Union[float, np.ndarray] = 0.0528171,
    PT: Union[float, np.ndarray] = 0.0,
    SiT: Union[float, np.ndarray] = 0.0,
    ST: Optional[Union[float, np.ndarray]] = None,
    FT: Optional[Union[float, np.ndarray]] = None,
    # Conditions
    T_in: Union[float, np.ndarray] = 25.0,
    T_out: Optional[Union[float, np.ndarray]] = None,
    S_in: Union[float, np.ndarray] = 35.0,
    S_out: Optional[Union[float, np.ndarray]] = None,
    P_in: Union[float, np.ndarray] = None,
    P_out: Optional[Union[float, np.ndarray]] = None,
    # Constants
    Ks: Optional[Union[dict, KValues]] = None,
    # Configuration
    unit: str = "umol",
    MyAMI_mode: str = "calculate",
    ) -> CBsystData:
    """
    Calculate the complete marine carbonate-boron system from any two known parameters.
    
    This is the main function for calculating seawater carbonate chemistry, boron speciation,
    and boron isotope fractionation. 
    
    The function requires some combination of:
        - Carbonate system: pH + any C species, or any two C species
        - Boron system: pH + any B species, or any two B species  
        - Isotope system: pH + any isotope parameter, or any two isotope parameters.
        
        When output conditions (T_out, S_out, P_out) are specified, the system
        is recalculated at those conditions with appropriate corrections for
        equilibrium constants and conservative ion concentrations.
        
        Gas parameters (pCO2, fCO2) are always in ppm regardless of unit setting.
                 
        The function automatically determines which calculation pathway to use based on the
        provided parameters and can handle different temperature, salinity, and pressure
        conditions for input and output.
    
    **Speciation constants** (Ks) can be provided directly as a dictionary, or will be calculated
    for the specified conditions (T, S, P) using  the [KGen](https://palaeocarbonatechemistry.github.io/Kgen/) module. This uses the 'Best Practices'
    Ks from Dickson, Sabine and Christian (2007), unless Ca or Mg differs from the default values.
    
    If Ca or Mg deviate from the default values, KGen will adjust the constants for modified
    seawater chemistry using the MyAMI pitzer model. If MyAMI is being used, it can add
    substantial overhead and calculation time. If you require fast calculations, set `MyAMI_mode`
    to `"approximate"` to use a polynomial approximation of the MyAMI model. See the KGen
    documentation for more details.
    
    **Uncertainties** will be propagated through calculations analytically using the [`uncertainties`](https://pythonhosted.org/uncertainties/)
    module, if you provide inputs with uncertainties (e.g. `uncertainties.ufloat` or 
    `uncertainties.unumpy.uarray` objects).
    
    Args:
        pHtot: pH on the total scale.
        pHsws: pH on the seawater scale.
        pHfree: pH on the free scale.
        pHNBS: pH on the NBS scale.
        DIC: Dissolved inorganic carbon concentration.
        TA: Total alkalinity.
        CO2: Dissolved CO2 concentration.
        HCO3: Bicarbonate ion concentration.
        CO3: Carbonate ion concentration.
        pCO2: Partial pressure of CO2 (ppm).
        fCO2: Fugacity of CO2 (ppm).
        OmegaC: Calcite saturation state.
        OmegaA: Aragonite saturation state.
        BT: Total boron concentration.
        BO3: Boric acid concentration.
        BO4: Borate ion concentration.
        dBT: Boron isotope composition of total boron (‰, default 39.61).
        dBO3: Boron isotope composition of boric acid (‰).
        dBO4: Boron isotope composition of borate (‰).
        ABT: Absolute boron isotope ratio of total boron.
        ABO3: Absolute boron isotope ratio of boric acid.
        ABO4: Absolute boron isotope ratio of borate.
        alphaB: Boron isotope fractionation factor (default 1.0272).
        Ca: Calcium concentration (mol/kg, default 0.0102821 for S=35).
        Mg: Magnesium concentration (mol/kg, default 0.0528171 for S=35).
        PT: Total phosphate concentration (mol/kg, default 0.0).
        SiT: Total silicate concentration (mol/kg, default 0.0).
        ST: Total sulfate concentration (mol/kg, auto-calculated from salinity if None).
        FT: Total fluoride concentration (mol/kg, auto-calculated from salinity if None).
        T_in: Input temperature (°C, default 25.0).
        T_out: Output temperature (°C, if different from input).
        S_in: Input salinity (PSU, default 35.0).
        S_out: Output salinity (PSU, if different from input).
        P_in: Input pressure (bar, default 0.0).
        P_out: Output pressure (bar, if different from input).
        Ks: Pre-calculated equilibrium constants (auto-calculated if None).
        unit: Concentration unit for input/output ("umol", "mmol", or "mol", default "umol").
        MyAMI_mode: MyAMI calculation mode, can be "calculate" or "approximate" (default "calculate").
    
    Returns:
        CBsystData: Complete carbonate-boron system results containing all calculated
            parameters, equilibrium constants, and metadata. Results include all
            carbonate species, boron speciation, pH on all scales, saturation states,
            and boron isotope compositions.
    
    Raises:
        ValueError: If insufficient parameters are provided to solve the system
            (need at least 2 from carbonate, boron, or isotope systems).
                
    Examples:
        Basic carbonate system calculation:
        
        >>> result = Csys(pHtot=8.1, DIC=2000, T_in=25, S_in=35)
        >>> print(f"TA = {result.TA:.1f} µmol/kg")
        TA = 2300.5 µmol/kg
        
        With output conditions:
        
        >>> result = Csys(pHtot=8.1, DIC=2000, T_in=25, T_out=20, S_in=35, S_out=30)
        >>> print(f"pH at input: {result.pHtot_in:.2f}, at output: {result.pHtot:.2f}")
        pH at input: 8.10, at output: 8.05
        
        Boron isotope calculation:
        
        >>> result = Csys(dBT=39.61, dBO3=49.05, T_in=25, S_in=35)
        >>> print(f"Calculated pH = {result.pHtot:.2f}")
        Calculated pH = 8.15
        
        Array inputs:
        
        >>> pH_array = np.array([7.8, 8.0, 8.2])
        >>> DIC_array = np.array([1950, 2000, 2050])
        >>> result = Csys(pHtot=pH_array, DIC=DIC_array, T_in=25, S_in=35)
        >>> print(result.TA.shape)
        (3,)
    """
    
    # get parameter defaults
    sig = inspect.signature(Csys)
    defaults = {
        param.name: param.default 
        for param in sig.parameters.values() 
        if param.default is not None
    }

    if dBO3 is not None and dBO4 is not None:
        dBT = None
    
    if ABO3 is not None and ABO4 is not None:
        ABT = None

    # create data object
    csys = create_dataclass(**locals())

    # calculation logic:
    n_pH_given = pH.n_given(csys)
    n_C_given = carbon.n_given(csys)
    n_B_given = boron.n_given(csys)
    n_iso_given = boron_isotopes.n_given(csys)

    if DEBUG: print(n_pH_given, n_C_given, n_B_given, n_iso_given)

    csys.Ks = constants.calc_Ks(csys)  # create_dataclasscalculate constants
    units.convert_to_molar(csys)  # convert all concentration units to molar
    constants.calc_conservative_composition(csys)  # calculate seawater composition from salinity

    # 1. pH not given
    if n_pH_given == 0:
        # a. two carbon species given --> use to get pH, then calculate boron and isotopes
        if n_C_given == 2:
            if DEBUG: print('calculating pH from carbon species')
            carbon.solve_C_system(csys)  # calculate carbon system
        # b. two boron species given --> use to get pH, then calculate carbon and isotopes
        elif n_B_given == 2:
            if DEBUG: print('calculating pH from boron species')
            boron.solve_B_system(csys)  # calculate boron system
        # c. two isotope params given --> use to get pH, then calculate carbon and boron speciation
        elif n_iso_given >= 2:
            if DEBUG: print('calculating pH from boron isotopes')
            boron_isotopes.solve_B_isotopes(csys)  # calculate boron isotopes
    
    # at this stage, pH is known
    pH.convert_scales(csys)
    
    # if the C system was partially provided, calculate it
    if n_C_given == 1:
        if DEBUG: print('calculating carbon')
        carbon.solve_C_system(csys)
    # if the B system was partially provided, calculate it
    if n_B_given == 1:
        if DEBUG: print('calculating boron')
        boron.solve_B_system(csys)
    # if the isotope system was partially provided, calculate it
    if n_iso_given == 1:
        if DEBUG: print('calculating boron isotopes')
        boron_isotopes.solve_B_isotopes(csys)

    # if has an output condition, recalculate at that condition.
    if utils.has_output_condition(csys):
        if DEBUG: print('recalculating at output conditions')
        # Store input conditions
        inputs = {k: csys.get(k) for k in csys.inputs}
        
        # Set output condition defaults
        T_out = csys.T_out if csys.T_out is not None else csys.T_in
        S_out = csys.S_out if csys.S_out is not None else csys.S_in
        P_out = csys.P_out if csys.P_out is not None else csys.P_in
        
        # Update salinity-dependent parameters if salinity changes
        if csys.S_out is not None:
            BT_out = csys.BT * S_out / csys.S_in
            ST_out = csys.ST * S_out / csys.S_in  
            FT_out = csys.FT * S_out / csys.S_in
        else:
            BT_out = csys.BT
            ST_out = csys.ST
            FT_out = csys.FT
        
        # Calculate at output conditions - use same parameter set as old implementation
        csys_out = Csys(
            TA=csys.TA, 
            DIC=csys.DIC, 
            dBT=getattr(csys, 'dBT', None),
            T_in=T_out,
            S_in=S_out,
            P_in=P_out,
            unit=None,
            Ca=csys.Ca,
            Mg=csys.Mg,
            BT=BT_out,
            ST=ST_out,
            FT=FT_out,
        )
        
        # rename conditions so that in/out are correctly preserved
        for k in ['T', 'S', 'P']:
            setattr(csys_out, k + '_out', getattr(csys_out, k + '_in', None))
            setattr(csys_out, k + '_in', getattr(csys, k + '_in', None))
        
        # store the input conditions, labelled as such.
        for k, v in inputs.items():
            if k not in ['T_out', 'T_in', 'S_out', 'S_in', 'P_out', 'P_in']:
                setattr(csys_out, k + "_in", v)
                
        # update inputs to include the _out parameters
        csys_out.inputs = csys.inputs

        # set units back to original
        setattr(csys_out, 'unit', csys.unit)
        units.convert_from_molar(csys_out)  # convert back to original concentration unit
        
        return csys_out

    else:
        units.convert_from_molar(csys)  # convert back to original concentration unit

        return csys

# for backward compatibility, alias the old Csys function
CBsys = Csys
Bsys = Csys
ABsys = Csys
