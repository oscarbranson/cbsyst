"""
Functions for calculating the carbon and boron chemistry of seawater.
"""
import inspect
import numpy as np
from cbsyst.uncertainties import remove_negatives

from .dataclasses import CBsystData, create_dataclass
from . import io, utils, units, constants, carbon, boron, boron_isotopes, pH

from typing import Optional, Union, overload
from typing_extensions import Unpack, TypedDict    

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
    P_in: Union[float, np.ndarray] = 0.0,
    P_out: Optional[Union[float, np.ndarray]] = None,
    # Constants
    Ks: Optional[object] = None,
    # Configuration
    unit: str = "umol",
    MyAMI_mode: str = "calculate",
    ) ->CBsystData:
    
    # get parameter defaults
    sig = inspect.signature(Csys)
    defaults = {
        param.name: param.default 
        for param in sig.parameters.values() 
        if param.default is not None
    }

    # create data object
    csys = create_dataclass(**locals())

    # identify inputs
    # csys.inputs = {
    #     k: csys.get(k)
    #     for k in csys.__dataclass_fields__ 
    #     if (csys.get(k) is not None) and 
    #     (k not in ['inputs']) and
    #     (defaults.get(k) != csys.get(k))
    # }

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
            carbon.solve_C_system(csys)  # calculate carbon system
        # b. two boron species given --> use to get pH, then calculate carbon and isotopes
        elif n_B_given == 2:
            boron.solve_B_system(csys)  # calculate boron system
        # c. two isotope params given --> use to get pH, then calculate carbon and boron speciation
        elif n_iso_given >= 2:
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
            unit=1,
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
