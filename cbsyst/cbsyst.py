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

# def Csys(
#         pHtot=None, DIC=None, TA=None,
#         CO2=None, HCO3=None, CO3=None,
#         pCO2=None, fCO2=None,
#         BT=None,
#         Ca=0.0102821, Mg=0.0528171,
#         T_in=25.0, T_out=None, 
#         S_in=35.0, S_out=None,
#         P_in=0.0, P_out=None,
#         PT=0.0, SiT=0.0,
#         ST=None, FT=None,
#         pHsws=None, pHfree=None, pHNBS=None,
#         unit="umol", Ks=None,
#         pdict=None,
#         OmegaC=None, OmegaA=None,
#         MyAMI_mode="calculate") -> CBsystData:
    
#     ps = ClassFactory(**locals())

#     io.validate_carbon_inputs(ps)

#     ps.Ks = constants.calc_Ks(ps)

#     units.convert_to_molar(ps)

#     carbon.use_Omega(ps)

#     constants.calc_conservative_elements(ps)

#     carbon.solve_C_system(ps)

#     carbon.calculate_Omegas(ps)

#     units.convert_from_molar(ps)

#     if utils.has_output_condition(ps):
#         out = Csys(
#             TA=ps.TA, 
#             DIC=ps.DIC, 
#             T_in=ps.T_out or ps.T_in,
#             P_in=ps.P_out or ps.P_in,
#             S_in=ps.S_out or ps.S_in,
#             unit=ps.unit,
#             Ca=ps.Ca,
#             Mg=ps.Mg,
#             )
#         out.input_conditions = ps

    # return ps

    


# class CarbonSystemCalculator:
#     """Main calculator for carbon system"""
    
#     def __init__(self):
#         pass
    
#     def calculate(self, **kwargs) -> CarbonSystemParams:
#         """Calculate carbon system from parameters"""
#         # Parse inputs
#         params = self._parse_inputs(kwargs)
        
#         # Validate inputs
#         self.validator.validate_inputs(params)
        
#         # Convert units
#         unit_multiplier = self.unit_converter.UNIT_MULTIPLIERS.get(params.unit, params.unit)
#         self.unit_converter.to_molar(params)
        
#         # Calculate defaults
#         self.seawater.calculate_defaults(params)
        
#         # Clean data
#         self.validator.clean_negative_values(params)
        
#         # Calculate constants
#         Ks = self._calculate_constants(params)
        
#         # Calculate pH scales
#         ph_scales = self._calculate_ph_scales(params, Ks)
        
#         # Handle omega inputs
#         self._handle_omega_inputs(params, Ks)
        
#         # Calculate carbon system
#         result = self._calculate_carbon_system(params, Ks)
#         result.update(ph_scales)
        
#         # Calculate derived properties
#         self._calculate_derived_properties(result, params, Ks)
        
#         # Format output
#         self._format_output(result, unit_multiplier)
        
#         # Handle output conditions if specified
#         if self._has_output_conditions(params):
#             result = self._calculate_output_conditions(result, params, unit_multiplier)
        
#         return result
    
#     def _parse_inputs(self, kwargs) -> CarbonSystemParams:
#         """Parse input arguments into structured parameters"""
#         # Handle pdict override
#         if 'pdict' in kwargs and isinstance(kwargs['pdict'], dict):
#             kwargs.update(kwargs['pdict'])
#             del kwargs['pdict']
        
#         # Create params object with validation
#         return CarbonSystemParams(**{k: v for k, v in kwargs.items() 
#                                    if k in CarbonSystemParams.__dataclass_fields__})
    
#     def _calculate_constants(self, params: CarbonSystemParams) -> Bunch:
#         """Calculate equilibrium constants"""
#         if isinstance(params.Ks, dict):
#             return Bunch(params.Ks)
#         else:
#             return Bunch(calc_Ks(
#                 temp_c=params.T_in, sal=params.S_in, p_bar=params.P_in,
#                 magnesium=params.Mg, calcium=params.Ca, 
#                 sulphate=params.ST, fluorine=params.FT, 
#                 MyAMI_mode=params.MyAMI_mode
#             ))
    
#     def _calculate_ph_scales(self, params: CarbonSystemParams, Ks: Bunch) -> Dict:
#         """Calculate pH on different scales"""
#         return calc_pH_scales(
#             pHtot=params.pHtot, pHfree=params.pHfree,
#             pHsws=params.pHsws, pHNBS=params.pHNBS,
#             ST=params.ST, FT=params.FT,
#             TempK=params.T_in + 273.15, Sal=params.S_in, Ks=Ks
#         )
    
#     def _handle_omega_inputs(self, params: CarbonSystemParams, Ks: Bunch) -> None:
#         """Convert omega values to CO3 if provided"""
#         if params.OmegaA is not None:
#             params.CO3 = params.OmegaA * Ks.KspA / (params.Ca * params.S_in / 35.)
#         elif params.OmegaC is not None:
#             params.CO3 = params.OmegaC * Ks.KspC / (params.Ca * params.S_in / 35.)
    
#     def _calculate_carbon_system(self, params: CarbonSystemParams, Ks: Bunch) -> Bunch:
#         """Calculate carbon system speciation"""
#         # Convert params to dict for calc_C_species
#         param_dict = {
#             'pHtot': params.pHtot, 'DIC': params.DIC, 'TA': params.TA,
#             'CO2': params.CO2, 'HCO3': params.HCO3, 'CO3': params.CO3,
#             'pCO2': params.pCO2, 'fCO2': params.fCO2,
#             'T_in': params.T_in, 'S_in': params.S_in,
#             'BT': params.BT, 'PT': params.PT, 'SiT': params.SiT,
#             'ST': params.ST, 'FT': params.FT, 'Ks': Ks
#         }
#         return calc_C_species(**param_dict)
    
#     def _calculate_derived_properties(self, result: Bunch, params: CarbonSystemParams, Ks: Bunch) -> None:
#         """Calculate derived properties like Omega and Revelle factor"""
#         # Revelle factor (only if no uncertainties)
#         has_uncertainties = any(hasattr(result.get(param), 'nominal_value') 
#                               for param in ['TA', 'DIC', 'CO2', 'HCO3', 'CO3'] 
#                               if result.get(param) is not None)
        
#         if not has_uncertainties:
#             result["revelle_factor"] = calc_revelle_factor(
#                 TA=result.TA, DIC=result.DIC, BT=params.BT,
#                 PT=params.PT, SiT=params.SiT, ST=params.ST, FT=params.FT, Ks=Ks
#             )
#         else:
#             result["revelle_factor"] = None
        
#         # Omega calculations
#         oCa = params.Ca * params.S_in / 35.
#         result['OmegaA'] = result['CO3'] * oCa / Ks.KspA
#         result['OmegaC'] = result['CO3'] * oCa / Ks.KspC
    
#     def _format_output(self, result: Bunch, unit_multiplier: float) -> None:
#         """Format output arrays and convert units"""
#         # Convert to arrays
#         outputs = [
#             "BT", "CO2", "CO3", "Ca", "DIC", "H", "HCO3", 
#             "Mg", "S_in", "T_in", "TA", "CAlk", "PAlk", 
#             "SiAlk", "OH", 'OmegaA', 'OmegaC', 'revelle_factor'
#         ]
#         for k in outputs:
#             if k in result and not isinstance(result[k], np.ndarray):
#                 result[k] = np.array(result[k], ndmin=1)
        
#         # Convert units back
#         self.unit_converter.from_molar(result, unit_multiplier)
    
#     def _has_output_conditions(self, params: CarbonSystemParams) -> bool:
#         """Check if output conditions are specified"""
#         return any(getattr(params, attr) is not None 
#                   for attr in ['T_out', 'S_out', 'P_out'])
    
#     def _calculate_output_conditions(self, result: Bunch, params: CarbonSystemParams, unit_multiplier: float) -> Bunch:
#         """Calculate results at output conditions"""
#         # Set defaults
#         T_out = params.T_out or params.T_in
#         S_out = params.S_out or params.S_in
#         P_out = params.P_out or params.P_in
        
#         # Adjust salinity-dependent parameters if needed
#         if params.S_out is not None and params.S_out != params.S_in:
#             BT = params.BT * S_out / params.S_in
#             ST = params.ST * S_out / params.S_in
#             FT = params.FT * S_out / params.S_in
#         else:
#             BT, ST, FT = params.BT, params.ST, params.FT
        
#         # Calculate at output conditions
#         out_cond = self.calculate(
#             TA=result.TA, DIC=result.DIC,
#             T_in=T_out, S_in=S_out, P_in=P_out,
#             unit=unit_multiplier, Ca=params.Ca, Mg=params.Mg,
#             BT=BT, FT=FT, ST=ST
#         )
        
#         # Rename existing results with "_in" suffix
#         output_params = [
#             "BAlk", "BT", "CAlk", "CO2", "CO3", "DIC", "H", "HCO3", 
#             "HF", "HSO4", "Hfree", "Ks", "OH", "PAlk", "SiAlk", "TA", "FT",
#             "PT", "ST", "SiT", "fCO2", "pCO2", "pHfree", "pHsws", 
#             "pHtot", "pHNBS", 'OmegaA', 'OmegaC', "revelle_factor"
#         ]
        
#         for param in output_params:
#             if param in result:
#                 result[param + "_in"] = result[param]
#                 result[param] = out_cond[param]
        
#         return result
