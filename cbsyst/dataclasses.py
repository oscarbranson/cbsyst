from dataclasses import dataclass
from typing import Optional, Union, Any, Iterator, Tuple, Callable, List
import numpy as np

class CBsystData:
    """
    Base class for CBsyst data structures providing dict-like interface.
    
    This class provides a dict-like interface to dataclass instances, allowing
    for convenient access to parameters using both attribute and dictionary
    syntax. It serves as the foundation for all CBsyst parameter classes.
    
    Note:
        All CBsyst parameter classes inherit from this base class to ensure
        consistent access patterns throughout the library.
    """
    
    def __getitem__(self, key: str) -> Any:
        """
        Enable dict-style getting: cp['pHtot'].
        
        Args:
            key: The parameter name to retrieve.
            
        Returns:
            The value of the requested parameter.
            
        Raises:
            KeyError: If the parameter name doesn't exist in this dataclass.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Enable dict-style setting: cp['pHtot'] = 8.1.
        
        Args:
            key: The parameter name to set.
            value: The value to assign to the parameter.
            
        Raises:
            KeyError: If the parameter name doesn't exist in this dataclass.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __contains__(self, key: str) -> bool:
        """
        Enable 'in' operator: 'pHtot' in cp.
        
        Args:
            key: The parameter name to check for existence.
            
        Returns:
            True if the parameter exists in this dataclass, False otherwise.
        """
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-style get with default: cp.get('pHtot', None).
        
        Args:
            key: The parameter name to retrieve.
            default: Default value to return if key doesn't exist.
            
        Returns:
            The parameter value if it exists, otherwise the default value.
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self) -> Iterator[str]:
        """
        Return field names like dict.keys().
        
        Returns:
            Iterator over all field names in this dataclass.
        """
        return self.__dataclass_fields__.keys()
    
    def values(self) -> Iterator[Any]:
        """
        Return field values like dict.values().
        
        Returns:
            Iterator over all field values in this dataclass.
        """
        return (getattr(self, field) for field in self.__dataclass_fields__)
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Return (key, value) pairs like dict.items().
        
        Returns:
            Iterator over (field_name, field_value) tuples for this dataclass.
        """
        return ((field, getattr(self, field)) for field in self.__dataclass_fields__)

class mixin_CBsyst_print:
    """
    Mixin class providing formatted string representation for CBsyst calculations.
    
    This mixin provides a structured output format for CBsyst calculation results,
    displaying inputs and calculated parameters in an organized table format.
    
    Note:
        This class is designed to be mixed into parameter dataclasses to provide
        consistent output formatting across all CBsyst calculation types.
    """
    
    # TODO: needs to work with arrays and uncertainties
    def __repr__(self) -> str:
        """
        Generate formatted string representation of calculation results.
        
        Creates a structured display showing input parameters and all calculated
        values in a tabular format with clear section divisions.
        
        Returns:
            Formatted string showing inputs and calculated parameters.
            
        Note:
            Currently optimized for scalar values. Array and uncertainty support
            is planned for future versions.
        """
        out = 'cbsyst Calculation\n'
        line_length = max(len(out) - 1, 30)
        
        line = '-' * line_length + '\n'
        section = '=' * line_length + '\n'
        
        col1 = 10
        col2 = line_length - col1

        out += section
        out += 'Inputs:\n'
        out += line
        for p in self.inputs:
            out += f'{p:<{col1}}\n'
        out += line
        out += 'Calculated:\n'
        out += line
        for p in ['pHtot', 'pHfree', 'pHsws', 'pHNBS', 'DIC', 'TA', 'CO2', 'HCO3', 'CO3', 'pCO2', 'fCO2', 'BT', 'BO3', 'BO4', 'dBT', 'dBO3', 'dBO4', 'OmegaC', 'OmegaA']:
            if self.get(p) is not None:
                out += f'{p:<{col1}}{self._fmt_value(self[p]):>{col2}}\n'
        out += section

        return out
    
    def _fmt_value(self, value: Any) -> str:
        """
        Format value for printing, handling arrays and uncertainties.
        
        Args:
            value: The value to format (scalar, array, or uncertainty).
            
        Returns:
            Formatted string representation of the value.
            
        Note:
            Handles numpy arrays by showing first and last values with count,
            uncertainties by showing nominal ± std_dev, and scalars with
            standard floating point formatting.
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return f'{value[0]:.2f} ... {value[-1]:.2f} (n={len(value)})' if len(value) > 1 else f'{value[0]:.2f}'
            elif value.ndim > 1:
                return f'{value.min():.2f} - {value.max():.2f} (shape={value.shape})' if value.shape[0] > 1 else f'{value[0, 0]:.2f}'
        elif hasattr(value, 'nominal_value'):
            return f'{value.nominal_value:.2f} ± {value.std_dev:.2f}'
        else:
            return f'{value:.2f}'


@dataclass(repr=False)
class KValues(CBsystData):
    """
    Dataclass for stoichiometric equilibrium constants.
    
    Contains all equilibrium constants needed for carbonate system calculations,
    including temperature and pressure corrections. All constants are on the
    total pH scale and appropriate ionic strength.
    
    Attributes:
        K0: CO2 solubility constant (mol/kg/atm).
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        KW: Water dissociation constant.
        KB: Boric acid dissociation constant.
        KS: Bisulfate dissociation constant.
        KspA: Aragonite solubility product.
        KspC: Calcite solubility product.
        KP1: First phosphoric acid dissociation constant.
        KP2: Second phosphoric acid dissociation constant.
        KP3: Third phosphoric acid dissociation constant.
        KSi: Silicic acid dissociation constant.
        KF: Hydrogen fluoride dissociation constant.
        
    Note:
        These constants are typically calculated from empirical relationships
        as functions of temperature, salinity, and pressure using functions
        in the constants module.
    """
    K0: Union[float, np.ndarray]
    K1: Union[float, np.ndarray]
    K2: Union[float, np.ndarray]
    KW: Union[float, np.ndarray]
    KB: Union[float, np.ndarray]
    KS: Union[float, np.ndarray]
    KspA: Union[float, np.ndarray]
    KspC: Union[float, np.ndarray]
    KP1: Union[float, np.ndarray]
    KP2: Union[float, np.ndarray]
    KP3: Union[float, np.ndarray]
    KSi: Union[float, np.ndarray]
    KF: Union[float, np.ndarray]

    def __repr__(self) -> str:
        """
        Format equilibrium constants for display.
        
        Returns:
            Formatted string showing all equilibrium constants in scientific notation.
        """
        out = 'K values:\n----------------\n'
        for k in self.__dataclass_fields__:
            out += f'{k:<5}{self[k]:.5e}\n'
        out += '----------------'
        return out

@dataclass(repr=False)
class mixin_ConservativeIons:
    """
    Mixin for conservative seawater ion concentrations.
    
    Contains the concentrations of major seawater ions that are conservative
    (their ratios remain constant with salinity). These are used in equilibrium
    constant calculations and alkalinity computations.
    
    Attributes:
        Ca: Calcium concentration (mol/kg), default for S=35.
        Mg: Magnesium concentration (mol/kg), default for S=35.
        PT: Total phosphate concentration (mol/kg).
        SiT: Total silicate concentration (mol/kg).
        ST: Total sulfate concentration (mol/kg), calculated from salinity if None.
        FT: Total fluoride concentration (mol/kg), calculated from salinity if None.
        BT: Total boron concentration (mol/kg), calculated from salinity if None.
        
    Note:
        Default values are for standard seawater composition at salinity 35.
        ST, FT, and BT are typically calculated from salinity relationships
        if not explicitly provided.
    """
    # Seawater Chemistry
    Ca: Union[float, np.ndarray] = 0.0102821
    Mg: Union[float, np.ndarray] = 0.0528171
    PT: Union[float, np.ndarray] = 0.0
    SiT: Union[float, np.ndarray] = 0.0
    ST: Optional[Union[float, np.ndarray]] = None
    FT: Optional[Union[float, np.ndarray]] = None
    BT: Optional[Union[float, np.ndarray]] = None

@dataclass(repr=False)
class mixin_Conditions:
    """
    Mixin for environmental conditions and equilibrium constants.
    
    Contains the temperature, salinity, and pressure conditions for both
    input measurements and output conditions, along with equilibrium constants.
    
    Attributes:
        T_in: Input temperature (°C).
        S_in: Input salinity (practical salinity scale).
        P_in: Input pressure (dbar).
        T_out: Output temperature (°C), defaults to T_in if None.
        S_out: Output salinity, defaults to S_in if None.
        P_out: Output pressure (dbar), defaults to P_in if None.
        Ks: Equilibrium constants dataclass or dictionary.
        
    Note:
        Input and output conditions allow for calculations where measurements
        were made under different conditions than the desired output conditions.
        This is common in oceanographic applications where samples are measured
        at lab conditions but results are needed for in-situ conditions.
    """
    # Environmental conditions
    T_in: Union[float, np.ndarray] = 25.0
    S_in: Union[float, np.ndarray] = 35.0
    P_in: Union[float, np.ndarray] = 0.0
    T_out: Optional[Union[float, np.ndarray]] = None
    S_out: Optional[Union[float, np.ndarray]] = None
    P_out: Optional[Union[float, np.ndarray]] = None

    # Constants
    Ks: Optional[KValues] = None

@dataclass(repr=False)
class mixin_pHConversion:
    """
    Mixin for pH scale conversions and hydrogen ion activities.
    
    Contains pH values on different scales and the conversion factors between
    them, along with hydrogen ion concentrations and activities.
    
    Attributes:
        FREEtoTOT: Conversion factor from free to total pH scale.
        SWStoTOT: Conversion factor from seawater to total pH scale.
        pHtot: pH on total scale.
        pHsws: pH on seawater scale.
        pHfree: pH on free scale.
        pHNBS: pH on NBS scale.
        H: Hydrogen ion concentration on total scale (mol/kg).
        fH: Free hydrogen ion concentration (mol/kg).
        
    Note:
        pH scale conversions account for the different ways of treating
        hydrogen ion association with sulfate and fluoride ions. The total
        scale includes both HSO4- and HF as H+ sources, the seawater scale
        includes only HF, and the free scale includes neither.
    """
    # pH conversion parts
    FREEtoTOT: Optional[Union[float, np.ndarray]] = None
    SWStoTOT: Optional[Union[float, np.ndarray]] = None

    # pH scales
    pHtot: Optional[Union[float, np.ndarray]] = None
    pHsws: Optional[Union[float, np.ndarray]] = None
    pHfree: Optional[Union[float, np.ndarray]] = None
    pHNBS: Optional[Union[float, np.ndarray]] = None

    # H and free H
    H: Optional[Union[float, np.ndarray]] = None
    fH: Optional[Union[float, np.ndarray]] = None

@dataclass(repr=False)
class mixin_CarbonSystem:
    """
    Mixin for carbonate system parameters.
    
    Contains all measurable and calculable parameters of the marine carbonate
    system, including carbon speciation, alkalinity components, and saturation
    states.
    
    Attributes:
        DIC: Dissolved inorganic carbon (mol/kg).
        TA: Total alkalinity (mol/kg).
        CO2: Dissolved CO2 concentration (mol/kg).
        HCO3: Bicarbonate ion concentration (mol/kg).
        CO3: Carbonate ion concentration (mol/kg).
        pCO2: Partial pressure of CO2 (μatm).
        fCO2: Fugacity of CO2 (μatm).
        OmegaC: Calcite saturation state.
        OmegaA: Aragonite saturation state.
        CAlk: Carbonate alkalinity (mol/kg).
        BAlk: Borate alkalinity (mol/kg).
        PAlk: Phosphate alkalinity (mol/kg).
        SiAlk: Silicate alkalinity (mol/kg).
        OH: Hydroxide ion concentration (mol/kg).
        Hfree: Free hydrogen ion concentration (mol/kg).
        HSO4: Bisulfate ion concentration (mol/kg).
        HF: Hydrogen fluoride concentration (mol/kg).
        
    Note:
        The carbonate system is fully determined by any two of the main
        parameters (DIC, TA, pH, pCO2, fCO2, CO3, HCO3) along with
        temperature, salinity, and pressure.
    """
    # Carbon parameters
    DIC: Optional[Union[float, np.ndarray]] = None
    TA: Optional[Union[float, np.ndarray]] = None
    CO2: Optional[Union[float, np.ndarray]] = None
    HCO3: Optional[Union[float, np.ndarray]] = None
    CO3: Optional[Union[float, np.ndarray]] = None
    pCO2: Optional[Union[float, np.ndarray]] = None
    fCO2: Optional[Union[float, np.ndarray]] = None

    # Omega parameters
    OmegaC: Optional[Union[float, np.ndarray]] = None
    OmegaA: Optional[Union[float, np.ndarray]] = None

    # Alkalinity Components
    CAlk: Optional[Union[float, np.ndarray]] = None
    BAlk: Optional[Union[float, np.ndarray]] = None
    PAlk: Optional[Union[float, np.ndarray]] = None
    SiAlk: Optional[Union[float, np.ndarray]] = None
    OH: Optional[Union[float, np.ndarray]] = None
    Hfree: Optional[Union[float, np.ndarray]] = None
    HSO4: Optional[Union[float, np.ndarray]] = None
    HF: Optional[Union[float, np.ndarray]] = None

@dataclass(repr=False)
class mixin_BoronSystem:
    """
    Mixin for boron system parameters.
    
    Contains boron speciation parameters for the B(OH)3/B(OH)4- equilibrium
    system in seawater.
    
    Attributes:
        BO3: Boric acid concentration (mol/kg).
        BO4: Borate ion concentration (mol/kg).
        BT: Total boron concentration (mol/kg).
        
    Note:
        The boron system is governed by a single equilibrium between boric acid
        and borate ion, making it a useful pH proxy in paleoceanography.
    """
    BO3: Optional[Union[float, np.ndarray]] = None
    BO4: Optional[Union[float, np.ndarray]] = None
    BT: Optional[Union[float, np.ndarray]] = None

@dataclass(repr=False)
class mixin_BoronIsotopes:
    """
    Mixin for boron isotope system parameters.
    
    Contains boron isotope fractionation factors and isotope abundances in
    both fractional and delta notation for total boron and individual species.
    
    Attributes:
        alphaB: Fractionation factor between B(OH)3 and B(OH)4-.
        epsilonB: Enrichment factor (alphaB - 1) * 1000.
        ABT: Fractional abundance of 11B in total dissolved boron.
        ABO3: Fractional abundance of 11B in boric acid (B(OH)3).
        ABO4: Fractional abundance of 11B in borate ion (B(OH)4-).
        dBT: δ11B of total dissolved boron (‰).
        dBO3: δ11B of boric acid (‰).
        dBO4: δ11B of borate ion (‰).
        
    Note:
        Boron isotopes fractionate between B(OH)3 and B(OH)4- as a function
        of pH, making them useful for paleoenvironmental pH reconstruction.
        Delta values are relative to NIST SRM 951 boric acid standard.
    """
    alphaB: Optional[Union[float, np.ndarray]] = None
    epsilonB: Optional[Union[float, np.ndarray]] = None

    ABT: Optional[Union[float, np.ndarray]] = None
    ABO3: Optional[Union[float, np.ndarray]] = None
    ABO4: Optional[Union[float, np.ndarray]] = None

    dBT: Optional[Union[float, np.ndarray]] = None
    dBO3: Optional[Union[float, np.ndarray]] = None
    dBO4: Optional[Union[float, np.ndarray]] = None

@dataclass(repr=False)
class mixin_CBsyst_config:
    """
    Mixin for CBsyst calculation configuration parameters.
    
    Contains metadata and configuration options for CBsyst calculations,
    including input tracking and unit specifications.
    
    Attributes:
        inputs: Tuple of parameter names that were provided as inputs.
        unit: Unit system for concentration parameters ("umol" or "mol").
        MyAMI_mode: Mode for MyAMI alkalinity calculations ("calculate" or other).
        
    Note:
        The inputs tuple is automatically populated during calculation to track
        which parameters were provided vs. calculated. This is useful for
        understanding calculation pathways and debugging.
    """
    # Configuration
    inputs: tuple = ()
    unit: str = "umol"
    MyAMI_mode: str = "calculate"

# Final aggregate dataclasses
@dataclass(repr=False)
class CarbonSystemParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_CarbonSystem,
    mixin_CBsyst_print,
    ):
    """
    Data class for carbon system parameters only.
    
    Complete dataclass for carbonate system calculations including all
    necessary environmental conditions, ion concentrations, pH scales,
    and carbon system parameters.
    
    Note:
        This class combines all mixins needed for standalone carbonate
        system calculations without boron or isotope considerations.
    """
    pass


@dataclass(repr=False)
class BoronSystemParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_BoronSystem,
    mixin_CBsyst_print,
):
    """
    Data class for boron system parameters only.
    
    Complete dataclass for boron speciation calculations including all
    necessary environmental conditions and boron system parameters.
    
    Note:
        This class provides boron speciation calculations without isotope
        considerations or full carbonate system calculations.
    """
    pass

@dataclass(repr=False)
class BoronIsotopeParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_BoronIsotopes,
    mixin_CBsyst_print,
):
    """
    Data class for boron isotope parameters only.
    
    Complete dataclass for boron isotope calculations including fractionation
    factors and isotope abundances in delta and fractional notation.
    
    Note:
        This class focuses on isotope calculations and may be used in
        conjunction with separate boron speciation calculations.
    """
    pass

@dataclass(repr=False)
class BoronParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_BoronSystem,
    mixin_BoronIsotopes,
    mixin_CBsyst_print,
):
    """
    Data class for boron system with isotopes.
    
    Complete dataclass combining boron speciation and isotope calculations
    for comprehensive boron system analysis.
    
    Note:
        This class provides both chemical speciation and isotope fractionation
        calculations for the complete boron system.
    """
    pass

@dataclass(repr=False)
class CarbonBoronParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_CarbonSystem,
    mixin_BoronSystem,
    mixin_CBsyst_print,
):
    """
    Data class for carbon and boron system parameters.
    
    Complete dataclass for combined carbonate and boron system calculations
    without isotope considerations.
    
    Note:
        This class enables coupled carbon-boron calculations where boron
        speciation provides additional pH constraints for the carbonate system.
    """
    pass

@dataclass(repr=False)
class CarbonBoronIsotopeParams(
    CBsystData,
    mixin_CBsyst_config,
    mixin_ConservativeIons,
    mixin_Conditions,
    mixin_pHConversion,
    mixin_CarbonSystem,
    mixin_BoronSystem,
    mixin_BoronIsotopes,
    mixin_CBsyst_print,
):
    """
    Data class for carbon and boron system with isotopes.
    
    Complete dataclass for comprehensive marine carbonate system calculations
    including carbon speciation, boron speciation, and boron isotope fractionation.
    
    Note:
        This is the most comprehensive dataclass, supporting all CBsyst
        calculation modes including paleoceanographic applications using
        boron isotopes as pH proxies.
    """
    pass    

def create_dataclass(**kwargs) -> Union[CarbonSystemParams, BoronSystemParams, BoronIsotopeParams, 
                                      BoronParams, CarbonBoronParams, CarbonBoronIsotopeParams]:
    """
    Create appropriate dataclass based on provided parameters.
    
    Automatically determines the most appropriate dataclass type based on which
    parameters are provided, then creates and returns an instance of that class
    with the provided parameters.
    
    Args:
        **kwargs: Arbitrary keyword arguments representing system parameters.
        
    Returns:
        Instance of the appropriate CBsyst dataclass populated with provided parameters.
        
    Note:
        The function analyzes which parameter categories (carbon, boron, isotopes)
        have non-None values and selects the minimal dataclass that can accommodate
        all provided parameters. Input tracking is automatically handled.
    """
    given_pH = [field for field in mixin_pHConversion.__dataclass_fields__ if kwargs.get(field) is not None]

    given_constants = [field for field in mixin_Conditions.__dataclass_fields__ if kwargs.get(field) is not None]
    
    given_carbon = [field for field in mixin_CarbonSystem.__dataclass_fields__ if kwargs.get(field) is not None]
    has_carbon = len(given_carbon) > 0
    # has_carbon = any(kwargs.get(field) is not None for field in mixin_CarbonSystem.__dataclass_fields__)
    given_boron = [field for field in mixin_BoronSystem.__dataclass_fields__ if kwargs.get(field) is not None]
    has_boron = len(given_boron) > 0
    # has_boron = any(kwargs.get(field) is not None for field in mixin_BoronSystem.__dataclass_fields__)

    given_isotopes = [field for field in mixin_BoronIsotopes.__dataclass_fields__ if kwargs.get(field) is not None]
    has_isotopes = len(given_isotopes) > 0
    # has_isotopes = any(kwargs.get(field) is not None for field in mixin_BoronIsotopes.__dataclass_fields__)

    if ((has_carbon and has_boron and has_isotopes)
        or (has_carbon and has_isotopes)):
        param_class = CarbonBoronIsotopeParams
        given = given_carbon + given_boron + given_isotopes
    elif has_carbon and has_boron:
        param_class = CarbonBoronParams
        given = given_carbon + given_boron
    elif has_boron and has_isotopes:
        param_class = BoronParams
        given = given_boron + given_isotopes
    elif has_carbon:
        param_class = CarbonSystemParams
        given = given_carbon
    elif has_boron:
        param_class = BoronSystemParams
        given = given_boron
    elif has_isotopes:
        param_class = BoronIsotopeParams
        given = given_isotopes
    
    valid_kwargs = {k: v for k, v in kwargs.items()
                    if k in param_class.__dataclass_fields__}
    valid_kwargs['inputs'] = tuple(given_pH + given + given_constants)
    
    return param_class(**valid_kwargs)

# Class for paramter solvers

@dataclass
class SolverRule:
    input_params: Tuple[str, str]
    target_params: List[str]
    function: Callable
    post_calculations: Optional[List[Callable]] = None
    pre_calculations: Optional[Callable] = None