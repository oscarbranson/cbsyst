from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

class CBsystData:
    def __getitem__(self, key):
        """Enable dict-style getting: cp['pHtot']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __setitem__(self, key, value):
        """Enable dict-style setting: cp['pHtot'] = 8.1"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __contains__(self, key):
        """Enable 'in' operator: 'pHtot' in cp"""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dict-style get with default: cp.get('pHtot', None)"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """Return field names like dict.keys()"""
        return self.__dataclass_fields__.keys()
    
    def values(self):
        """Return field values like dict.values()"""
        return (getattr(self, field) for field in self.__dataclass_fields__)
    
    def items(self):
        """Return (key, value) pairs like dict.items()"""
        return ((field, getattr(self, field)) for field in self.__dataclass_fields__)

class mixin_CBsyst_print:
    # TODO: needs to work with arrays and uncertainties
    def __repr__(self):
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
            if self.get(p):
                out += f'{p:<{col1}}{self._fmt_value(self[p]):>{col2}}\n'
        out += section

        return out
    
    def _fmt_value(self, value):
        """Format value for printing, handling arrays and uncertainties"""
        if isinstance(value, np.ndarray):
            return f'{value[0]:.2f} ... {value[-1]:.2f} (n={len(value)})' if len(value) > 1 else f'{value[0]:.2f}'
        elif hasattr(value, 'nominal_value'):
            return f'{value.nominal_value:.2f} ± {value.std_dev:.2f}'
        else:
            return f'{value:.2f}'


@dataclass(repr=False)
class KValues(CBsystData):
    K0: Union[float,np.ndarray]
    K1: Union[float,np.ndarray]
    K2: Union[float,np.ndarray]
    KW: Union[float,np.ndarray]
    KB: Union[float,np.ndarray]
    KS: Union[float,np.ndarray]
    KspA: Union[float,np.ndarray]
    KspC: Union[float,np.ndarray]
    KP1: Union[float,np.ndarray]
    KP2: Union[float,np.ndarray]
    KP3: Union[float,np.ndarray]
    KSi: Union[float,np.ndarray]
    KF: Union[float,np.ndarray]

    def __repr__(self):
        out = 'K values:\n----------------\n'
        for k in self.__dataclass_fields__:
            out += f'{k:<5}{self[k]:.5e}\n'
        out += '----------------'
        return out

@dataclass(repr=False)
class mixin_ConservativeIons:
    # Seawater Chemistry
    Ca: Union[float,np.ndarray] = 0.0102821
    Mg: Union[float,np.ndarray] = 0.0528171
    PT: Union[float,np.ndarray] = 0.0
    SiT: Union[float,np.ndarray] = 0.0
    ST: Optional[Union[float,np.ndarray]] = None
    FT: Optional[Union[float,np.ndarray]] = None
    BT: Optional[Union[float,np.ndarray]] = None

@dataclass(repr=False)
class mixin_Conditions:
    # Environmental conditions
    T_in: Union[float,np.ndarray] = 25.0
    S_in: Union[float,np.ndarray] = 35.0
    P_in: Union[float,np.ndarray] = 0.0
    T_out: Optional[Union[float,np.ndarray]] = None
    S_out: Optional[Union[float,np.ndarray]] = None
    P_out: Optional[Union[float,np.ndarray]] = None

    # Constants
    Ks: Optional[KValues] = None

@dataclass(repr=False)
class mixin_pHConversion:
    # pH conversion parts
    FREEtoTOT: Optional[Union[float,np.ndarray]] = None
    SWStoTOT: Optional[Union[float,np.ndarray]] = None

    # pH scales
    pHtot: Optional[Union[float,np.ndarray]] = None
    pHsws: Optional[Union[float,np.ndarray]] = None
    pHfree: Optional[Union[float,np.ndarray]] = None
    pHNBS: Optional[Union[float,np.ndarray]] = None

    # H and free H
    H: Optional[Union[float,np.ndarray]] = None
    fH: Optional[Union[float,np.ndarray]] = None

@dataclass(repr=False)
class mixin_CarbonSystem:
    # Carbon parameters
    DIC: Optional[Union[float,np.ndarray]] = None
    TA: Optional[Union[float,np.ndarray]] = None
    CO2: Optional[Union[float,np.ndarray]] = None
    HCO3: Optional[Union[float,np.ndarray]] = None
    CO3: Optional[Union[float,np.ndarray]] = None
    pCO2: Optional[Union[float,np.ndarray]] = None
    fCO2: Optional[Union[float,np.ndarray]] = None

    # Omega parameters
    OmegaC: Optional[Union[float,np.ndarray]] = None
    OmegaA: Optional[Union[float,np.ndarray]] = None

    # Alkalinity Components
    CAlk: Optional[Union[float,np.ndarray]] = None
    BAlk: Optional[Union[float,np.ndarray]] = None
    PAlk: Optional[Union[float,np.ndarray]] = None
    SiAlk: Optional[Union[float,np.ndarray]] = None
    OH: Optional[Union[float,np.ndarray]] = None
    Hfree: Optional[Union[float,np.ndarray]] = None
    HSO4: Optional[Union[float,np.ndarray]] = None
    HF: Optional[Union[float,np.ndarray]] = None

@dataclass(repr=False)
class mixin_BoronSystem:
    BO3: Optional[Union[float,np.ndarray]] = None
    BO4: Optional[Union[float,np.ndarray]] = None
    BT: Optional[Union[float,np.ndarray]] = None

@dataclass(repr=False)
class mixin_BoronIsotopes:
    alphaB: Optional[Union[float,np.ndarray]] = None
    epsilonB: Optional[Union[float,np.ndarray]] = None

    ABT: Optional[Union[float,np.ndarray]] = None
    ABO3: Optional[Union[float,np.ndarray]] = None
    ABO4: Optional[Union[float,np.ndarray]] = None

    dBT: Optional[Union[float,np.ndarray]] = None
    dBO3: Optional[Union[float,np.ndarray]] = None
    dBO4: Optional[Union[float,np.ndarray]] = None

@dataclass(repr=False)
class mixin_CBsyst_config:
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
    """Data class for carbon system parameters only"""
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
    """Data class for boron system parameters only"""
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
    """Data class for boron isotope parameters only"""
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
    """Data class for boroz system with isotopes"""
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
    """Data class for carbon and boron system parameters"""
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
    """Data class for carbon and boron system with isotopes"""
    pass    

def create_dataclass(**kwargs):
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

