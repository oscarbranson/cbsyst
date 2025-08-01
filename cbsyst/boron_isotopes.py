# B isotope fns

import numpy as np
from cbsyst.helpers import NnotNone, Bunch
from .boron import chiB_calc
from .uncertainties import negative_log10_preserve_type, sqrt_preserve_type
from typing import Union, Optional, Dict, Tuple, Callable, List, Any
from .dataclasses import KValues, CBsystData

# B isotope fractionation factors
def get_alphaB() -> float:
    """
    Get Klochko fractionation factor for boron isotopes.
    
    Returns the standard alpha fractionation factor for boron isotopes
    between boric acid (B(OH)3) and borate ion (B(OH)4-) from Klochko et al.
    
    Returns:
        Alpha fractionation factor (dimensionless).
        
    Note:
        Based on Klochko et al. fractionation data. This is the ratio of
        isotope ratios: (11/10 B in B(OH)3) / (11/10 B in B(OH)4-).
    """
    return 1.0272

def get_epsilonB() -> float:
    """
    Get Klochko epsilon for boron isotope fractionation.
    
    Returns the standard epsilon fractionation factor (alpha expressed
    in delta notation) for boron isotopes between boric acid and borate ion.
    
    Returns:
        Epsilon fractionation factor in permil.
        
    Note:
        Epsilon = (alpha - 1) * 1000, expressing the fractionation
        in delta notation units.
    """
    return alpha_to_epsilon(get_alphaB())

def alpha_to_epsilon(alphaB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert alpha fractionation factor to epsilon (delta notation).

    Converts the isotope fractionation factor alpha to epsilon, which expresses
    the same fractionation in delta notation (permil scale). This is useful
    for expressing fractionation in the same units as isotope measurements.

    Args:
        alphaB: The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns:
        Alpha fractionation factor expressed in delta notation (epsilon).
        
    Note:
        Epsilon = (alpha - 1) * 1000. Both represent the same physical
        fractionation but in different mathematical forms.
    """
    return (alphaB-1)*1000

def epsilon_to_alpha(epsilonB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert epsilon (delta notation) to alpha fractionation factor.

    Converts the isotope fractionation factor from epsilon (delta notation)
    to alpha (ratio form). This is the inverse of alpha_to_epsilon.

    Args:
        epsilonB: The isotope fractionation factor (11/10 BO3)/(11/10 BO4) 
                 expressed in delta notation (permil).

    Returns:
        The isotope fractionation factor as a ratio (alpha).
        
    Note:
        Alpha = (epsilon / 1000) + 1. This converts from permil units
        back to the ratio form used in equilibrium calculations.
    """
    return (epsilonB/1000)+1


# Isotope Unit Converters
def A11_to_d11(A11: Union[float, np.ndarray], SRM_ratio: float = 4.04367) -> Union[float, np.ndarray]:
    """
    Convert fractional abundance (A11) to delta notation (d11).

    Converts the fractional abundance of 11B to delta notation relative
    to a standard reference material (typically NIST SRM 951).

    Args:
        A11: The fractional abundance of 11B: 11B / (11B + 10B).
        SRM_ratio: The 11B/10B ratio of the standard reference material,
                  by default NIST951 which is 4.04367.

    Returns:
        A11 expressed in delta notation (d11) in permil.
        
    Note:
        Delta notation: d11 = ((Rsample/Rstandard) - 1) * 1000,
        where R = 11B/10B ratio.
    """
    return ((A11 / (1 - A11)) / SRM_ratio - 1) * 1000

def A11_to_R11(A11: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert fractional abundance (A11) to isotope ratio (R11).

    Converts the fractional abundance of 11B to the isotope ratio 11B/10B.
    This is a simple algebraic conversion between two ways of expressing
    the same isotopic composition.

    Args:
        A11: The fractional abundance of 11B: 11B / (11B + 10B).

    Returns:
        A11 expressed as an isotope ratio (R11 = 11B/10B).
        
    Note:
        Conversion: R11 = A11 / (1 - A11). This follows from the
        definition of fractional abundance.
    """
    return A11 / (1 - A11)

def d11_to_A11(d11: Union[float, np.ndarray], SRM_ratio: float = 4.04367) -> Union[float, np.ndarray]:
    """
    Convert delta notation (d11) to fractional abundance (A11).

    Converts delta notation (permil deviation from standard) to fractional
    abundance of 11B. This is the inverse of A11_to_d11.

    Args:
        d11: The isotope composition expressed in delta notation (permil).
        SRM_ratio: The 11B/10B ratio of the standard reference material,
                  by default NIST951 which is 4.04367.

    Returns:
        Delta notation (d11) expressed as fractional abundance (A11).
        
    Note:
        Inverse conversion of A11_to_d11. First converts to ratio,
        then to fractional abundance.
    """
    return SRM_ratio * (d11 / 1e3 + 1.0) / (SRM_ratio * (d11 / 1e3 + 1.0) + 1.0)

def d11_to_R11(d11: Union[float, np.ndarray], SRM_ratio: float = 4.04367) -> Union[float, np.ndarray]:
    """
    Convert delta notation (d11) to isotope ratio (R11).

    Converts delta notation (permil deviation from standard) to the
    isotope ratio 11B/10B. This is a direct conversion from delta
    notation to ratio form.

    Args:
        d11: The isotope composition expressed in delta notation (permil).
        SRM_ratio: The 11B/10B ratio of the standard reference material,
                  by default NIST951 which is 4.04367.

    Returns:
        Delta notation (d11) expressed as isotope ratio (R11 = 11B/10B).
        
    Note:
        Conversion: R11 = (d11/1000 + 1) * SRM_ratio. This follows
        directly from the definition of delta notation.
    """
    return (d11 / 1000 + 1) * SRM_ratio

def R11_to_d11(R11: Union[float, np.ndarray], SRM_ratio: float = 4.04367) -> Union[float, np.ndarray]:
    """
    Convert isotope ratio (R11) to delta notation (d11).

    Converts the isotope ratio 11B/10B to delta notation relative to
    a standard reference material. This is the inverse of d11_to_R11.

    Args:
        R11: The isotope ratio (11B/10B).
        SRM_ratio: The 11B/10B ratio of the standard reference material,
                  by default NIST951 which is 4.04367.

    Returns:
        R11 expressed in delta notation (d11) in permil.
        
    Note:
        Standard delta notation calculation: 
        d11 = (R11/SRM_ratio - 1) * 1000.
    """
    return (R11 / SRM_ratio - 1) * 1000

def R11_to_A11(R11: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert isotope ratio (R11) to fractional abundance (A11).

    Converts the isotope ratio 11B/10B to fractional abundance of 11B.
    This is a simple algebraic conversion between two ways of expressing
    the same isotopic composition.

    Args:
        R11: The isotope ratio (11B/10B).

    Returns:
        R11 expressed as fractional abundance (A11).
        
    Note:
        Conversion: A11 = R11 / (1 + R11). This follows from the
        definition of fractional abundance and isotope ratio.
    """
    return R11 / (1 + R11)

# Alpha Converters
def ABO3_to_ABO4(ABO3: Union[float, np.ndarray], alphaB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert isotope fractional abundance of boric acid to borate ion.

    Calculates the fractional abundance of 11B in borate ion (B(OH)4-)
    from the fractional abundance in boric acid (B(OH)3) using the
    isotope fractionation factor.

    Args:
        ABO3: The fractional abundance of 11B in boric acid (B(OH)3).
        alphaB: The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns:
        ABO4: The fractional abundance of 11B in borate ion (B(OH)4-).
        
    Note:
        Uses the equilibrium fractionation relationship between the two
        boron species. The fractionation factor relates the isotope ratios
        of the two species at equilibrium.
    """
    return (1 / ((alphaB / ABO3) - alphaB + 1) )

def ABO3_or_ABO4(ABO3: Optional[Union[float, np.ndarray]], ABO4: Optional[Union[float, np.ndarray]], 
                 alphaB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Helper function to determine ABO4 from ABO3 if needed.

    Ensures that ABO4 is available by converting from ABO3 if necessary.
    This is a utility function to handle cases where either ABO3 or ABO4
    might be provided, but ABO4 is needed for calculations.

    Args:
        ABO3: The fractional abundance of 11B in boric acid (B(OH)3).
        ABO4: The fractional abundance of 11B in borate ion (B(OH)4-).
        alphaB: The isotope fractionation factor for (11/10 BO3)/(11/10 BO4).

    Returns:
        ABO4: The fractional abundance of 11B in borate ion (B(OH)4-).
        
    Raises:
        ValueError: If neither ABO3 nor ABO4 is specified.
        
    Note:
        At least one of ABO3 or ABO4 must be provided. If only ABO3 is
        given, ABO4 is calculated using the fractionation factor.
    """
    if NnotNone(ABO3, ABO4) < 1:
        raise(ValueError("Either ABO4 or ABO3 must be specified"))
    elif ABO4 is None:
        ABO4 = ABO3_to_ABO4(ABO3,alphaB)
    return ABO4


# Base Functions
# Calculate total boron isotope fractional abundance using borate ion (B(OH)4)
def calculate_ABT(H: Union[float, np.ndarray], Ks: Union[KValues, dict], alphaB: Union[float, np.ndarray], 
                  ABO4: Optional[Union[float, np.ndarray]] = None, 
                  ABO3: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate total boron isotope fractional abundance from pH and species abundance.

    Calculates the fractional abundance of 11B in total dissolved boron (ABT)
    from hydrogen ion concentration and the fractional abundance in one of the
    boron species (either B(OH)3 or B(OH)4-), using the speciation and 
    fractionation relationships.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        alphaB: The fractionation factor between B(OH)3 and B(OH)4-.
        ABO4: The fractional abundance of 11B in B(OH)4-.
        ABO3: The fractional abundance of 11B in B(OH)3.

    Returns:
        The fractional abundance of 11B in total dissolved boron (ABT).
        
    Note:
        Either ABO4 or ABO3 must be provided. The calculation uses boron
        speciation (chiB) and isotope fractionation to determine the
        bulk isotope composition.
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

    chiB = chiB_calc(H, Ks)
    return (
        ABO4
        * (
            -ABO4 * alphaB * chiB
            + ABO4 * alphaB
            + ABO4 * chiB
            - ABO4
            + alphaB * chiB
            - chiB
            + 1
        )
        / (ABO4 * alphaB - ABO4 + 1))

# Calculate pH using isotope fractional abundance of borate ion (B(OH)4)
def calculate_H(Ks: Union[KValues, dict], alphaB: Union[float, np.ndarray], ABT: Union[float, np.ndarray], 
                ABO4: Optional[Union[float, np.ndarray]] = None, 
                ABO3: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from isotope abundances.

    Calculates the hydrogen ion concentration ([H+]) from the total boron
    isotope abundance (ABT) and the abundance in one of the boron species,
    using the isotope fractionation and chemical speciation relationships.

    Args:
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        alphaB: Fractionation factor between B(OH)3 and B(OH)4-.
        ABT: Fractional abundance of 11B in total dissolved boron.
        ABO4: Fractional abundance of 11B in B(OH)4-.
        ABO3: Fractional abundance of 11B in B(OH)3.
        
    Returns:
        Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        Either ABO4 or ABO3 must be provided. This is the inverse calculation
        to calculate_ABT, solving for [H+] given the isotope constraints.
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)

    return (Ks.KB / ((alphaB / (1 - ABO4 + alphaB * ABO4) - 1) / (ABT / ABO4 - 1) - 1))

# Calculate isotope fractional abundance of boric acid (B(OH)3)
def calculate_ABO3(H: Union[float, np.ndarray], Ks: Union[KValues, dict], ABT: Union[float, np.ndarray], 
                   alphaB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractional abundance of 11B in boric acid from system constraints.

    Calculates the fractional abundance of 11B in boric acid (B(OH)3) from
    hydrogen ion concentration, total boron isotope abundance, and fractionation
    factor. Uses the quadratic solution to the isotope mass balance equations.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        ABT: The fractional abundance of 11B in total dissolved boron.
        alphaB: The fractionation factor between B(OH)3 and B(OH)4-.

    Returns:
        The fractional abundance of 11B in B(OH)3.
        
    Note:
        This calculation involves solving a quadratic equation that arises from
        the mass balance and fractionation constraints. The sqrt_preserve_type
        function maintains uncertainty propagation if present.
    """
    chiB = chiB_calc(H, Ks)
    return (
        ABT * alphaB
        - ABT
        + alphaB * chiB
        - chiB
        - sqrt_preserve_type(
            ABT ** 2 * alphaB ** 2
            - 2 * ABT ** 2 * alphaB
            + ABT ** 2
            - 2 * ABT * alphaB ** 2 * chiB
            + 2 * ABT * alphaB
            + 2 * ABT * chiB
            - 2 * ABT
            + alphaB ** 2 * chiB ** 2
            - 2 * alphaB * chiB ** 2
            + 2 * alphaB * chiB
            + chiB ** 2
            - 2 * chiB
            + 1
        )
        + 1
    ) / (2 * chiB * (alphaB - 1))

# Calculate isotope fractional abundance of borate ion (B(OH)4)
def calculate_ABO4(H: Union[float, np.ndarray], Ks: Union[KValues, dict], ABT: Union[float, np.ndarray], 
                   alphaB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractional abundance of 11B in borate ion from system constraints.

    Calculates the fractional abundance of 11B in borate ion (B(OH)4-) from
    hydrogen ion concentration, total boron isotope abundance, and fractionation
    factor. Uses the quadratic solution to the isotope mass balance equations.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        ABT: The fractional abundance of 11B in total dissolved boron.
        alphaB: The fractionation factor between B(OH)3 and B(OH)4-.

    Returns:
        The fractional abundance of 11B in B(OH)4-.
        
    Note:
        This is the complementary calculation to calculate_ABO3, solving for
        the borate ion abundance instead of boric acid. Also uses quadratic
        solution with uncertainty preservation.
    """
    chiB = chiB_calc(H, Ks)
    return -(
        ABT * alphaB
        - ABT
        - alphaB * chiB
        + chiB
        + sqrt_preserve_type(
            ABT ** 2 * alphaB ** 2
            - 2 * ABT ** 2 * alphaB
            + ABT ** 2
            - 2 * ABT * alphaB ** 2 * chiB
            + 2 * ABT * alphaB
            + 2 * ABT * chiB
            - 2 * ABT
            + alphaB ** 2 * chiB ** 2
            - 2 * alphaB * chiB ** 2
            + 2 * alphaB * chiB
            + chiB ** 2
            - 2 * chiB
            + 1
        )
        - 1
    ) / (2 * alphaB * chiB - 2 * alphaB - 2 * chiB + 2)

# Calculate alpha using isotope fractional abundance of boric acid (B(OH)3)
def calculate_alpha_ABO3(H: Union[float, np.ndarray], Ks: Union[KValues, dict], ABT: Union[float, np.ndarray], 
                         ABO3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractionation factor from boric acid isotope abundance.

    Calculates the isotope fractionation factor (alpha) between B(OH)3 and
    B(OH)4- from the hydrogen ion concentration, total boron isotope abundance,
    and the fractional abundance of 11B in boric acid.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        ABT: The fractional abundance of 11B in total dissolved boron.
        ABO3: The fractional abundance of 11B in boric acid (B(OH)3).

    Returns:
        The fractionation factor between B(OH)3 and B(OH)4- (alpha).
        
    Note:
        This is an inverse calculation that determines what fractionation
        factor would be required to produce the observed isotope distribution
        given the chemical speciation.
    """
    return ( (1
            / ((H/Ks.KB) * (ABT - ABO3) + ABT)) 
            / (ABO3 -1))

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_alpha_ABO4(H: Union[float, np.ndarray], Ks: Union[KValues, dict], ABT: Union[float, np.ndarray], 
                         ABO4: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate fractionation factor from borate ion isotope abundance.

    Calculates the isotope fractionation factor (alpha) between B(OH)3 and
    B(OH)4- from the hydrogen ion concentration, total boron isotope abundance,
    and the fractional abundance of 11B in borate ion.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Dictionary or dataclass of stoichiometric equilibrium constants.
        ABT: The fractional abundance of 11B in total dissolved boron.
        ABO4: The fractional abundance of 11B in borate ion (B(OH)4-).

    Returns:
        The fractionation factor between B(OH)3 and B(OH)4- (alpha).
        
    Note:
        Complementary calculation to calculate_alpha_ABO3, determining the
        fractionation factor from the borate ion isotope abundance instead
        of the boric acid abundance.
    """
    return ( (1/ABO4 - 1)
            / (1 / (ABT - ((ABO4-ABT)/(H/Ks.KB))) -1) )

# Calculate alpha using isotope fractional abundance of borate ion (B(OH)4)
def calculate_KB(H: Union[float, np.ndarray], alphaB: Union[float, np.ndarray], ABT: Union[float, np.ndarray], 
                 ABO4: Optional[Union[float, np.ndarray]] = None, 
                 ABO3: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate stoichiometric equilibrium constant for boron from isotope constraints.

    Calculates the boron equilibrium constant (KB) from hydrogen ion concentration,
    fractionation factor, total boron isotope abundance, and the abundance in one
    of the boron species. This is an inverse calculation that determines what KB
    would be required to produce the observed isotope distribution.

    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        alphaB: The fractionation factor between B(OH)3 and B(OH)4-.
        ABT: The fractional abundance of 11B in total dissolved boron.
        ABO4: The fractional abundance of 11B in borate ion (B(OH)4-).
        ABO3: The fractional abundance of 11B in boric acid (B(OH)3).

    Returns:
        The stoichiometric equilibrium constant for boron (KB).
        
    Note:
        Either ABO4 or ABO3 must be provided. This inverse calculation can be
        used to determine KB from isotope measurements, though KB is typically
        known from independent measurements.
    """
    ABO4 = ABO3_or_ABO4(ABO3,ABO4,alphaB)
    return (H
            / ((ABO4 - ABT)
            / ( ABT 
            - 1 / ( (1/alphaB) * (1/ABO4 -1) + 1) )))

def calc_B_isotopes(pHtot: Optional[Union[float, np.ndarray]] = None, 
                    ABT: Optional[Union[float, np.ndarray]] = None, 
                    ABO3: Optional[Union[float, np.ndarray]] = None, 
                    ABO4: Optional[Union[float, np.ndarray]] = None, 
                    alphaB: Optional[Union[float, np.ndarray]] = None, 
                    Ks: Optional[Union[KValues, dict]] = None, **kwargs) -> Bunch:
    """
    Calculate all boron isotope species from minimal input parameters.

    This is the main boron isotope calculation function that determines all
    boron isotope abundances and pH from minimal input. Can solve from various
    combinations of pH, total isotope abundance, and species-specific abundances.

    Args:
        pHtot: pH on total scale.
        ABT: Fractional abundance of 11B in total dissolved boron.
        ABO3: Fractional abundance of 11B in boric acid (B(OH)3).
        ABO4: Fractional abundance of 11B in borate ion (B(OH)4-).
        alphaB: Fractionation factor between B(OH)3 and B(OH)4-.
        Ks: Dictionary or dataclass of equilibrium constants.
        **kwargs: Additional keyword arguments.

    Returns:
        Bunch object containing all calculated boron isotope species and pH.

    Raises:
        ValueError: If insufficient parameters are provided to solve the system.

    Note:
        Requires at least two parameters to solve the system. If pH is provided,
        needs either ABT or both ABO3/ABO4. If pH is not provided, needs ABT
        plus one of ABO3 or ABO4.
    """
    # determine pH and ABT
    if pHtot is not None:  # pH is known
        H = 10.0**-pHtot
        if ABT is None:
            ABT = calculate_ABT(H=H, Ks=Ks, alphaB=alphaB, ABO3=ABO3, ABO4=ABO4)
    else:  # pH is not known
        if ABT is not None:
            H = calculate_H(Ks=Ks, alphaB=alphaB, ABT=ABT, ABO3=ABO3, ABO4=ABO4)
            pHtot = negative_log10_preserve_type(H)
        else:
            raise ValueError('ABT and one of ABO3 or ABO4 must be specified if pH is missing.')
    
    if ABO3 is None:
        ABO3 = calculate_ABO3(H=H, Ks=Ks, ABT=ABT, alphaB=alphaB)
    if ABO4 is None:
        ABO4 = calculate_ABO4(H=H, Ks=Ks, ABT=ABT, alphaB=alphaB)
    
    return Bunch({
        'pHtot': pHtot,
        'ABT': ABT,
        'ABO4': ABO4,
        'ABO3': ABO3,
        'H': H
    })

# CBsyst 1.0 functions

def solve_pH_ABT(params: CBsystData) -> None:
    """Solve boron isotope system from pH and total abundance."""
    if params.H is None: params.H = 10**-params.pHtot

# TODO: needs optimising
def solve_pH_ABO4_ABO3(params: CBsystData) -> None:
    """Solve boron isotope system from pH and species-specific abundances."""
    if params.H is None: params.H = 10**-params.pHtot
    params.ABT = calculate_ABT(H=params.H, Ks=params.Ks, alphaB=params.alphaB, ABO3=params.ABO3, ABO4=params.ABO4)

# TODO: needs optimising
def solve_ABT_ABO3_ABO4(params: CBsystData) -> None:
    """Solve boron isotope system from total and species-specific abundances."""
    if params.H is None: params.H = calculate_H(Ks=params.Ks, ABT=params.ABT, ABO3=params.ABO3, ABO4=params.ABO4, alphaB=params.alphaB)

def solve_ABO3_ABO4(params: CBsystData) -> None:
    """Placeholder solver - insufficient parameters."""
    raise NotImplementedError('ABT and one of ABO3 or ABO4 must be specified if pH is missing.')

SOLVERS: Dict[Tuple[str, ...], Callable[[CBsystData], None]] = {
    ('pHtot', 'ABT'): solve_pH_ABT,
    ('pHtot', 'ABO4'): solve_pH_ABO4_ABO3,
    ('pHtot', 'ABO3'): solve_pH_ABO4_ABO3,
    ('ABT', 'ABO4'): solve_ABT_ABO3_ABO4,
    ('ABT', 'ABO3'): solve_ABT_ABO3_ABO4,
    ('ABO3', 'ABO4'): solve_ABO3_ABO4,
    ('ABT', 'ABO3', 'ABO4'): solve_ABO3_ABO4,
}

def given(params: CBsystData) -> List[Any]:
    """
    Check which boron isotope parameters are given in the parameters.
    
    Identifies which boron isotope parameters have been provided (are not None)
    in the input dataclass. Used to determine which solver to use.
    
    Args:
        params: CBsyst data structure containing boron isotope parameters.
        
    Returns:
        List of parameter values that are not None.
    """
    valid_inputs = ['ABT', 'ABO3', 'ABO4', 'dBT', 'dBO3', 'dBO4']
    return [params.get(p) for p in valid_inputs if params.get(p) is not None]

def n_given(params: CBsystData) -> int:
    """
    Count number of provided boron isotope parameters.
    
    Args:
        params: CBsyst data structure containing boron isotope parameters.
        
    Returns:
        Number of boron isotope parameters that are not None.
    """
    return len(given(params))

def calculate_ABO3_ABO4(params: CBsystData) -> None:
    """
    Calculate missing boron isotope species abundances.
    
    Calculates ABO3 and ABO4 from hydrogen ion concentration, total abundance,
    and fractionation factor if they are not already provided.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.ABO3 is None: params.ABO3 = calculate_ABO3(H=params.H, Ks=params.Ks, ABT=params.ABT, alphaB=params.alphaB)
    if params.ABO4 is None: params.ABO4 = calculate_ABO4(H=params.H, Ks=params.Ks, ABT=params.ABT, alphaB=params.alphaB)

def delta_to_abundance(params: CBsystData) -> None:
    """
    Convert delta notation to fractional abundance where needed.
    
    Converts boron isotope delta values to fractional abundances if delta
    values are provided but abundances are not.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.dBT is not None:
        if params.ABT is None: params.ABT = d11_to_A11(params.dBT)
    if params.dBO3 is not None:
        if params.ABO3 is None: params.ABO3 = d11_to_A11(params.dBO3)
    if params.dBO4 is not None:
        if params.ABO4 is None: params.ABO4 = d11_to_A11(params.dBO4)

def abundance_to_delta(params: CBsystData) -> None:
    """
    Convert fractional abundance to delta notation where needed.
    
    Converts boron isotope fractional abundances to delta notation if
    abundances are available but delta values are not.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.ABT is not None:
        if params.dBT is None: params.dBT = A11_to_d11(params.ABT)
    if params.ABO3 is not None:
        if params.dBO3 is None: params.dBO3 = A11_to_d11(params.ABO3)
    if params.ABO4 is not None:
        if params.dBO4 is None: params.dBO4 = A11_to_d11(params.ABO4)

def solve_B_isotopes(params: CBsystData) -> None:
    """
    Solve the complete boron isotope system from provided parameters.
    
    Main boron isotope system solver that:
    1. Converts delta notation to abundances as needed
    2. Identifies the parameter combination and selects appropriate solver
    3. Solves for missing parameters
    4. Calculates all remaining species abundances
    5. Converts abundances back to delta notation
    
    Args:
        params: CBsyst data structure, modified in place.
        
    Raises:
        ValueError: If no solver found for the parameter combination.
    """
    delta_to_abundance(params)

    AB_params = ['pHtot', 'ABT', 'ABO4', 'ABO3']
    provided = tuple([p for p in AB_params if params.get(p) is not None])
    # params.inputs += provided
    
    solver = SOLVERS.get(provided)
    if solver is None:
        raise ValueError(f"No solver found for parameter combination: {provided}")

    solver(params)

    if params.pHtot is None: params.pHtot = negative_log10_preserve_type(params.H)
    calculate_ABO3_ABO4(params)
    abundance_to_delta(params)
