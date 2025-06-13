import scipy.optimize as opt
import numpy as np
import uncertainties
import uncertainties.unumpy as unp
from cbsyst.helpers import noms, cast_array, Bunch, maxShape, calc_fH

def _zero_wrapper(ps, fn, bounds=(10 ** -14, 10 ** -1)):
    """
    Wrapper to handle zero finders.
    
    If any parameters have uncertainties, finite difference is used to
    propagate uncertainties through the zero finder.
    """
    # Check if any parameters have uncertainties
    has_uncertainties = any(_has_uncertainties(p) for p in ps if p is not None)
    
    if not has_uncertainties:
        # No uncertainties - use original implementation
        try:
            return opt.brentq(fn, *bounds, args=tuple(ps), xtol=1e-16)
            # brentq is ~100 times faster.
        except ValueError:
            return opt.fsolve(fn, 1, args=tuple(ps))[0]
            # but can be fragile if limits aren't right.
    else:
        # Has uncertainties - use finite difference for uncertainty propagation
        return _zero_wrapper_with_uncertainties(ps, fn, bounds)

def _zero_wrapper_with_uncertainties(ps, fn, bounds=(10 ** -14, 10 ** -1)):
    """
    Zero finder wrapper that propagates uncertainties using finite differences.
    """
    # Extract nominal values for the zero finder
    ps_nominal = []
    for p in ps:
        ps_nominal.append(_extract_nominal_values(p))
    
    # Find the zero using nominal values
    try:
        result_nominal = opt.brentq(fn, *bounds, args=tuple(ps_nominal), xtol=1e-16)
    except ValueError:
        result_nominal = opt.fsolve(fn, 1, args=tuple(ps_nominal))[0]
    
    # Calculate derivatives numerically using finite differences
    derivatives = []
    epsilon = 1e-8  # Small perturbation for finite differences
    
    for i, p in enumerate(ps):
        if hasattr(p, 'nominal_value'):
            # This parameter has uncertainty, calculate derivative
            ps_plus = ps_nominal.copy()
            ps_minus = ps_nominal.copy()
            
            # Calculate delta for perturbation
            delta = abs(ps_nominal[i] * epsilon) if ps_nominal[i] != 0 else epsilon
            ps_plus[i] = ps_nominal[i] + delta
            ps_minus[i] = ps_nominal[i] - delta
            
            # Calculate function values at perturbed points
            try:
                result_plus = opt.brentq(fn, *bounds, args=tuple(ps_plus), xtol=1e-16)
            except ValueError:
                result_plus = opt.fsolve(fn, 1, args=tuple(ps_plus))[0]
                
            try:
                result_minus = opt.brentq(fn, *bounds, args=tuple(ps_minus), xtol=1e-16)
            except ValueError:
                result_minus = opt.fsolve(fn, 1, args=tuple(ps_minus))[0]
            
            # Calculate derivative
            derivative = (result_plus - result_minus) / (2 * delta)
            derivatives.append(derivative)
        else:
            # No uncertainty in this parameter
            derivatives.append(0.0)
    
    # Propagate uncertainties using linear error propagation
    total_variance = 0.0
    for i, p in enumerate(ps):
        if hasattr(p, 'nominal_value'):
            # Scalar ufloat
            total_variance += (derivatives[i] * p.std_dev) ** 2
    
    # Create result with uncertainty
    if total_variance > 0:
        return uncertainties.ufloat(result_nominal, np.sqrt(total_variance))
    else:
        return result_nominal


def _has_uncertainties(obj):
    """
    Check if an object contains uncertainties.
    
    Returns True if:
    - obj is a ufloat (has nominal_value attribute)
    - obj is a uarray (numpy array containing ufloat objects)
    """
    if hasattr(obj, 'nominal_value'):
        # Individual ufloat object
        return True
    elif hasattr(obj, '__iter__') and hasattr(obj, 'dtype') and obj.dtype == object:
        # Possibly a uarray (numpy array with object dtype)
        if len(obj) > 0 and hasattr(obj.flat[0], 'nominal_value'):
            return True
    return False


def _extract_nominal_values(obj):
    """
    Extract nominal values from ufloat or uarray objects.
    
    Returns:
    - For ufloat: obj.nominal_value
    - For uarray: array of nominal values
    - For regular objects: obj unchanged
    """
    if hasattr(obj, 'nominal_value'):
        # Individual ufloat object
        return obj.nominal_value
    elif hasattr(obj, '__iter__') and hasattr(obj, 'dtype') and obj.dtype == object:
        # Possibly a uarray (numpy array with object dtype)
        if len(obj) > 0 and hasattr(obj.flat[0], 'nominal_value'):
            return np.array([item.nominal_value for item in obj])
    # Regular object without uncertainties
    return obj


def uncertainty_propagation_decorator(func):
    """
    Decorator to handle uncertainty propagation for iterative functions.
    
    If any input parameters have uncertainties, this decorator uses finite 
    differences to calculate derivatives and propagate uncertainties.
    If no uncertainties are present, it calls the original function directly.
    
    This is designed for iterative functions like CO2_TA that use Newton-Raphson
    methods which are incompatible with uncertainty objects.
    """
    def wrapper(*args, **kwargs):
        # Check if any arguments have uncertainties
        has_uncertainties = any(_has_uncertainties(arg) for arg in args)
        has_uncertainties = has_uncertainties or any(_has_uncertainties(val) for val in kwargs.values())
        
        if not has_uncertainties:
            # No uncertainties, call original function
            return func(*args, **kwargs)
        
        # Extract nominal values for the original calculation
        args_nominal = []
        for arg in args:
            args_nominal.append(_extract_nominal_values(arg))
        
        kwargs_nominal = {}
        for key, val in kwargs.items():
            kwargs_nominal[key] = _extract_nominal_values(val)
        
        # Calculate nominal result
        result_nominal = func(*args_nominal, **kwargs_nominal)
        
        # Calculate derivatives for uncertainty propagation
        epsilon = 1e-8
        derivatives = []
        
        # Calculate derivatives for positional arguments
        for i, arg in enumerate(args):
            if _has_uncertainties(arg):
                if hasattr(arg, 'nominal_value'):
                    # Individual ufloat object
                    args_plus = list(args_nominal)
                    args_minus = list(args_nominal)
                    
                    # Calculate delta for perturbation
                    delta = abs(args_nominal[i] * epsilon) if args_nominal[i] != 0 else epsilon
                    args_plus[i] = args_nominal[i] + delta
                    args_minus[i] = args_nominal[i] - delta
                    
                    # Calculate function values at perturbed points
                    result_plus = func(*args_plus, **kwargs_nominal)
                    result_minus = func(*args_minus, **kwargs_nominal)
                    
                    # Calculate derivative
                    derivative = (result_plus - result_minus) / (2 * delta)
                    derivatives.append((arg, derivative))
                    
                elif hasattr(arg, '__iter__') and hasattr(arg, 'dtype') and arg.dtype == object:
                    # uarray object - calculate derivatives for each element
                    for j, element in enumerate(arg):
                        if hasattr(element, 'nominal_value'):
                            args_plus = list(args_nominal)
                            args_minus = list(args_nominal)
                            
                            # Create perturbed versions of the array
                            arg_plus = args_nominal[i].copy()
                            arg_minus = args_nominal[i].copy()
                            
                            delta = abs(args_nominal[i][j] * epsilon) if args_nominal[i][j] != 0 else epsilon
                            arg_plus[j] = args_nominal[i][j] + delta
                            arg_minus[j] = args_nominal[i][j] - delta
                            
                            args_plus[i] = arg_plus
                            args_minus[i] = arg_minus
                            
                            # Calculate function values at perturbed points
                            result_plus = func(*args_plus, **kwargs_nominal)
                            result_minus = func(*args_minus, **kwargs_nominal)
                            
                            # Calculate derivative
                            derivative = (result_plus - result_minus) / (2 * delta)
                            derivatives.append((element, derivative, i, j))  # Include indices for uarray handling
        
        # Calculate derivatives for keyword arguments
        for key, val in kwargs.items():
            if _has_uncertainties(val):
                if hasattr(val, 'nominal_value'):
                    # Individual ufloat object
                    kwargs_plus = kwargs_nominal.copy()
                    kwargs_minus = kwargs_nominal.copy()
                    
                    # Calculate delta for perturbation
                    delta = abs(kwargs_nominal[key] * epsilon) if kwargs_nominal[key] != 0 else epsilon
                    kwargs_plus[key] = kwargs_nominal[key] + delta
                    kwargs_minus[key] = kwargs_nominal[key] - delta
                    
                    # Calculate function values at perturbed points
                    result_plus = func(*args_nominal, **kwargs_plus)
                    result_minus = func(*args_nominal, **kwargs_minus)
                    
                    # Calculate derivative
                    derivative = (result_plus - result_minus) / (2 * delta)
                    derivatives.append((val, derivative))
                    
                elif hasattr(val, '__iter__') and hasattr(val, 'dtype') and val.dtype == object:
                    # uarray object - calculate derivatives for each element
                    for j, element in enumerate(val):
                        if hasattr(element, 'nominal_value'):
                            kwargs_plus = kwargs_nominal.copy()
                            kwargs_minus = kwargs_nominal.copy()
                            
                            # Create perturbed versions of the array
                            val_plus = kwargs_nominal[key].copy()
                            val_minus = kwargs_nominal[key].copy()
                            
                            delta = abs(kwargs_nominal[key][j] * epsilon) if kwargs_nominal[key][j] != 0 else epsilon
                            val_plus[j] = kwargs_nominal[key][j] + delta
                            val_minus[j] = kwargs_nominal[key][j] - delta
                            
                            kwargs_plus[key] = val_plus
                            kwargs_minus[key] = val_minus
                            
                            # Calculate function values at perturbed points
                            result_plus = func(*args_nominal, **kwargs_plus)
                            result_minus = func(*args_nominal, **kwargs_minus)
                            
                            # Calculate derivative
                            derivative = (result_plus - result_minus) / (2 * delta)
                            derivatives.append((element, derivative, key, j))  # Include indices for uarray handling
        
        
        # Propagate uncertainties using linear error propagation
        # Handle both individual ufloat and uarray cases
        
        # Check if result is scalar or array
        is_result_array = hasattr(result_nominal, '__len__') and not isinstance(result_nominal, str) and np.ndim(result_nominal) > 0
        
        if is_result_array:
            # Array result - need to calculate uncertainty for each element
            result_length = len(result_nominal)
            result_uncertainties = np.zeros(result_length)
            
            # Calculate uncertainty for each result element
            for k in range(result_length):
                total_variance = 0.0
                for item in derivatives:
                    if len(item) == 2:
                        # Individual ufloat: (param, derivative)
                        param, derivative = item
                        if np.isscalar(derivative):
                            total_variance += (derivative * param.std_dev) ** 2
                        else:
                            total_variance += (derivative[k] * param.std_dev) ** 2
                    elif len(item) == 4:
                        # uarray element: (element, derivative, position, index)
                        element, derivative, position, index = item
                        if np.isscalar(derivative):
                            total_variance += (derivative * element.std_dev) ** 2
                        else:
                            total_variance += (derivative[k] * element.std_dev) ** 2
                
                result_uncertainties[k] = np.sqrt(total_variance)
            
            # Create result array with uncertainties
            if np.any(result_uncertainties > 0):
                result_array = np.array([uncertainties.ufloat(result_nominal[k], result_uncertainties[k]) 
                                       for k in range(result_length)])
                # If it's a single element array, return the scalar
                if result_array.size == 1:
                    return result_array.item()
                return result_array
            else:
                # No uncertainties propagated
                if len(result_nominal) == 1:
                    return result_nominal.item() if hasattr(result_nominal, 'item') else result_nominal[0]
                return result_nominal
                
        else:
            # Scalar result
            total_variance = 0.0
            for item in derivatives:
                if len(item) == 2:
                    # Individual ufloat: (param, derivative)
                    param, derivative = item
                    total_variance += (derivative * param.std_dev) ** 2
                elif len(item) == 4:
                    # uarray element: (element, derivative, position, index)
                    element, derivative, position, index = item
                    total_variance += (derivative * element.std_dev) ** 2
            
            # Handle scalar result
            if hasattr(result_nominal, 'item'):
                result_nominal = result_nominal.item()  # Convert 0-d array to scalar
            if total_variance > 0:
                return uncertainties.ufloat(result_nominal, np.sqrt(total_variance))
            else:
                return result_nominal
    
    return wrapper


# Function types
# Zero-finders: 2-5, 10-15
# Algebraic: 1, 6-9


# Zeebe & Wolf-Gladrow, Appendix B
# 1. CO2 and pH given
def CO2_pH(CO2, pH, Ks):
    """
    Returns DIC
    """
    h = 10.0**-pH
    return CO2 * (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h ** 2)


# 2. CO2 and HCO3 given
def CO2_HCO3(CO2, HCO3, Ks):
    """
    Returns H
    """
    # Don't strip uncertainties - let _zero_wrapper handle them
    par = cast_array(CO2, HCO3, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(CO2, HCO3, Ks.K1, Ks.K2)  # get shape of output

    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_HCO3).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_CO2_HCO3(h, CO2, HCO3, K1, K2):
    # Roots: two negative, one positive - use positive.
    LH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    RH = HCO3 * (h ** 2 + h ** 3 / K1 + K2 * h)
    return LH - RH


# 3. CO2 and CO3
def CO2_CO3(CO2, CO3, Ks):
    """
    Returns H
    """
    par = cast_array(CO2, CO3, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(CO2, CO3, Ks.K1, Ks.K2)  # get shape of output

    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_CO3).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_CO2_CO3(h, CO2, CO3, K1, K2):
    # Roots: one positive, three negative. Use positive.
    LH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    RH = CO3 * (h ** 2 + h ** 3 / K2 + h ** 4 / (K1 * K2))
    return LH - RH


# 4. CO2 and TA
@uncertainty_propagation_decorator
def CO2_TA(CO2, TA, BT, PT, SiT, ST, FT, Ks):
    """
    Returns pH

    Taken from matlab CO2SYS
    """
    
    fCO2 = CO2 / Ks.K0
    L = maxShape(TA, CO2, BT, PT, SiT, ST, FT, Ks.K1)
    if len(L) == 0:
        L = (1,)
        
    pHguess = 8.0
    pHtol = 0.0000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while np.any(abs(deltapH) > pHtol):
        H = 10 ** -pHx
        HCO3 = Ks.K0 * Ks.K1 * fCO2 / H
        CO3 = Ks.K0 * Ks.K1 * Ks.K2 * fCO2 / H ** 2
        CAlk = HCO3 + 2 * CO3
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
        PhosBot = (
            H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        )
        PAlk = PT * PhosTop / PhosBot
        SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + ST / Ks.KS)
        HSO4 = ST / (1 + Ks.KS / Hfree)
        HF = FT / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF
        Slope = ln10 * (HCO3 + 4.0 * CO3 + BAlk * H / (Ks.KB + H) + OH + H)
        deltapH = Residual / Slope

        while np.any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx if pHx.size > 1 else pHx.item()


# 5. CO2 and DIC
def CO2_DIC(CO2, DIC, Ks):
    """
    Returns H
    """
    par = cast_array(CO2, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(CO2, DIC, Ks.K1, Ks.K2)  # get shape of output

    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_DIC).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_CO2_DIC(h, CO2, DIC, K1, K2):
    # Roots: one positive, one negative. Use positive.
    LH = DIC * h ** 2
    RH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    return LH - RH


# 6. pH and HCO3
def pH_HCO3(pH, HCO3, Ks):
    """
    Returns DIC
    """
    h = 10.0**-pH
    return HCO3 * (1 + h / Ks.K1 + Ks.K2 / h)


# 7. pH and CO3
def pH_CO3(pH, CO3, Ks):
    """
    Returns DIC
    """
    h = 10.0**-pH
    return CO3 * (1 + h / Ks.K2 + h ** 2 / (Ks.K1 * Ks.K2))


# 8. pH and TA
def pH_TA(pH, TA, BT, PT, SiT, ST, FT, Ks):
    """
    Returns DIC

    Taken directly from MATLAB CO2SYS.
    """
    H = 10 ** -pH
    # negative alk
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
    PhosBot = H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = PT * PhosTop / PhosBot
    SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
    # positive alk
    Hfree = H / (1 + ST / Ks.KS)
    HSO4 = ST / (1 + Ks.KS / Hfree)
    HF = FT / (1 + Ks.KF / Hfree)
    CAlk = TA - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

    return CAlk * (H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2) / (Ks.K1 * (H + 2.0 * Ks.K2))


# 9. pH and DIC
def pH_DIC(pH, DIC, Ks):
    """
    Returns CO2
    """
    h = 10.0**-pH
    return DIC / (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h ** 2)


# 10. HCO3 and CO3
def HCO3_CO3(HCO3, CO3, Ks):
    """
    Returns H
    """
    par = cast_array(HCO3, CO3, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(HCO3, CO3, Ks.K1, Ks.K2)  # get shape of output
    
    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_CO3).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_HCO3_CO3(h, HCO3, CO3, K1, K2):
    # Roots: one pos, two neg. Use pos.
    LH = HCO3 * (h + h ** 2 / K1 + K2)
    RH = CO3 * (h + h ** 2 / K2 + h ** 3 / (K1 * K2))
    return LH - RH


# 11. HCO3 and TA
@uncertainty_propagation_decorator
def HCO3_TA(HCO3, TA, BT, Ks):
    """
    Returns H
    """
    par = cast_array(
        HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW
    )  # cast parameters into array
    shape = maxShape(HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # get shape of output

    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_TA).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_HCO3_TA(h, HCO3, TA, BT, K1, K2, KB, KW):
    # Roots: one pos, four neg. Use pos.
    LH = TA * (KB + h) * (h ** 3 + K1 * h ** 2 + K1 * K2 * h)
    RH = (
        HCO3
        * (h + h ** 2 / K1 + K2)
        * ((KB + 2 * K2) * K1 * h + 2 * KB * K1 * K2 + K1 * h ** 2)
    ) + (
        (h ** 2 + K1 * h + K1 * K2)
        * (KB * BT * h + KW * KB + KW * h - KB * h ** 2 - h ** 3)
    )
    return LH - RH


# 12. HCO3 amd DIC
def HCO3_DIC(HCO3, DIC, Ks):
    """
    Returns H
    """
    par = cast_array(HCO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(HCO3, DIC, Ks.K1, Ks.K2)  # get shape of output
    
    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_DIC).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_HCO3_DIC(h, HCO3, DIC, K1, K2):
    # Roots: two pos. Use smaller.
    LH = HCO3 * (h + h ** 2 / K1 + K2)
    RH = h * DIC
    return LH - RH


# 13. CO3 and TA
@uncertainty_propagation_decorator
def CO3_TA(CO3, TA, BT, Ks):
    """
    Returns H
    """
    par = cast_array(
        CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW
    )  # cast parameters into array
    shape = maxShape(CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # get shape of output
    
    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_TA).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_CO3_TA(h, CO3, TA, BT, K1, K2, KB, KW):
    # Roots: three neg, two pos. Use larger pos.
    LH = TA * (KB + h) * (h ** 3 + K1 * h ** 2 + K1 * K2 * h)
    RH = (
        CO3
        * (h + h ** 2 / K2 + h ** 3 / (K1 * K2))
        * (K1 * h ** 2 + K1 * h * (KB + 2 * K2) + 2 * KB * K1 * K2)
    ) + (
        (h ** 2 + K1 * h + K1 * K2)
        * (KB * BT * h + KW * KB + KW * h - KB * h ** 2 - h ** 3)
    )
    return LH - RH


# 14. CO3 and DIC
def CO3_DIC(CO3, DIC, Ks):
    """
    Returns H
    """
    par = cast_array(CO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    shape = maxShape(CO3, DIC, Ks.K1, Ks.K2)  # get shape of output

    result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_DIC).reshape(shape)
    
    # If result is a single-element array, extract the scalar
    if result.size == 1:
        return result.item()
    return result


def zero_CO3_DIC(h, CO3, DIC, K1, K2):
    # Roots: one pos, one neg. Use neg.
    LH = CO3 * (1 + h / K2 + h ** 2 / (K1 * K2))
    RH = DIC
    return LH - RH


# 15. TA and DIC
@uncertainty_propagation_decorator
def TA_DIC(TA, DIC, BT, PT, SiT, ST, FT, Ks):
    """
    Returns pH

    Taken directly from MATLAB CO2SYS.
    """
    # determine shape of input
    L = maxShape(TA, DIC, BT, PT, SiT, ST, FT, Ks.K1)
    if len(L) == 0:
        L = (1,)
        
    pHguess = 8.0
    pHtol = 0.00000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while np.any(abs(deltapH) > pHtol):
        H = 10 ** -pHx
        # negative
        Denom = H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2
        CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
        PhosBot = (
            H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        )
        PAlk = PT * PhosTop / PhosBot
        SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + ST / Ks.KS)
        HSO4 = ST / (1 + Ks.KS / Hfree)
        HF = FT / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

        Slope = ln10 * (
            DIC * Ks.K1 * H * (H ** 2 + Ks.K1 * Ks.K2 + 4 * H * Ks.K2) / Denom / Denom
            + BAlk * H / (Ks.KB + H)
            + OH
            + H
        )
        deltapH = Residual / Slope

        while np.any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx if pHx.size > 1 else pHx.item()


def zero_TA_DIC(h, TA, DIC, BT, K1, K2, KB, KW):
    # Roots: one pos, four neg. Use pos.
    LH = DIC * (KB + h) * (K1 * h ** 2 + 2 * K1 * K2 * h)
    RH = (TA * (KB + h) * h - KB * BT * h - KW * (KB + h) + (KB + h) * h ** 2) * (
        h ** 2 + K1 * h + K1 * K2
    )
    return LH - RH


# 1.1.9
def cCO2(H, DIC, Ks):
    """
    Returns CO2
    """
    return DIC / (1 + Ks.K1 / H + Ks.K1 * Ks.K2 / H ** 2)


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
    return DIC / (1 + H / Ks.K2 + H ** 2 / (Ks.K1 * Ks.K2))


# 1.5.80
def cTA(H, DIC, BT, PT, SiT, ST, FT, Ks, mode="multi"):
    """
    Calculate Alkalinity. H is on Total scale.

    Returns
    -------
    If mode == 'multi' returns TA, CAlk, PAlk, SiAlk, OH
    else: returns TA
    """
    # negative
    Denom = H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2
    CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
    PhosBot = H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = PT * PhosTop / PhosBot
    SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
    # positive
    Hfree = H / (1 + ST / Ks.KS)
    HSO4 = ST / (1 + Ks.KS / Hfree)
    HF = FT / (1 + Ks.KF / Hfree)

    TA = CAlk + BAlk + OH + PAlk + SiAlk - Hfree - HSO4 - HF

    if mode == "multi":
        return TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF
    else:
        return TA

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


def pCO2_to_fCO2(pCO2, Tc):
    """
    Calculate fCO2 from pCO2

    Taken from matlab CO2SYS.

    This assumes that the pressure is at one atmosphere, or close to it.
    Otherwise, the Pres term in the exponent affects the results.
    Weiss, R. F., Marine Chemistry 2:203-215, 1974.

    For a mixture of CO2 and air at 1 atm (at low CO2 concentrations)
    Delta and B in cm3/mol
    """
    Tk = Tc + 273.15
    P = 1.01325  # in bar
    RT = 83.1451 * Tk

    a0, a1, a2, a3 = (-1636.75, 12.0408, -3.27957e-2, 3.16528e-05)
    b0, b1 = (57.7, -0.118)

    B = a0 + a1 * Tk + a2 * Tk ** 2 + a3 * Tk ** 3
    delta = b0 + b1 * Tk

    return pCO2 * np.exp(P * (B + 2 * delta) / RT)


def fCO2_to_pCO2(fCO2, Tc):
    """
    Calculate pCO2 from fCO2

    Taken from matlab CO2SYS.

    This assumes that the pressure is at one atmosphere, or close to it.
    Otherwise, the Pres term in the exponent affects the results.
    Weiss, R. F., Marine Chemistry 2:203-215, 1974.

    For a mixture of CO2 and air at 1 atm (at low CO2 concentrations)
    Delta and B in cm3/mol
    """
    Tk = Tc + 273.15
    P = 1.01325  # in bar
    RT = 83.1451 * Tk

    a0, a1, a2, a3 = (-1636.75, 12.0408, -3.27957e-2, 3.16528e-05)
    b0, b1 = (57.7, -0.118)

    B = a0 + a1 * Tk + a2 * Tk ** 2 + a3 * Tk ** 3
    delta = b0 + b1 * Tk

    return fCO2 / np.exp(P * (B + 2 * delta) / RT)


def calc_C_species(
    pHtot=None,
    DIC=None,
    CO2=None,
    HCO3=None,
    CO3=None,
    TA=None,
    fCO2=None,
    pCO2=None,
    T_in=None,
    S_in=None,
    BT=None,
    PT=0,
    SiT=0,
    ST=0,
    FT=0,
    Ks=None,
    **kwargs
):
    """
    Calculate all carbon species from minimal input.
    """

    # if fCO2 is given but CO2 is not, calculate CO2
    if CO2 is None:
        if fCO2 is not None:
            CO2 = fCO2_to_CO2(fCO2, Ks)
        elif pCO2 is not None:
            CO2 = fCO2_to_CO2(pCO2_to_fCO2(pCO2, T_in), Ks)

    # Carbon System Calculations (logic from Zeebe & Wolf-Gladrow, Appendix B)
    # 1. CO2 and pH
    if CO2 is not None and pHtot is not None:
        H = 10.0**-pHtot
        DIC = CO2_pH(CO2, pHtot, Ks)
    # 2. CO2 and HCO3
    elif CO2 is not None and HCO3 is not None:
        H = CO2_HCO3(CO2, HCO3, Ks)
        DIC = CO2_pH(CO2, -unp.log10(H), Ks)
    # 3. CO2 and CO3
    elif CO2 is not None and CO3 is not None:
        H = CO2_CO3(CO2, CO3, Ks)
        DIC = CO2_pH(CO2, -unp.log10(H), Ks)
    # 4. CO2 and TA
    elif CO2 is not None and TA is not None:
        # unit conversion because OH and H wrapped
        # up in TA fns - all need to be in same units.
        pHtot = CO2_TA(CO2=CO2, TA=TA, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
        H = 10.0**-pHtot
        DIC = CO2_pH(CO2, pHtot, Ks)
    # 5. CO2 and DIC
    elif CO2 is not None and DIC is not None:
        H = CO2_DIC(CO2, DIC, Ks)
    # 6. pHtot and HCO3
    elif pHtot is not None and HCO3 is not None:
        H = 10.0**-pHtot
        DIC = pH_HCO3(pHtot, HCO3, Ks)
    # 7. pHtot and CO3
    elif pHtot is not None and CO3 is not None:
        H = 10.0**-pHtot
        DIC = pH_CO3(pHtot, CO3, Ks)
    # 8. pHtot and TA
    elif pHtot is not None and TA is not None:
        H = 10.0**-pHtot
        DIC = pH_TA(pH=pHtot, TA=TA, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    # 9. pHtot and DIC
    elif pHtot is not None and DIC is not None:
        H = 10.0**-pHtot
    # 10. HCO3 and CO3
    elif HCO3 is not None and CO3 is not None:
        H = HCO3_CO3(HCO3, CO3, Ks)
        DIC = pH_CO3(-unp.log10(H), CO3, Ks)
    # 11. HCO3 and TA
    elif HCO3 is not None and TA is not None:
        Warning(
            "Nutrient alkalinity not implemented for this input combination.\nCalculations use only C and B alkalinity."
        )
        H = HCO3_TA(HCO3, TA, BT, Ks)
        DIC = pH_HCO3(-unp.log10(H), HCO3, Ks)
    # 12. HCO3 amd DIC
    elif HCO3 is not None and DIC is not None:
        H = HCO3_DIC(HCO3, DIC, Ks)
    # 13. CO3 and TA
    elif CO3 is not None and TA is not None:
        Warning(
            "Nutrient alkalinity not implemented for this input combination.\nCalculations use only C and B alkalinity."
        )
        H = CO3_TA(CO3, TA, BT, Ks)
        DIC = pH_CO3(-unp.log10(H), CO3, Ks)
    # 14. CO3 and DIC
    elif CO3 is not None and DIC is not None:
        H = CO3_DIC(CO3, DIC, Ks)
    # 15. TA and DIC
    elif TA is not None and DIC is not None:
        pHtot = TA_DIC(TA=TA, DIC=DIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
        H = 10.0**-pHtot

    # The above makes sure that DIC and H are known,
    # this next bit calculates all the missing species
    # from DIC and H.
    if CO2 is None:
        CO2 = cCO2(H, DIC, Ks)
    if fCO2 is None:
        fCO2 = CO2_to_fCO2(CO2, Ks)
    if pCO2 is None:
        pCO2 = fCO2_to_pCO2(fCO2, T_in)
    if HCO3 is None:
        HCO3 = cHCO3(H, DIC, Ks)
    if CO3 is None:
        CO3 = cCO3(H, DIC, Ks)
    # Calculate all elements of Alkalinity
    (TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF) = cTA(
        H=H, DIC=DIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks, mode="multi"
    )

    # if pH not calced yet, calculate on all scales.
    if pHtot is None:
        pHtot = np.array(-unp.log10(H), ndmin=1)
    
    FREEtoTOT = -unp.log10((1 + ST / Ks.KS))
    SWStoTOT = -unp.log10((1 + ST / Ks.KS) / (1 + ST / Ks.KS + FT / Ks.KF))
    fH = calc_fH(T_in + 273.15, S_in)
    
    return Bunch(
        {
            "pHtot": pHtot,
            "pHfree": pHtot - FREEtoTOT,
            "pHsws": pHtot - SWStoTOT,
            "pHNBS": pHtot - SWStoTOT - np.log10(fH),
            "TA": TA,
            "DIC": DIC,
            "CO2": CO2,
            "H": H,
            "HCO3": HCO3,
            "fCO2": fCO2,
            "pCO2": pCO2,
            "CO3": CO3,
            "CAlk": CAlk,
            "BAlk": BAlk,
            "PAlk": PAlk,
            "SiAlk": SiAlk,
            "OH": OH,
            "Hfree": Hfree,
            "HSO4": HSO4,
            "HF": HF,
        }
    )


def calc_revelle_factor(TA, DIC, BT, PT, SiT, ST, FT, Ks):
    """
    Calculate Revelle Factor

    (dpCO2 / dDIC)
    """
    dDIC = 1e-6  # (1 umol kg-1)

    pH = TA_DIC(TA=TA, DIC=DIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2 = cCO2(10.0**-pH, DIC, Ks) / Ks.K0

    # Calculate new fCO2 above and below given value
    pH_hi = TA_DIC(TA=TA, DIC=DIC + dDIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2_hi = cCO2(10.0**-pH_hi, DIC, Ks) / Ks.K0

    pH_lo = TA_DIC(TA=TA, DIC=DIC - dDIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2_lo = cCO2(10.0**-pH_lo, DIC, Ks) / Ks.K0

    return (fCO2_hi - fCO2_lo) * DIC / (fCO2 * 2 * dDIC)
