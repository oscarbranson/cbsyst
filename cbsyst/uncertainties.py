"""
Utility functions for handling uncertainties while preserving data types.

The main issue with uncertainties.unumpy functions is that they always return
object dtype arrays, even when the input doesn't contain uncertainty objects.
This breaks downstream numpy operations that expect numeric dtypes.
"""

import numpy as np
import uncertainties
import uncertainties.unumpy as unp
from .helpers import isnone

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
        try:
            if len(obj) > 0 and hasattr(obj.flat[0], 'nominal_value'):
                return True
        except (TypeError, AttributeError):
            pass
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
        try:
            if len(obj) > 0 and hasattr(obj.flat[0], 'nominal_value'):
                return np.array([item.nominal_value for item in obj])
        except (TypeError, AttributeError):
            pass
    # Regular object without uncertainties
    return obj



def log10_preserve_type(value):
    """
    Calculate log10 while preserving data types.
    
    Uses uncertainties.unumpy.log10() only when uncertainty objects are present,
    otherwise uses numpy.log10() to preserve numeric dtypes.
    
    Parameters
    ----------
    value : array-like
        Input value(s)
        
    Returns
    -------
    array-like
        log10(value) with preserved dtype when possible
    """
    if _has_uncertainties(value):
        return unp.log10(value)
    else:
        return np.log10(value)


def log_preserve_type(value):
    """
    Calculate natural log while preserving data types.
    
    Uses uncertainties.unumpy.log() only when uncertainty objects are present,
    otherwise uses numpy.log() to preserve numeric dtypes.
    
    Parameters
    ----------
    value : array-like
        Input value(s)
        
    Returns
    -------
    array-like
        log(value) with preserved dtype when possible
    """
    if _has_uncertainties(value):
        return unp.log(value)
    else:
        return np.log(value)


def negative_log10_preserve_type(value):
    """
    Calculate -log10 while preserving data types.
    
    This is commonly used for pH calculations.
    
    Parameters
    ----------
    value : array-like
        Input value(s)
        
    Returns
    -------
    array-like
        -log10(value) with preserved dtype when possible
    """
    if _has_uncertainties(value):
        return -unp.log10(value)
    else:
        return -np.log10(value)

def sqrt_preserve_type(value):
    """
    Calculate square root while preserving data types.
    
    Uses uncertainties.unumpy.sqrt() only when uncertainty objects are present,
    otherwise uses numpy.sqrt() to preserve numeric dtypes.
    
    Parameters
    ----------
    value : array-like
        Input value(s)
        
    Returns
    -------
    array-like
        sqrt(value) with preserved dtype when possible
    """
    if _has_uncertainties(value):
        return unp.sqrt(value)
    else:
        return np.sqrt(value)

def remove_negatives(value):
    """
    Remove negative values from data while preserving uncertainty structure.
    
    For uncertainty objects, checks nominal values and sets negatives to NaN.
    For regular arrays/values, sets negatives to NaN directly.
    
    Parameters
    ----------
    value : array-like, ufloat, or uarray
        Input value(s) to process
        
    Returns
    -------
    array-like, ufloat, or uarray
        Value(s) with negatives replaced by NaN
    """
    if isnone(value):
        return value
        
    if _has_uncertainties(value):
        # For uncertainty objects, check nominal values and replace negatives with NaN
        nominal_values = _extract_nominal_values(value)
        if np.any(nominal_values < 0):
            return np.where(nominal_values < 0, np.nan, value)
        return value
    elif isinstance(value, np.ndarray):
        # Handle regular numpy array
        if np.any(value < 0):
            result = value.copy()
            result[result < 0] = np.nan
            return result
        return value
    elif hasattr(value, '__len__') and hasattr(value, 'loc'):  # pandas Series
        if np.any(value < 0):
            result = value.copy()
            result[result < 0] = np.nan
            return result
        return value
    elif value < 0:
        return np.nan
    else:
        return value
    
def _calculate_finite_difference_derivative(func, args_nominal, kwargs_nominal, param_index, element_index=None, is_kwarg=False, key=None, epsilon=1e-8):
    """
    Calculate finite difference derivative for a single parameter.
    
    Parameters:
    -----------
    func : callable
        Function to calculate derivative for
    args_nominal : list
        Nominal values of positional arguments
    kwargs_nominal : dict
        Nominal values of keyword arguments
    param_index : int
        Index of parameter to perturb
    element_index : int, optional
        Index within array parameter (for uarray elements)
    is_kwarg : bool
        Whether parameter is a keyword argument
    key : str, optional
        Key name for keyword argument
    epsilon : float
        Relative perturbation size
    
    Returns:
    --------
    float or array
        Finite difference derivative
    """
    if is_kwarg:
        # Keyword argument
        if isnone(element_index):
            # Scalar kwarg
            delta = abs(kwargs_nominal[key] * epsilon) if kwargs_nominal[key] != 0 else epsilon
            
            kwargs_plus = kwargs_nominal.copy()
            kwargs_minus = kwargs_nominal.copy()
            kwargs_plus[key] = kwargs_nominal[key] + delta
            kwargs_minus[key] = kwargs_nominal[key] - delta
            
            result_plus = func(*args_nominal, **kwargs_plus)
            result_minus = func(*args_nominal, **kwargs_minus)
        else:
            # Array kwarg element
            delta = abs(kwargs_nominal[key][element_index] * epsilon) if kwargs_nominal[key][element_index] != 0 else epsilon
            
            kwargs_plus = kwargs_nominal.copy()
            kwargs_minus = kwargs_nominal.copy()
            
            val_plus = kwargs_nominal[key].copy()
            val_minus = kwargs_nominal[key].copy()
            val_plus[element_index] = kwargs_nominal[key][element_index] + delta
            val_minus[element_index] = kwargs_nominal[key][element_index] - delta
            
            kwargs_plus[key] = val_plus
            kwargs_minus[key] = val_minus
            
            result_plus = func(*args_nominal, **kwargs_plus)
            result_minus = func(*args_nominal, **kwargs_minus)
    else:
        # Positional argument
        if isnone(element_index):
            # Scalar arg
            delta = abs(args_nominal[param_index] * epsilon) if args_nominal[param_index] != 0 else epsilon
            
            args_plus = list(args_nominal)
            args_minus = list(args_nominal)
            args_plus[param_index] = args_nominal[param_index] + delta
            args_minus[param_index] = args_nominal[param_index] - delta
            
            result_plus = func(*args_plus, **kwargs_nominal)
            result_minus = func(*args_minus, **kwargs_nominal)
        else:
            # Array arg element
            delta = abs(args_nominal[param_index][element_index] * epsilon) if args_nominal[param_index][element_index] != 0 else epsilon
            
            args_plus = list(args_nominal)
            args_minus = list(args_nominal)
            
            arg_plus = args_nominal[param_index].copy()
            arg_minus = args_nominal[param_index].copy()
            arg_plus[element_index] = args_nominal[param_index][element_index] + delta
            arg_minus[element_index] = args_nominal[param_index][element_index] - delta
            
            args_plus[param_index] = arg_plus
            args_minus[param_index] = arg_minus
            
            result_plus = func(*args_plus, **kwargs_nominal)
            result_minus = func(*args_minus, **kwargs_nominal)
    
    return (result_plus - result_minus) / (2 * delta)


def _calculate_derivatives_for_parameters(func, args, kwargs, args_nominal, kwargs_nominal, epsilon=1e-8):
    """
    Calculate finite difference derivatives for all uncertain parameters.
    
    Returns:
    --------
    list
        List of (param/element, derivative, [additional_info]) tuples
    """
    derivatives = []
    
    # Calculate derivatives for positional arguments
    for i, arg in enumerate(args):
        if _has_uncertainties(arg):
            if hasattr(arg, 'nominal_value'):
                # Individual ufloat object
                derivative = _calculate_finite_difference_derivative(
                    func, args_nominal, kwargs_nominal, i, epsilon=epsilon
                )
                derivatives.append((arg, derivative))
                
            elif hasattr(arg, '__iter__') and hasattr(arg, 'dtype') and arg.dtype == object:
                # uarray object - calculate derivatives for each element
                for j, element in enumerate(arg):
                    if hasattr(element, 'nominal_value'):
                        derivative = _calculate_finite_difference_derivative(
                            func, args_nominal, kwargs_nominal, i, element_index=j, epsilon=epsilon
                        )
                        derivatives.append((element, derivative, i, j))
    
    # Calculate derivatives for keyword arguments
    for key, val in kwargs.items():
        if _has_uncertainties(val):
            if hasattr(val, 'nominal_value'):
                # Individual ufloat object
                derivative = _calculate_finite_difference_derivative(
                    func, args_nominal, kwargs_nominal, None, is_kwarg=True, key=key, epsilon=epsilon
                )
                derivatives.append((val, derivative))
                
            elif hasattr(val, '__iter__') and hasattr(val, 'dtype') and val.dtype == object:
                # uarray object - calculate derivatives for each element
                for j, element in enumerate(val):
                    if hasattr(element, 'nominal_value'):
                        derivative = _calculate_finite_difference_derivative(
                            func, args_nominal, kwargs_nominal, None, element_index=j, 
                            is_kwarg=True, key=key, epsilon=epsilon
                        )
                        derivatives.append((element, derivative, key, j))
    
    return derivatives


def _propagate_uncertainties(derivatives, result_nominal):
    """
    Propagate uncertainties using linear error propagation.
    
    Parameters:
    -----------
    derivatives : list
        List of (param/element, derivative, [additional_info]) tuples
    result_nominal : scalar or array
        Nominal result value(s)
    
    Returns:
    --------
    scalar, ufloat, array, or uarray
        Result with propagated uncertainties
    """
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
        args_nominal = [_extract_nominal_values(arg) for arg in args]
        kwargs_nominal = {key: _extract_nominal_values(val) for key, val in kwargs.items()}
        
        # Calculate nominal result
        result_nominal = func(*args_nominal, **kwargs_nominal)
        
        # Calculate derivatives for uncertainty propagation
        derivatives = _calculate_derivatives_for_parameters(func, args, kwargs, args_nominal, kwargs_nominal)
        
        # Propagate uncertainties
        return _propagate_uncertainties(derivatives, result_nominal)
    
    return wrapper

def _zero_finder_finite_difference_derivative(zero_func, params_nominal, param_index, element_index=None, bounds=(10 ** -14, 10 ** -1), epsilon=1e-8):
    """
    Calculate finite difference derivative for zero finder functions.
    """
    import scipy.optimize as opt
    
    if isnone(element_index):
        # Scalar parameter
        delta = abs(params_nominal[param_index] * epsilon) if params_nominal[param_index] != 0 else epsilon
        
        params_plus = params_nominal.copy()
        params_minus = params_nominal.copy()
        params_plus[param_index] = params_nominal[param_index] + delta
        params_minus[param_index] = params_nominal[param_index] - delta
    else:
        # Array parameter element
        delta = abs(params_nominal[param_index][element_index] * epsilon) if params_nominal[param_index][element_index] != 0 else epsilon
        
        params_plus = params_nominal.copy()
        params_minus = params_nominal.copy()
        
        param_plus = params_nominal[param_index].copy()
        param_minus = params_nominal[param_index].copy()
        param_plus[element_index] = params_nominal[param_index][element_index] + delta
        param_minus[element_index] = params_nominal[param_index][element_index] - delta
        
        params_plus[param_index] = param_plus
        params_minus[param_index] = param_minus
    
    # Calculate function values at perturbed points
    try:
        result_plus = opt.brentq(zero_func, *bounds, args=tuple(params_plus), xtol=1e-16)
    except ValueError:
        result_plus = opt.fsolve(zero_func, 1, args=tuple(params_plus))[0]
        
    try:
        result_minus = opt.brentq(zero_func, *bounds, args=tuple(params_minus), xtol=1e-16)
    except ValueError:
        result_minus = opt.fsolve(zero_func, 1, args=tuple(params_minus))[0]
    
    return (result_plus - result_minus) / (2 * delta)


def _zero_finder_with_uncertainties(params, zero_func, bounds=(10 ** -14, 10 ** -1)):
    """
    Zero finder that propagates uncertainties using finite differences.
    This is a cleaner version that uses shared helper functions.
    """
    import scipy.optimize as opt
    
    # Extract nominal values for the zero finder
    params_nominal = [_extract_nominal_values(p) for p in params]
    
    # Find the zero using nominal values
    try:
        result_nominal = opt.brentq(zero_func, *bounds, args=tuple(params_nominal), xtol=1e-16)
    except ValueError:
        result_nominal = opt.fsolve(zero_func, 1, args=tuple(params_nominal))[0]
    
    # Calculate derivatives numerically using finite differences
    derivatives = []
    
    for i, p in enumerate(params):
        if _has_uncertainties(p):
            if hasattr(p, 'nominal_value'):
                # Individual ufloat object
                derivative = _zero_finder_finite_difference_derivative(
                    zero_func, params_nominal, i, bounds=bounds
                )
                derivatives.append((p, derivative))
                
            elif hasattr(p, '__iter__') and hasattr(p, 'dtype') and p.dtype == object:
                # uarray object - calculate derivatives for each element
                for j, element in enumerate(p):
                    if hasattr(element, 'nominal_value'):
                        derivative = _zero_finder_finite_difference_derivative(
                            zero_func, params_nominal, i, element_index=j, bounds=bounds
                        )
                        derivatives.append((element, derivative, i, j))
    
    # For zero-finders, handle array inputs specially since they typically return scalars
    result_shape = None
    for p in params:
        if hasattr(p, '__iter__') and hasattr(p, 'dtype') and p.dtype == object:
            result_shape = p.shape
            break
    
    if not isnone(result_shape):
        # Array inputs - calculate uncertainty (same for all elements since result is scalar)
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
        
        if total_variance > 0:
            # Return array of ufloats matching input shape (all with same uncertainty)
            uncertainty = np.sqrt(total_variance)
            result_array = np.array([uncertainties.ufloat(result_nominal, uncertainty) 
                                   for _ in range(np.prod(result_shape))]).reshape(result_shape)
            if result_array.size == 1:
                return result_array.item()
            return result_array
        else:
            return result_nominal
    else:
        # Scalar inputs - use standard uncertainty propagation
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
        
        # Create result with uncertainty
        if total_variance > 0:
            return uncertainties.ufloat(result_nominal, np.sqrt(total_variance))
        else:
            return result_nominal