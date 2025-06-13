"""
Utility functions for handling uncertainties while preserving data types.

The main issue with uncertainties.unumpy functions is that they always return
object dtype arrays, even when the input doesn't contain uncertainty objects.
This breaks downstream numpy operations that expect numeric dtypes.
"""

import numpy as np
import uncertainties.unumpy as unp
import math


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
    if value is None:
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