import numpy as np

def type_K1K2(p, TK, lnTK, S):
    """Calculate K1 or K2 from given parameters

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : arra-ylike
        natural log of temperature in kelvin
    S : arry-like
        Salinity

    Returns
    -------
    array-like
        K1 or K2 on XXXXX pH scale.
    """
    
    return np.power(10, 
        p[0] +
        p[1] / TK +
        p[2] * lnTK +
        p[3] * S +
        p[4] * S * S
    )
    
def type_KW(p, TK, lnTK, S, sqrtS):
    """Calculate KW from given parameters.

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : arra-ylike
        natural log of temperature in kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
        
    Returns
    -------
    array-like
        KW on XXXXX pH scale.
    """
    return np.exp(
        p[0] +
        p[1] / TK +
        p[2] * lnTK +
        + (p[3] / TK + p[4] + p[5] * lnTK) * sqrtS +
        p[6] * S
    )
    
def type_KB(p, TK, lnTK, S, sqrtS):
    """Calculate KB from given parameters.

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : arra-ylike
        natural log of temperature in kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
        
    Returns
    -------
    array-like
        KB on XXXXX pH scale.
    """
    return np.exp(
        (
            p[0] +
            p[1] * sqrtS +
            p[2] * S +
            p[3] * S * sqrtS +
            p[4] * S * S
        )
        / TK +
        (p[5] + p[6] * sqrtS + p[7] * S) +
        (p[8] + p[9] * sqrtS + p[10] * S) * lnTK +
        p[11] * sqrtS * TK
    )
    
def type_K0(p, TK, S):
    """Calculate K0 from given parameters.

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    S : arry-like
        Salinity
            
    Returns
    -------
    array-like
        K0 on XXXXX pH scale.
    """
    return np.exp(
        p[0] +
        p[1] / TK +
        p[2] * np.log(TK / 100) +
        S * (p[3] - p[4] * TK + p[5] * TK * TK)
    )

def type_KHSO4(p, TK, lnTK, S):
    Istr = (
        19.924 * S / (1000 - 1.005 * S)
    )
    # Ionic strength after Dickson 1990a; see Dickson et al 2007
    
    return np.exp(
        p[0]
        + p[1] / TK
        + p[2] * lnTK
        + np.sqrt(Istr)
        * (p[3] + p[4] / TK + p[5] * lnTK)
        + Istr
        * (p[6] + p[7] / TK + p[8] * lnTK)
        + p[9] / TK * Istr * np.sqrt(Istr)
        + p[10] / TK * Istr ** 2
        + np.log(1 - 0.001005 * S)
    )
    
def type_Ksp(p, TK, S, sqrtS):
    """Calculate Ksp from given parameters

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
    """

K_params = {
    'K0': {
        'type': 'K0',
        'p': [-60.2409, 9345.17, 23.3585, 0.023517, -0.023656E-2, 0.0047036E-4]
    },
    'K1': {
        'type': 'K1K2',
        'p': [-3633.86, 61.2172, -9.67770, 0.011555, -0.0001152]
    },
    'K2': {
        'type': 'K1K2',
        'p': [-471.78, -25.9290, 3.16967, 0.01781, -0.0001122]
    },
    'KW': {
        'type': 'KW',
        'p': [148.9652, -13847.26, -23.6521, 118.67, -5.977, 1.0495, -0.01615]
    },
    'KB': {
        'type': 'KB',
        'p': [-8966.90, -2890.53, -77.942, 1.728, -0.0996, 148.0248, 137.1942, 1.62142, -24.4344, -25.085, -0.2474, 0.053105]
    },
    'KHSO4': {
        'type': 'KHSO4',
        'p': [ 141.328, -4276.1, -23.093, 324.57, -13856, -47.986, -771.54, 35474, 114.723, -2698, 1776 ]
    }
}