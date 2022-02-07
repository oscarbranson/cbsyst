import numpy as np

def fn_K1K2(p, TK, lnTK, S):
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
    
def fn_KW(p, TK, lnTK, S, sqrtS):
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
    
def fn_KB(p, TK, lnTK, S, sqrtS):
    """Calculate KB from given parameters.

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : array-like
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
    
def fn_K0(p, TK, S):
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

def fn_KHSO4(p, TK, lnTK, S):
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
    
def fn_Ksp(p, TK, S, sqrtS):
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

    return np.power(
        10,
        (
            p[0] + 
            p[1] * TK +
            p[2] / TK +
            p[3] * np.log10(TK) +
            (p[4] + p[5] * TK + p[6] / TK) * sqrtS +
            p[7] * S +
            p[8] * S * sqrtS
        ),
    )

def fn_KP(p, TK, lnTK, S, sqrtS):
    """Calculate KP(s) from given parameters

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : array-like
        natural log of temperature in kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
    """

    return np.exp(
        p[0] / TK
        + p[1]
        + p[2] * lnTK
        + (p[3] / TK + p[4]) * sqrtS
        + (p[5] / TK + p[6]) * S
    )

def fn_KSi(p, TK, lnTK, S, sqrtS):
    """Calculate KSi from given parameters

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : array-like
        natural log of temperature in kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
    """

    Istr = 19.924 * S / (1000 - 1.005 * S)

    return np.exp(
        p[0] / TK + 
        p[1] +
        p[2] * np.log(TK) +
        (p[3] / TK + p[4]) * Istr ** 0.5 +
        (p[5] / TK + p[6]) * Istr +
        (p[7] / TK + p[8]) * Istr ** 2
    ) * (1 - 0.001005 * S)

def fn_KF(p, TK, lnTK, S, sqrtS):
    """Calculate KSi from given parameters

    Parameters
    ----------
    p : array-like
        parameters for K calculation
    TK : array-like
        Temperature in Kelvin
    lnTK : array-like
        natural log of temperature in kelvin
    S : arry-like
        Salinity
    sqrtS : array-like
        square root of salinity
    """

    return np.exp(
        p[0] / TK + 
        p[1] + 
        p[2] * sqrtS
    )
    # Istr = 19.924 * S / (1000 - 1.005 * S)

    # return np.exp(p[0] / TK + p[1] + p[2] * Istr ** 0.5) * (1 - 0.001005 * S)
