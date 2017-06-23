import numpy as np
from cbsyst.helpers import prescorr


# Non-MyAMI Equilibrium Coefficients
# ----------------------------------
def calc_KPs(TempC, Sal, P=None):
    """
    Calculate equilibrium constants for P species.

    KP1 = H3PO4
    KP2 = H2PO4
    KP3 = HPO4

    Chapter 5, Section 7.2.5 of Dickson, Sabine and Christian
    (2007, http://cdiac.ornl.gov/oceans/Handbook_2007.html)

    **WITHOUT APPROX PH SCALE CONVERSION IN CONSTANT**
    (See footnote 5 in 'Best Practices' Guide)
    This produces constants on SWS pH Scale.
    Must be converted to Total scale before use.

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius.
    Sal : array-like
        Salinity in PSU
    P : array-like
        Pressure in bar

    Returns
    -------
    dict of KPs
    """
    TempK = TempC + 273.15
    lnTempK = np.log(TempK)
    KP1 = np.exp(-4576.752 / TempK +
                 115.54 +
                 -18.453 * lnTempK +
                 (-106.736 / TempK + 0.69171) * Sal**0.5 +
                 (-0.65643 / TempK - 0.01844) * Sal)
    KP2 = np.exp(-8814.715 / TempK +
                 172.1033 +
                 -27.927 * lnTempK +
                 (-160.340 / TempK + 1.3566) * Sal**0.5 +
                 (0.37335 / TempK - 0.05778) * Sal)
    KP3 = np.exp(-3070.75 / TempK +
                 -18.126 +
                 (17.27039 / TempK + 2.81197) * Sal**0.5 +
                 (-44.99486 / TempK - 0.09984) * Sal)

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if (P is not None):
        KP1 *= prescorr(P, TempC,
                        *[-14.51, 0.1211, -0.000321, -2.67, 0.0427])
        KP2 *= prescorr(P, TempC,
                        *[-23.12, 0.1758, -2.647e-3, -5.15, 0.09])
        KP3 *= prescorr(P, TempC,
                        *[-26.57, 0.2020, -3.042e-3, -4.08, 0.0714])

    return {'KP1': KP1,
            'KP2': KP2,
            'KP3': KP3}


def calc_KSi(TempC, Sal, P=None):
    """
    Calculate equilibrium constants for Si species.

    Chapter 5, Section 7.2.6 of Dickson, Sabine and Christian
    (2007, http://cdiac.ornl.gov/oceans/Handbook_2007.html)

    **WITHOUT APPROX PH SCALE CONVERSION IN CONSTANT**
    (See footnote 5 in 'Best Practices' Guide)
    This produces constants on SWS pH Scale.
    Must be converted to Total scale before use.

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius.
    Sal : array-like
        Salinity in PSU
    P : array-like
        Pressure in bar

    Returns
    -------
    dict containing KSi
    """
    TempK = TempC + 273.15
    Istr = 19.924 * Sal / (1000 - 1.005 * Sal)

    KSi = np.exp(-8904.2 / TempK +
                 117.4 +
                 -19.334 * np.log(TempK) +
                 (-458.79 / TempK + 3.5913) * Istr**0.5 +
                 (188.74 / TempK - 1.5998) * Istr +
                 (-12.1652 / TempK + 0.07871) * Istr**2) * (1 - 0.001005 * Sal)

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if (P is not None):
        KSi *= prescorr(P, TempC,
                        *[-29.48, 0.1622, -2.608e-3, -2.84, 0])
        # Note: Same as Boric acid constants

    return {'KSi': KSi}


def calc_KF(TempC, Sal, P=None):
    """
    Calculate equilibrium constants for HF.

    Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979

    This produces constants on SWS pH Scale.
    Must be converted to Total scale before use.

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius.
    Sal : array-like
        Salinity in PSU
    P : array-like
        Pressure in bar

    Returns
    -------
    dict of KF
    """
    TempK = TempC + 273.15
    Istr = 19.924 * Sal / (1000 - 1.005 * Sal)
    # Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979
    KF = np.exp(1590.2 / TempK - 12.641 + 1.525 * Istr**0.5) * (1 - 0.001005 * Sal)

    # Chapter 5, Section 7.2.4 of Dickson, Sabine and Christian
    # (2007, http://cdiac.ornl.gov/oceans/Handbook_2007.html)
    # KF = np.exp(874. / TempK +
    #             -9.68 +
    #             0.111 * Sal**0.5)

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if (P is not None):
        KF *= prescorr(P, TempC,
                       *[-9.78, -0.0090, -0.942e-3, -3.91, 0.054])

    return {'KF': KF}
