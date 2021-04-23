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

    a0, a1, a2, a3, a4, a5, a6 = (
        -4576.752,
        115.54,
        -18.453,
        -106.736,
        0.69171,
        -0.65643,
        -0.01844,
    )
    b0, b1, b2, b3, b4, b5, b6 = (
        -8814.715,
        172.1033,
        -27.927,
        -160.340,
        1.3566,
        0.37335,
        -0.05778,
    )
    c0, c1, c3, c4, c5, c6 = (-3070.75, -18.126, 17.27039, 2.81197, -44.99486, -0.09984)

    KP1 = np.exp(
        a0 / TempK
        + a1
        + a2 * lnTempK
        + (a3 / TempK + a4) * Sal ** 0.5
        + (a5 / TempK + a6) * Sal
    )
    KP2 = np.exp(
        b0 / TempK
        + b1
        + b2 * lnTempK
        + (b3 / TempK + b4) * Sal ** 0.5
        + (b5 / TempK + b6) * Sal
    )
    KP3 = np.exp(
        c0 / TempK + c1 + (c3 / TempK + c4) * Sal ** 0.5 + (c5 / TempK + c6) * Sal
    )

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if P is not None:
        ppar = {
            "KP1": [-14.51, 0.1211, -0.000321, -2.67, 0.0427],
            "KP2": [-23.12, 0.1758, -2.647e-3, -5.15, 0.09],
            "KP3": [-26.57, 0.2020, -3.042e-3, -4.08, 0.0714],
        }
        KP1 *= prescorr(P, TempC, *ppar["KP1"])
        KP2 *= prescorr(P, TempC, *ppar["KP2"])
        KP3 *= prescorr(P, TempC, *ppar["KP3"])

    return {"KP1": KP1, "KP2": KP2, "KP3": KP3}


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

    a, b, c, d, e, f, g, h, i = (
        -8904.2,
        117.4,
        -19.334,
        -458.79,
        3.5913,
        188.74,
        -1.5998,
        -12.1652,
        0.07871,
    )

    KSi = np.exp(
        a / TempK
        + b
        + c * np.log(TempK)
        + (d / TempK + e) * Istr ** 0.5
        + (f / TempK + g) * Istr
        + (h / TempK + i) * Istr ** 2
    ) * (1 - 0.001005 * Sal)

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if P is not None:
        ppar = {"KSi": [-29.48, 0.1622, -2.608e-3, -2.84, 0]}
        KSi *= prescorr(P, TempC, *ppar["KSi"])
        # Note: Same as Boric acid constants

    return {"KSi": KSi}


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

    (
        a,
        b,
        c,
    ) = (1590.2, -12.641, 1.525)

    KF = np.exp(a / TempK + b + c * Istr ** 0.5) * (1 - 0.001005 * Sal)

    # Chapter 5, Section 7.2.4 of Dickson, Sabine and Christian
    # (2007, http://cdiac.ornl.gov/oceans/Handbook_2007.html)
    # KF = np.exp(874. / TempK +
    #             -9.68 +
    #             0.111 * Sal**0.5)

    # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
    # Checked against CO2SYS
    if P is not None:
        ppar = {"KF": [-9.78, -0.0090, -0.942e-3, -3.91, 0.054]}
        KF *= prescorr(P, TempC, *ppar["KF"])

    return {"KF": KF}
