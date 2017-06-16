import uncertainties.unumpy as unp
import numpy as np
import pandas as pd


# Helpers udeful to the user
# --------------------------
def data_out(cbdat, path=None, include_constants=False):
    """
    Save output from cbsyst.

    Parameters
    ----------
    cbdat : dict / Bunch
        The output from Csys, Bsys, ABsys or CBsys.
    path : str
        The file name (and path) where you want to
        save the data. If not provided, data are not
        saved to a file.

        The extension of the file determines the output
        format. Can be 'csv', 'xls', 'html, 'tex', or 'pkl'.
    include_constants : bool
        If True, include pK and alpha constants in output.

    Returns
    -------
    * pandas.DataFrame of output
    * Saves file (if specified)

    """

    cols = ['pH', 'DIC', 'fCO2', 'pCO2', 'CO2', 'HCO3', 'CO3',
            'TA', 'BT', 'BO3', 'BO4',
            'dBT', 'dBO3', 'dBO4', 'ABT', 'ABO3', 'ABO4',
            'T', 'S', 'P', 'Ca', 'Mg']

    consts = ['K0', 'K1', 'K2', 'KB', 'KW', 'KSO4',
              'KspA', 'KspC']

    size = cbdat.pH.size
    out = pd.DataFrame(index=range(size))

    for c in cols:
        if c in cbdat and cbdat[c] is not None:
            if (np.ndim(cbdat[c]) == 1) & (cbdat[c].size == 1):
                cbdat[c] = cbdat[c][0]
            if c in cbdat:
                out.loc[:, c] = cbdat[c]

    if include_constants:
        for c in consts:
            if c in cbdat.Ks and cbdat.Ks[c] is not None:
                if (np.ndim(cbdat.Ks[c]) == 1) & (cbdat.Ks[c].size == 1):
                    cbdat.Ks[c] = cbdat.Ks[c][0]
                out.loc[:, 'p' + c] = -np.log10(cbdat.Ks[c])
        if 'alphaB' in cbdat and cbdat.alphaB is not None:
            if (np.ndim(cbdat.alphaB) == 1) & (cbdat.alphaB.size == 1):
                cbdat.alphaB = cbdat.alphaB[0]
            out.loc[:, 'alphaB'] = cbdat.alphaB

    if path is not None:
        fmt = path.split('.')[-1]
        fdict = {'csv': 'to_csv',
                 'html': 'to_html',
                 'xls': 'to_excel',
                 'pkl': 'to_pickle',
                 'tex': 'to_latex'}

        if fmt not in fdict:
            raise ValueError(('File extension does not match available output\n' +
                              "options. Should be one of 'csv', 'html', 'xls',\n" +
                              "'pkl' (pickle) or 'tex' (LaTeX)."))
        try:
            _ = getattr(out, fdict[fmt])(path, index=None)
        except TypeError:
            _ = getattr(out, fdict[fmt])(path)

    return out


# Programmatic helpers for code elsewhere
# ---------------------------------------

# Bunch modifies dict to allow item access using dot (.) operator
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


def noms(*it):
    """
    Return nominal_values for provided objects.

    Parameters
    ----------
    *it : n objects
    """
    return [unp.nominal_values(i) for i in it]


def maxL(*it):
    """
    Calculate maximum length of provided items.

    Parameters
    ----------
    *it : objects
        Items of various lengths. Only lengths
        of iterables are returned.

    Returns
    -------
    Length of longest object (int).
    """
    m = set()
    for i in it:
        try:
            m.add(len(i))
        except TypeError:
            pass
    if len(m) > 0:
        return max(m)
    else:
        return 1


def cast_array(*it):
    """
    Recasts inputs into array of shape (len(it), maxL(*it))
    """
    new = np.zeros((len(it), maxL(*it)))
    for i, t in enumerate(it):
        new[i, :] = t
    return new


def NnotNone(*it):
    """
    Returns the number of elements of it tha are not None.

    Parameters
    ----------
    it : iterable
        iterable of elements that are either None, or not None

    Returns
    -------
    int
    """
    return sum([i is not None for i in it])


# pK <--> K converters
def ch(pK):
    """
    Convert pK to K
    """
    return np.power(10., np.multiply(pK, -1.))


def cp(K):
    """
    Convert K to pK
    """
    return -np.log10(K)


# Helpers for aspects of seawater chemistry
# -----------------------------------------
def prescorr(P, Tc, a0, a1, a2, b0, b1):
    """
    Calculate pressore correction factor for thermodynamic Ks.

    From Millero et al (2007, doi:10.1021/cr0503557)
    Eqns 38-40

    Usage:
    K_corr / K_orig = [output]
    Kcorr = [output] * K_orig
    """
    dV = a0 + a1 * Tc + a2 * Tc**2
    dk = (b0 + b1 * Tc) / 1000
    # factor of 1000 not mentioned in Millero,
    # but present in Zeebe book, and used in CO2SYS
    RT = 83.131 * (Tc + 273.15)
    return np.exp((-dV + 0.5 * dk * P) * P / RT)


def swdens(TempC, Sal):
    """
    Seawater Density (kg / L) from Temp (C) and Sal (PSU)

    Chapter 5, Section 4.2 of Dickson, Sabine and Christian
    (2007, http://cdiac.ornl.gov/oceans/Handbook_2007.html)

    Parameters
    ----------
    TempC : array-like
        Temperature in celcius.
    Sal : array-like
        Salinity in PSU

    Returns
    -------
    Density in kg / L
    """
    # convert temperature to IPTS-68
    T68 = (TempC + 0.0002) / 0.99975
    pSMOW = (999.842594 +
             6.793952e-2 * T68 +
             -9.095290e-3 * T68**2 +
             1.001685e-4 * T68**3 +
             -1.120083e-6 * T68**4 +
             6.536332e-9 * T68**5)
    A = (8.24493e-1 +
         -4.0899e-3 * T68 +
         7.6438e-5 * T68**2 +
         -8.2467e-7 * T68**3 +
         5.3875e-9 * T68**4)
    B = (-5.72466e-3 +
         1.0227e-4 * T68 +
         -1.6546e-6 * T68**2)
    C = 4.8314e-4
    return (pSMOW + A * Sal + B * Sal**1.5 + C * Sal**2) / 1000


def calc_TS(Sal):
    """
    Calculate total Sulphur

    Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
    this is .02824.*Sali./35. = .0008067.*Sali
    """
    return (0.14 / 96.062) * (Sal / 1.80655)  # mol/kg-SW


def calc_TF(Sal):
    """
    Calculate total Fluorine

    Riley, J. P., Deep-Sea Research 12:219-220, 1965:
    this is .000068.*Sali./35. = .00000195.*Sali
    """
    return (0.000067 / 18.998) * (Sal / 1.80655)  # mol/kg-SW