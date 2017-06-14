import uncertainties.unumpy as unp
import numpy as np
import pandas as pd


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
