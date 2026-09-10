import uncertainties.unumpy as unp
import numpy as np
import pandas as pd


# Helpers useful to the user
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

    cols = [
        "pHtot",
        "pHfree",
        "pHsws",
        "pHNBS",
        "DIC",
        "fCO2",
        "pCO2",
        "CO2",
        "HCO3",
        "CO3",
        "TA",
        "BT",
        "BO3",
        "BO4",
        "dBT",
        "dBO3",
        "dBO4",
        "ABT",
        "ABO3",
        "ABO4",
        "T_in",
        "S_in",
        "P_in",
        "T_out",
        "S_out",
        "P_out",
        "Ca",
        "Mg",
    ]

    consts = ["K0", "K1", "K2", "KB", "KW", "KS", "KspA", "KspC"]

    size = cbdat.pHtot.size
    out = pd.DataFrame(index=range(size))

    for c in cols:
        if c in cbdat and cbdat[c] is not None:
            if (np.ndim(cbdat[c]) == 1) and (cbdat[c].size == 1):
                cbdat[c] = cbdat[c][0]
            if c in cbdat:
                out.loc[:, c] = cbdat[c]

    if include_constants:
        for c in consts:
            if c in cbdat.Ks and cbdat.Ks[c] is not None:
                if (np.ndim(cbdat.Ks[c]) == 1) and (cbdat.Ks[c].size == 1):
                    cbdat.Ks[c] = cbdat.Ks[c][0]
                out.loc[:, "p" + c] = -np.log10(cbdat.Ks[c])
        if "alphaB" in cbdat and cbdat.alphaB is not None:
            if (np.ndim(cbdat.alphaB) == 1) and (cbdat.alphaB.size == 1):
                cbdat.alphaB = cbdat.alphaB[0]
            out.loc[:, "alphaB"] = cbdat.alphaB

    if path is not None:
        fmt = path.split(".")[-1]
        fdict = {
            "csv": "to_csv",
            "html": "to_html",
            "xls": "to_excel",
            "pkl": "to_pickle",
            "tex": "to_latex",
        }

        if fmt not in fdict:
            raise ValueError(
                (
                    "File extension does not match available output\n"
                    + "options. Should be one of 'csv', 'html', 'xls',\n"
                    + "'pkl' (pickle) or 'tex' (LaTeX)."
                )
            )
        try:
            _ = getattr(out, fdict[fmt])(path, index=None)
        except TypeError:
            _ = getattr(out, fdict[fmt])(path)

    return out
   
def maxSize(*it):
    """
    Calculate maximum size of provided items.

    Parameters
    ----------
    *it : objects
        Items of various sizes. Only sizes
        of iterables are returned.

    Returns
    -------
    size of largest object (int).
    """
    m = set()
    for i in it:
        try:
            m.add(np.size(i))
        except TypeError:
            pass
    if np.size(m) > 0:
        return max(m)
    else:
        return 1

def maxShape(*it):
    """
    Returns the shape of the largest array.
    """
    size = 0
    shape = None
    for i in it:
        i = np.asanyarray(i)
        if i.size > size:
            size = i.size
            shape = i.shape
    return shape

def cast_array(*it):
    """
    Recasts inputs into array of shape (len(it), maxSize(*it))
    """
    max_size = maxSize(*it)
    new = np.empty((len(it), max_size), dtype=object)
    for i, t in enumerate(it):
        raveled = np.ravel(t)
        if raveled.size == 1:
            # If single value, broadcast to all positions
            new[i, :] = raveled[0]
        else:
            # If array, assign directly
            new[i, :raveled.size] = raveled
            if raveled.size < max_size:
                # Fill remaining positions with last value
                new[i, raveled.size:] = raveled[-1]
    return new

def isnone(x):
    """
    True if x is None, or an object array containing only None.

    np.array(None, dtype=object) is not None, so plain `is None` checks
    treat it as a supplied value. This catches both forms.
    """
    if x is None:
        return True
    if isinstance(x, np.ndarray) and x.dtype == object:
        return all(v is None for v in x.flat)  # True for empty arrays too
    return False