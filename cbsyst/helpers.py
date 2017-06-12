import uncertainties.unumpy as unp
import numpy as np


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


# pK <--> K converters
def ch(pK):
    """
    Convert pK to K
    """
    if not isinstance(pK, np.ndarray):
        pK = np.array(pK)
    return 10**-pK


def cp(K):
    """
    Convert K to pK
    """
    return -np.log10(K)
