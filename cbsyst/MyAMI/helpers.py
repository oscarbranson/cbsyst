"""
Helper functions used throughout MyAMI.
"""
import json
import numpy as np
import pkg_resources as pkgrs

def expand_dims(orig, target):
    """
    Adds additional dimensions to orig so it can be broadcast on target.
    """
    return np.expand_dims(orig, tuple(range(orig.ndim, target.ndim + orig.ndim)))
    
    # on = orig.copy()
    # while on.ndim < (target.ndim + orig.ndim):
    #     on = np.expand_dims(on, -1)
    # return on

def match_dims(orig, target):
    """
    Adds additional dimensions to orig to match the number of dimensions in target.
    """
    return np.expand_dims(orig, tuple(range(orig.ndim, target.ndim)))

def load_params(param_file, asarrays=True):
    """
    Load parameters from a json file and convert the entries to np.ndarray.

    Parameters
    ----------
    param_file : string
        The name of the file within MyAMI/resources to load.
    asarrays : bool, optional
        If true, entries are converted to np.ndarray, by default True

    Returns
    -------
    dict
        The contents of the json file as a dict.
    """
    with open(pkgrs.resource_filename('cbsyst', f'MyAMI/parameters/{param_file}'), 'r') as f:
        params = json.load(f)
    
    if asarrays:
        return {k: np.array(v) for k, v in params.items()}
    return params

def calc_Istr(Sal):
    """
    Calculation ionic strength from Salinity
    """
    return 19.924 * Sal / (1000 - 1.005 * Sal)

def standard_seawater(S=35.):
    """
    Return modern seawater ionic composition at specified salinity
    in units of mol/kg.

    Parameters
    ----------
    S : array-like
        Salinity in PSU

    Returns
    -------
    tuple of arrays
        Containing (cations, anions) in the order:
        cations = [H, Na, K, Mg, Ca, Sr]
        anions = [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4] 
    """

    cation_concs = np.array([
        0.00000001,  # H ion; pH of about 8
        0.4689674,  # Na Millero et al., 2008; Dickson OA-guide
        0.0102077,  # K Millero et al., 2008; Dickson OA-guide
        0.0528171,  # Mg Millero et al., 2008; Dickson OA-guide
        0.0102821,  # Ca Millero et al., 2008; Dickson OA-guide
        0.0000907  # Sr Millero et al., 2008; Dickson OA-guide
    ]) * S / 35.

    anion_concs = np.array([
        0.0000010,  # OH ion; pH of about 8
        0.5458696,  # Cl Millero et al., 2008; Dickson OA-guide
        0.0001008,  # BOH4 Millero et al., 2008; Dickson OA-guide; pH of about 8 -- borate,
        0.0017177,  # HCO3 Millero et al., 2008; Dickson OA-guide
        0.0282352 * 1e-6,  # HSO4 Millero et al., 2008; Dickson OA-guide
        0.0002390,  # CO3 Millero et al., 2008; Dickson OA-guide
        0.0282352  # SO4 Millero et al., 2008; Dickson OA-guide
    ]) * S / 35.

    return cation_concs, anion_concs
