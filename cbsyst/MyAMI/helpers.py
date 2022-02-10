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