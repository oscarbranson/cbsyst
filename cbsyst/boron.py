import numpy as np
from cbsyst.helpers import Bunch
from .uncertainties import negative_log10_preserve_type
from . import pH

def chiB_calc(H, Ks):
    return 1 / (1 + Ks.KB / H)

# B conc fns
def BT_BO3(BT, BO3, Ks):
    """
    Returns H
    """
    return Ks.KB / (BT / BO3 - 1)


def BT_BO4(BT, BO4, Ks):
    """
    Returns H
    """
    return Ks.KB * (BT / BO4 - 1)


def pH_BO3(pH, BO3, Ks):
    """
    Returns BT
    """
    H = 10.0**-pH
    return BO3 * (1 + Ks.KB / H)


def pH_BO4(pH, BO4, Ks):
    """
    Returns BT
    """
    H = 10.0**-pH
    return BO4 * (1 + H / Ks.KB)


def cBO4(BT, H, Ks):
    return BT / (1 + H / Ks.KB)


def cBO3(BT, H, Ks):
    return BT / (1 + Ks.KB / H)

def calc_B_species(pHtot=None, BT=None, BO3=None, BO4=None, Ks=None, **kwargs):
    # B system calculations
    if pHtot is not None and BT is not None:
        H = 10.0**-pHtot
    elif BT is not None and BO3 is not None:
        H = BT_BO3(BT, BO3, Ks)
    elif BT is not None and BO4 is not None:
        H = BT_BO4(BT, BO4, Ks)
    elif BO3 is not None and BO4 is not None:
        BT = BO3 + BO4
        H = BT_BO3(BT, BO3, Ks)
    elif pHtot is not None and BO3 is not None:
        H = 10.0**-pHtot
        BT = pH_BO3(pHtot, BO3, Ks)
    elif pHtot is not None and BO4 is not None:
        H = 10.0**-pHtot
        BT = pH_BO4(pHtot, BO4, Ks)

    # The above makes sure that BT and H are known,
    # this next bit calculates all the missing species
    # from BT and H.

    if BO3 is None:
        BO3 = cBO3(BT, H, Ks)
    if BO4 is None:
        BO4 = cBO4(BT, H, Ks)
    if pHtot is None:
        pHtot = np.array(negative_log10_preserve_type(H), ndmin=1)

    return Bunch({"pHtot": pHtot, "H": H, "BT": BT, "BO3": BO3, "BO4": BO4})

# CBsyst 1.0 functions

def solve_pH_BT(params):
    params.H = 10.0**-params.pHtot

def solve_BT_BO3(params):
    params.H = BT_BO3(params.BT, params.BO3, params.Ks)

def solve_BT_BO4(params):
    params.H = BT_BO4(params.BT, params.BO4, params.Ks)

def solve_BO3_BO4(params):
    params.BT = params.BO3 + params.BO4
    params.H = BT_BO3(params.BT, params.BO3, params.Ks)
    
def solve_pH_BO3(params):
    params.H = 10.0**-params.pHtot
    params.BT = pH_BO3(params.pHtot, params.BO3, params.Ks)


def solve_pH_BO4(params):
    params.H = 10.0**-params.pHtot
    params.BT = pH_BO4(params.pHtot, params.BO4, params.Ks)

SOLVERS = {
    ('pHtot', 'BT'): solve_pH_BT,
    ('BT', 'BO3'): solve_BT_BO3,
    ('BT', 'BO4'): solve_BT_BO4,
    ('BO3', 'BO4'): solve_BO3_BO4,
    ('pHtot', 'BO3'): solve_pH_BO3,
    ('pHtot', 'BO4'): solve_pH_BO4
}

def given(params):
    """Check which boron parameters are given in the parameters"""
    valid_inputs = ['BT', 'BO3', 'BO4']
    return [params.get(p) for p in valid_inputs if params.get(p) is not None]

def n_given(params):
    return len(given(params))

def calc_remaining_B_species(params):
    if 'BO3' in params.__dataclass_fields__:
        params.BO3 = params.BO3 or cBO3(params.BT, params.H, params.Ks)
        params.BO4 = params.BO4 or cBO4(params.BT, params.H, params.Ks)
    params.pHtot = params.pHtot or negative_log10_preserve_type(params.H)

def solve_B_system(params):
    boron_params = ['pHtot', 'BT', 'BO3', 'BO4']
    provided = tuple([p for p in boron_params if params.get(p) is not None])

    solver = SOLVERS.get(provided)
    if solver is None:
        raise ValueError(f"No solver found for parameter combination: {provided}")
    
    solver(params)

    calc_remaining_B_species(params)

