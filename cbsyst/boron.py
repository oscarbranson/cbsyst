import numpy as np
from cbsyst.helpers import ch, cp, Bunch

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
    H = ch(pH)
    return BO3 * (1 + Ks.KB / H)


def pH_BO4(pH, BO4, Ks):
    """
    Returns BT
    """
    H = ch(pH)
    return BO4 * (1 + H / Ks.KB)


def cBO4(BT, H, Ks):
    return BT / (1 + H / Ks.KB)


def cBO3(BT, H, Ks):
    return BT / (1 + Ks.KB / H)

def calc_B_species(pHtot=None, BT=None, BO3=None, BO4=None, Ks=None, **kwargs):
    # B system calculations
    if pHtot is not None and BT is not None:
        H = ch(pHtot)
    elif BT is not None and BO3 is not None:
        H = BT_BO3(BT, BO3, Ks)
    elif BT is not None and BO4 is not None:
        H = BT_BO4(BT, BO4, Ks)
    elif BO3 is not None and BO4 is not None:
        BT = BO3 + BO3
        H = BT_BO3(BT, BO3, Ks)
    elif pHtot is not None and BO3 is not None:
        H = ch(pHtot)
        BT = pH_BO3(pHtot, BO3, Ks)
    elif pHtot is not None and BO4 is not None:
        H = ch(pHtot)
        BT = pH_BO4(pHtot, BO4, Ks)

    # The above makes sure that BT and H are known,
    # this next bit calculates all the missing species
    # from BT and H.

    if BO3 is None:
        BO3 = cBO3(BT, H, Ks)
    if BO4 is None:
        BO4 = cBO4(BT, H, Ks)
    if pHtot is None:
        pHtot = np.array(cp(H), ndmin=1)

    return Bunch({"pHtot": pHtot, "H": H, "BT": BT, "BO3": BO3, "BO4": BO4})
