import kgen
from ..MyAMI import calc_Fcorr, approximate_Fcorr
from ..helpers import Bunch

def calc_Ks(T, S, P, Mg=None, Ca=None, TS=None, TF=None, Ks=None, MyAMI_mode='calculate'):
    """
    Returns all Ks on the Total scale at given conditions.
    """
    if isinstance(Ks, dict):
        Ks = Bunch(Ks)
    else:
        # calculate empirical Ks on Total scale
        Ks = Bunch(kgen.calc_Ks(TempC=T, Sal=S, Pres=P))

        if Ca is not None or Mg is not None:
            if MyAMI_mode == 'calculate':
                Fcorr = calc_Fcorr(Sal=S, TempC=T, Mg=Mg, Ca=Ca)
            else:
                Fcorr = approximate_Fcorr(Sal=S, TempC=T, Mg=Mg, Ca=Ca)
            
            for k, f in Fcorr.items():
                Ks[k] *= f

    return Ks