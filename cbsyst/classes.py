class const:
    def __init__(self, T, S, P, Mg=None, Ca=None, TS=None, TF=None, TB=None, mode='calculate'):
        if TS is None:
            TS = calc_TS(S)
        if TF is None:
            TF = calc_TF(S)
        
        self.TS = TS
        self.TF = TF
        self.Mg = Mg
        self.Ca = Ca
        
        self.Ks = calc_Ks(T=T, S=S, P=P, Mg=Mg, Ca=Ca, TS=TS, TF=TF, MyAMI_Mode=mode)