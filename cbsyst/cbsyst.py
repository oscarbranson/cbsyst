import numpy as np
from cbsyst.MyAMI_V2 import MyAMI_params, MyAMI_pK_calc
from tqdm import tqdm


class cbsyst(object):
    """
    Calculate the speciation of Carbon and Boron in Seawater.

    Given a minimal parameter set, cbsyst calculates the speciation of C
    and B in seawater, and the isotope fractionation of B.

    Parameters
    ----------
    pH : float or array-like
        pH of the water.


    Carbon Inputs:
    DIC   (2000 µmol/kg)
    pK1, pK2    Defaults to T and S sensitive values, calculated following
                Mehrbach et al, 1973.
    CO2, HCO3 or CO3  (optional)  If one of these is provided it is used to
                determine total DIC before calculating the other carbonate species.

    Boron Inputs:
    BT    (433 µmol/kg)
    pKB   Defaults to T and S sensitive value calculated from Dickson, 1990 (eq 23)
    BO3, BO4  (optional) If one of these is provided it will be used to
          calculate BT and the other B species.

    Boron Isotope Inputs:
    ABT   (0.8078..., equivalent to 39.6 ∂11B)
    alphaB  Defaults to T sensitive parameterisation of Hönisch et al, 2008

    Universal Parameters:
    S     (35)
    T     (25)
    Ca    (0.0102821)
    Mg    (0.0528171)

    Constants calculated using Mathis Hain's MyAMI model:

    Hain, M.P., Sigman, D.M., Higgins, J.A., and Haug, G.H. (2015) The effects of secular
    calcium and magnesium concentration changes on the thermodynamics of seawater acid/base
    chemistry: Implications for Eocene and Cretaceous ocean carbon chemistry and buffering,
    Global Biogeochemical Cycles, 29, doi:10.1002/2014GB004986

    A modified version of MyAMI is included with cbsyst.
    """
    def __init__(self, pH=8.2,
                 # carbonate parameters
                 DIC=2000., pK1=None, pK2=None,
                 CO2=None, HCO3=None, CO3=None,
                 # boron parameters
                 BT=433., pKB=None,
                 BO3=None, BO4=None,
                 ABT=0.807817779214075, alphaB=None,
                 ABO3=None, ABO4=None,
                 # universal parameters
                 s=35., t=25.,
                 Ca=0.0102821, Mg=0.0528171,
                 constants='MyAMI', update=True):
        if pH is None:
            raise ValueError('Must provide pH')

        # initiate provided parameters
        for key, param in locals().items():
            if key is not 'self':
                setattr(self, key, param)
        self.tK = self.t + 273.15
        self.H = 10**-self.pH

        # calculate constants
        if constants == 'MyAMI':
            self.pKcalc_MyAMI(Ca, Mg)
        else:
            if self.pKB is None:
                self.pKBcalc()
            if self.pK1 is None and self.pK2 is None:
                self.pKcalc()
            self.pKWcalc()

        if self.alphaB is None:
            self.alphaB_calc()
        
        self.chiB_calc()
        self.chiCarb_calc()

        if self.update:
            self.run()

    # set parameters, and update calculate (if update flag)
    def set_params(self, pH=None,
                   DIC=None, pK1=None, pK2=None,
                   CO2=None, HCO3=None, CO3=None,
                   BT=None, pKB=None,
                   BO3=None, BO4=None,
                   ABT=None, alphaB=None,
                   ABO3=None, ABO4=None,
                   s=None, t=None):
        Bset = ['BT', 'BO3', 'BO4']
        Aset = ['ABT', 'ABO3', 'ABO4']
        Cset = ['DIC', 'CO2', 'HCO3', 'CO3']
        univ = ['s', 't']
        for key, param in locals().items():
            if param is not None and key is not 'self':
                if key in univ:
                    # re-calculate constants
                    self.pKcalc()
                    self.pKBcalc()
                    self.pKWcalc()
                    self.alphaB_calc()
                    self.chiCarb_calc()
                if key in Aset + Bset + Cset:
                    # clear concentrations if one parameter is updated to
                    # ensure full calculation
                    if key in Bset:
                        [setattr(self, i, None) for i in Bset]
                    # clear isotopes if one parameter is updated to ensure
                    # full calculation
                    if key in Aset:
                        [setattr(self, i, None) for i in Aset]
                    # clear carbon parameters to ensure full calculation
                    if key in Cset:
                        [setattr(self, i, None) for i in Cset]
                    # re-assign provided attributes
                    setattr(self, key, param)
        if self.update:
            self.run()

        return

    def run(self):
        # calculate Carbon system
        if self.CO2 is None and self.HCO3 is None and self.CO3 is None:
            self.CO2 = self.DIC * self.chiCO2
            self.HCO3 = self.DIC * self.chiHCO3
            self.CO3 = self.DIC * self.chiCO3
        else:
            if self.CO2 is not None:
                self.DIC = self.CO2 / self.chiCO2
                self.HCO3 = self.DIC * self.chiHCO3
                self.CO3 = self.DIC * self.chiCO3
            if self.HCO3 is not None:
                self.DIC = self.HCO3 / self.chiHCO3
                self.CO2 = self.DIC * self.chiCO2
                self.CO3 = self.DIC * self.chiCO3
            if self.CO3 is not None:
                self.DIC = self.CO3 / self.chiCO3
                self.CO2 = self.DIC * self.chiCO2
                self.HCO3 = self.DIC * self.chiHCO3

        # calculate Boron system
        # Concentration
        if self.BO3 is None and self.BO4 is None:
            self.BO3 = self.BT * self.chiB
            self.BO4 = self.BT * (1 - self.chiB)
        else:
            if self.BO3 is not None:
                self.BT = self.BO3 / self.chiB
                self.BO4 = self.BT * (1 - self.chiB)
            if self.BO4 is not None:
                self.BT = self.BO4 / (1 - self.chiB)
                self.BO3 = self.BT * self.chiB

        # calculate alkalinity
        self.alkalinity = (self.HCO3 + 2 * self.CO3 + self.BO4 + self.KW /
                           self.H - self.H)

        # Isotopes
        if self.ABO3 is None and self.ABO4 is None:
            # ABO3 as a function of ABT, alphaB and chiB
            self.ABO3 = (self.ABT * self.alphaB - self.ABT + self.alphaB *
                         self.chiB - self.chiB -
                         np.sqrt(self.ABT ** 2 * self.alphaB ** 2 - 2 *
                                 self.ABT ** 2 * self.alphaB + self.ABT ** 2 -
                                 2 * self.ABT * self.alphaB ** 2 * self.chiB +
                                 2 * self.ABT * self.alphaB + 2 * self.ABT *
                                 self.chiB - 2 * self.ABT + self.alphaB ** 2 *
                                 self.chiB ** 2 - 2 * self.alphaB *
                                 self.chiB ** 2 + 2 * self.alphaB * self.chiB +
                                 self.chiB ** 2 - 2 * self.chiB + 1) + 1) / \
                         (2 * self.chiB * (self.alphaB - 1))
            # ABO4 as a function of ABT, alphaB and chiB
            self.ABO4 = -(self.ABT * self.alphaB - self.ABT - self.alphaB *
                          self.chiB + self.chiB +
                          np.sqrt(self.ABT ** 2 * self.alphaB ** 2 - 2 *
                                  self.ABT ** 2 * self.alphaB + self.ABT ** 2 -
                                  2 * self.ABT * self.alphaB ** 2 * self.chiB +
                                  2 * self.ABT * self.alphaB + 2 * self.ABT *
                                  self.chiB - 2 * self.ABT + self.alphaB ** 2 *
                                  self.chiB ** 2 - 2 * self.alphaB *
                                  self.chiB ** 2 + 2 * self.alphaB *
                                  self.chiB + self.chiB ** 2 - 2 * self.chiB +
                                  1) - 1)/(2 * self.alphaB * self.chiB - 2 *
                                           self.alphaB - 2 * self.chiB + 2)
        else:
            if self.ABO4 is not None:
                # ABT as a function of ABO4, alphaB and chiB
                self.ABT = self.ABO4 * (-self.ABO4 * self.alphaB * self.chiB +
                                        self.ABO4 * self.alphaB + self.ABO4 *
                                        self.chiB - self.ABO4 + self.alphaB *
                                        self.chiB - self.chiB + 1) / \
                           (self.ABO4 * self.alphaB - self.ABO4 + 1)
                # ABO3 as a function of ABT, alphaB and chiB
                self.ABO3 = (self.ABT * self.alphaB - self.ABT + self.alphaB *
                             self.chiB - self.chiB -
                             np.sqrt(self.ABT ** 2 * self.alphaB ** 2 - 2 *
                                     self.ABT ** 2 * self.alphaB +
                                     self.ABT ** 2 - 2 * self.ABT *
                                     self.alphaB ** 2 * self.chiB + 2 *
                                     self.ABT * self.alphaB + 2 * self.ABT *
                                     self.chiB - 2 * self.ABT +
                                     self.alphaB ** 2 * self.chiB ** 2 - 2 *
                                     self.alphaB * self.chiB ** 2 + 2 *
                                     self.alphaB * self.chiB + self.chiB ** 2 -
                                     2 * self.chiB + 1) + 1) / \
                            (2 * self.chiB * (self.alphaB - 1))
            if self.ABO3 is not None:
                # ABT as a function of ABO3, alphaB and chiB
                self.ABT = self.ABO3 * (-self.ABO3 * self.alphaB * self.chiB +
                                        self.ABO3 * self.chiB + self.alphaB *
                                        self.chiB - self.chiB + 1) / \
                           (-self.ABO3 * self.alphaB + self.ABO3 + self.alphaB)
                # ABO4 as a function of ABT, alphaB and chiB
                self.ABO4 = -(self.ABT * self.alphaB - self.ABT - self.alphaB *
                              self.chiB + self.chiB +
                              np.sqrt(self.ABT ** 2 * self.alphaB ** 2 - 2 *
                                      self.ABT ** 2 * self.alphaB + self.ABT **
                                      2 - 2 * self.ABT * self.alphaB ** 2 *
                                      self.chiB + 2 * self.ABT * self.alphaB +
                                      2 * self.ABT * self.chiB - 2 * self.ABT +
                                      self.alphaB ** 2 * self.chiB ** 2 - 2 *
                                      self.alphaB * self.chiB ** 2 + 2 *
                                      self.alphaB * self.chiB + self.chiB **
                                      2 - 2 * self.chiB + 1) - 1) / \
                             (2 * self.alphaB * self.chiB - 2 * self.alphaB -
                              2 * self.chiB + 2)
        return

    # constant calculators
    def pKcalc_MyAMI(self, Ca, Mg):
        Ca = np.array(Ca)
        Mg = np.array(Mg)

        if (Ca.size == Mg.size) | (Mg.size == 1):
            CaMg = np.zeros((Ca.size, 2))
            CaMg[:, 0] = Ca
            CaMg[:, 1] = Mg
        elif Ca.size == 1:
            CaMg = np.zeros((Mg.size, 2))
            CaMg[:, 0] = Ca
            CaMg[:, 1] = Mg
        else:
            raise ValueError('Ca and Mg must be the same length, or one must have a size of 1.')

        # calculate Ks
        size = np.max(CaMg.shape[0])
        pK1s = np.zeros(size)
        pK2s = np.zeros(size)
        pKBs = np.zeros(size)
        KWs = np.zeros(size)

        if isinstance(self.t, (str, float)):
            t = np.array([self.t] * size)
        else:
            t = self.t

        if isinstance(self.s, (str, float)):
            s = np.array([self.s] * size)
        else:
            s = self.s

        # determine unique combinations of Ca and Mg
        unique = set([(ca, mg) for ca, mg in CaMg])
        # calculate parameters for each unique combo
        upars = {}

        for (ca, mg) in tqdm(unique, desc='Calculating MyAMI Constants', leave=False):
            upars[(ca, mg)] = MyAMI_params(ca, mg)

            ind = (CaMg[:, 0] == ca) & (CaMg[:, 1] == mg)

            pKs = MyAMI_pK_calc(t[ind], s[ind], upars[(ca, mg)])

            pK1s[ind] = pKs['K1']
            pK2s[ind] = pKs['K2']
            pKBs[ind] = pKs['Kb']
            KWs[ind] = 10**-pKs['Kw']

        self.pK1 = pK1s
        self.pK2 = pK2s
        self.pKB = pKBs
        self.KW = KWs
        self.upars = upars

        return

    def pKcalc(self):
        # Calculate fH
        # fH = (1.2948 - 0.002036 * self.tK +
        #       (0.0004607 - 0.000001475 * self.tK) * self.s**2)
        # Takahashi et al, Chapter 3 in
        # GEOSECS Pacific Expedition, v. 3, 1982 (p. 80)

        # calculate K1 & K2
        pK1 = (-13.7201 + 0.031334 * self.tK + 3235.76 / self.tK + 1.3e-5 *
               self.s * self.tK - 0.1032*self.s**0.5)
        # K1 = 10**(-pK1)/fH)

        pK2 = (5371.9645 + 1.671221*self.tK + 0.22913 * self.s + 18.3802 *
               np.log10(self.s) - 128375.28 / self.tK - 2194.3055 *
               np.log10(self.tK) - 8.0944e-4 * self.s * self.tK - 5617.11 *
               np.log10(self.s) / self.tK + 2.136 * self.s / self.tK)
        # K2 = 10**(-pK2) /fH
        # GEOSECS and Peng et al use K1, K2 from Mehrbach et al,
        # Limnology and Oceanography, 18(6):897-907, 1973.

        self.pK1 = pK1
        self.pK2 = pK2

        return self.pK1, self.pK2

    def pKBcalc(self):
        # From Dickson, 1990: eqn 23
        K = (-8966.9 - 2890.53 * self.s ** 0.5 - 77.942 * self.s + 1.728 *
             self.s ** 1.5 - 0.0996 * self.s ** 2) \
            * (1/self.tK) \
            + (148.0248 + 137.1942 * self.s ** 0.5 + 1.62142 * self.s) \
            + (-24.4344 - 25.085 * self.s ** 0.5 - 0.2474 * self.s) \
            * np.log(self.tK) \
            + (0.053105 * self.s ** 0.5) * self.tK
        self.pKB = -np.log10(np.exp(K))
        return self.pKB

    def pKWcalc(self):
        # From Millero, Geochemica et Cosmochemica Acta 59:661-677, 1995
        lnKW = (148.9802 - 13847.26 / self.tK - 23.6521 * np.log(self.tK) +
                (-5.977 + 118.67 / self.tK + 1.0495 * np.log(self.tK)) *
                self.s**0.5 - 0.01615 * self.s)
        self.KW = np.exp(lnKW)

    def alphaB_calc(self):
        self.alphaB = 1.0293 - 0.000082 * self.t  # From Honisch et al, 2008
        return self.alphaB

    def chiB_calc(self):
        self.chiB = 10 ** -self.pH / (10 ** -self.pKB + 10 ** -self.pH)
        return self.chiB

    def chiCarb_calc(self):
        self.chiCO2 = 1 / (1 + 10**-self.pK1 / self.H +
                           (10**-self.pK1 * 10**-self.pK2) / self.H**2)
        self.chiHCO3 = 1 / (1 + self.H / 10**-self.pK1 +
                            10**-self.pK2 / self.H)
        self.chiCO3 = 1 / (1 + self.H / 10**-self.pK2 + self.H**2 /
                           (10**-self.pK1 * 10**-self.pK2))


# Unit Converters
def A11_2_d11(A11, NIST951=4.04367):
    return ((A11 / (1-A11)) / NIST951 - 1) * 1000


def A11_2_R11(A11):
    return A11 / (1 - A11)


def d11_2_A11(d11, NIST951=4.04367):
    return NIST951 * (d11 / 1000 + 1) / (NIST951 * (d11 / 1000 + 1) + 1)


def d11_2_R11(d11, NIST951=4.04367):
    return (d11/1000 + 1) * NIST951


def R11_2_d11(R11, NIST951=4.04367):
    return (R11 / NIST951 - 1) * 1000


def R11_2_A11(R11):
    return R11 / (1 + R11)
