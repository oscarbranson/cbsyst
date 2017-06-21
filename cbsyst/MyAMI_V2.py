# June 2017 : Oscar Branson : oscarbranson@gmail.com
# --------------------------------------------------
#
# Adapted Mathis Hain's original code to:
#  1. Work with Python 3.
#  2. Vectorise with numpy, for speed.
#  3. Conform to PEP8 formatting.
#  4. Condense functions into two files
#  5. Make it work with the cbsyst module
#     (https://github.com/oscarbranson/cbsyst) for
#     calculating seawater carbonate and B chem in seawater.
#
# Original Header
# ---------------
# MyAMI Specific Ion Interaction Model (Version 1.0):
# This is a Python script to calculate thermodynamic pK's and conditional pK's
# Author: Mathis P. Hain -- m.p.hain@soton.ac.uk
#
# Reference:
# Hain, M.P., Sigman, D.M., Higgins, J.A., and Haug, G.H. (2015) The effects of secular calcium and magnesium concentration changes on the thermodynamics of seawater acid/base chemistry: Implications for Eocene and Cretaceous ocean carbon chemistry and buffering, Global Biogeochemical Cycles, 29, doi:10.1002/2014GB004986
#
# For general context on the calculations see Millero, 2007 (Chemical Reviews) and Millero and Pierrot, 1998 (Aquatic Geochemistry)

import itertools
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from cbsyst.helpers import Bunch, prescorr


# Functions from K_thermo_conditional.py
# --------------------------------------
# definition of the function that takes (Temp) as input and returns the K at that temp
def CalculateKcond(Tc, Sal):
    """
    Calculate thermodynamic Ks adjusted for salinity.

    Parameters
    ----------
    Tc : float or array-like
        Temperature in C
    Sal : float or array-like
        Salinity in PSU
    P : float of array-like:
        Pressure in bar.
    """
    sqrtSal = np.sqrt(Sal)
    T = Tc + 273.15
    lnT = np.log(T)
    I = 19.924 * Sal / (1000 - 1.005 * Sal)  # Ionic strength after Dickson 1990a; see Dickson et al 2007

    KspCcond = np.power(10, (-171.9065 - 0.077993 * T + 2839.319 / T + 71.595 *
                             np.log10(T) + (-0.77712 + 0.0028426 * T + 178.34 / T) *
                             sqrtSal - 0.07711 * Sal + 0.0041249 * Sal * sqrtSal))

    K1cond = np.power(10, -3633.86 / T + 61.2172 - 9.67770 * lnT + 0.011555 * Sal - 0.0001152 * Sal * Sal)  # Dickson
    # K1cond = np.exp(290.9097 - 14554.21 / T - 45.0575 * lnT + (-228.39774 + 9714.36839 / T + 34.485796 * lnT) * sqrtSal + (54.20871 - 2310.48919 / T - 8.19515 * lnT) * Sal + (-3.969101 + 170.22169 / T + 0.603627 * lnT) * Sal * sqrtSal - 0.00258768 * Sal * Sal) #Millero95

    K2cond = np.power(10, -471.78 / T - 25.9290 + 3.16967 * lnT + 0.01781 * Sal - 0.0001122 * Sal * Sal)

    KWcond = np.exp(148.9652 - 13847.26 / T - 23.6521 * lnT + (118.67 / T - 5.977 + 1.0495 * lnT) * sqrtSal - 0.01615 * Sal)

    KBcond = np.exp((-8966.90 - 2890.53 * sqrtSal - 77.942 * Sal + 1.728 * Sal * sqrtSal - 0.0996 * Sal * Sal) /
                    T + (148.0248 + 137.1942 * sqrtSal + 1.62142 * Sal) + (-24.4344 - 25.085 * sqrtSal - 0.2474 * Sal) *
                    lnT + 0.053105 * sqrtSal * T)  # Dickson90b

    KspAcond = np.power(10, (-171.945 - 0.077993 * T + 2903.293 / T + 71.595 * np.log10(T) +
                             (-0.068393 + 0.0017276 * T + 88.135 / T) * sqrtSal -
                             0.10018 * Sal + 0.0059415 * Sal * sqrtSal))

    K0cond = np.exp(-60.2409 + 93.4517 * 100 / T + 23.3585 * np.log(T / 100) +
                    Sal * (0.023517 - 0.023656 * T / 100 + 0.0047036 * (T / 100) * (T / 100)))  # Weiss74

    param_HSO4_cond = np.array([141.328, -4276.1, -23.093, 324.57, -13856, -47.986, -771.54, 35474, 114.723, -2698, 1776])  # Dickson 1990

    KHSO4cond = np.exp(param_HSO4_cond[0] +
                       param_HSO4_cond[1] / T +
                       param_HSO4_cond[2] * np.log(T) + np.sqrt(I) *
                       (param_HSO4_cond[3] +
                        param_HSO4_cond[4] / T +
                        param_HSO4_cond[5] * np.log(T)) + I *
                       (param_HSO4_cond[6] +
                        param_HSO4_cond[7] / T +
                        param_HSO4_cond[8] * np.log(T)) +
                       param_HSO4_cond[9] / T * I * np.sqrt(I) +
                       param_HSO4_cond[10] / T * I**2 + np.log(1 - 0.001005 * Sal))

    return KspCcond, K1cond, K2cond, KWcond, KBcond, KspAcond, K0cond, KHSO4cond


# Functions from PitzerParams.py
# --------------------------------------
def SupplyParams(T):  # assumes T [K] -- not T [degC]
    """
    Return Pitzer params for given T (Kelvin).
    """
    if isinstance(T, (float, int)):
        T = [T]

    Tinv = 1 / T
    lnT = np.log(T)
    # ln_of_Tdiv29815 = np.log(T / 298.15)
    Tpower2 = T**2
    Tpower3 = T**3
    Tpower4 = T**4

    # PART 1 -- calculate thermodynamic pK's for acids, gases and complexes

    # paramerters [A, B, C, D] according to Millero (2007) Table 11
    # param_HF = [-12.641, 1590.2, 0, 0]
    # param_H2S = [225.8375, -13275.324, -34.64354, 0]
    # param_H2O = [148.9802, -13847.26, -23.6521, 0]
    # param_BOH3 = [148.0248, -8966.901, -24.4344, 0]
    # param_HSO4 = [141.411, -4340.704, -23.4825, 0.016637]
    # param_NH4 = [-0.25444, -6285.33, 0, 0.0001635]
    # param_H2CO3 = [290.9097, -14554.21, -45.0575, 0]
    # param_HCO3 = [207.6548, -11843.79, -33.6485, 0]
    # param_H2SO3 = [554.963, -16700.1, -93.67, 0.1022]
    # param_HSO3 = [-358.57, 5477.1, 65.31, -0.1624]
    # param_H3PO4 = [115.54, -4576.7518, -18.453, 0]
    # param_H2PO4 = [172.1033, -8814.715, -27.927, 0]
    # param_HPO4 = [-18.126, -3070.75, 0, 0]
    # param_CO2 = [-60.2409, 9345.17, 18.7533, 0]
    # param_SO2 = [-142.679, 8988.76, 19.8967, -0.0021]
    # param_Aragonite = [303.5363, -13348.09, -48.7537, 0]
    # param_Calcite = [303.1308, -13348.09, -48.7537, 0]

    # definition of the function that takes (Temp, param) as input and returns the lnK at that temp
    #     def Eq_lnK_calcABCD(T, paramABCD):
    #         return paramABCD[0] + paramABCD[1] / T + paramABCD[2] * np.log(T) + paramABCD[3] * T
    # How to use:  ln_of_K_HCO3_at_18degC = lnK_calcABCD(18, param_HCO3)

    # paramerters [A, B, C] according to Millero (2007) Table 12
    # param_MgOH = [3.87, -501.6, 0]
    # param_MgF = [3.504, -501.6, 0]
    # param_CaF = [3.014, -501.6, 0]
    # param_MgCO3 = [1.028, 0, 0.0066154]
    # param_CaCO3 = [1.178, 0, 0.0066154]
    # param_SrCO3 = [1.028, 0, 0.0066154]
    # param_MgH2PO4 = [1.13, 0, 0]
    # param_CaH2PO4 = [1, 0, 0]
    # param_MgHPO4 = [2.7, 0, 0]
    # param_CaHPO4 = [2.74, 0, 0]
    # param_MgPO4 = [5.63, 0, 0]
    # param_CaPO4 = [7.1, 0, 0]

    # definition of the function that takes (Temp, param) as input and returns the lnK at that temp
    #     def lnK_calcABC(T, paramABC):
    #         return paramABC[0] + paramABC[1] / T + paramABC[2] * T
    # How to use:  ln_of_K_CaHPO4_at_18degC = lnK_calcABC(18, param_CaHPO4)

    ################################################################################
    # PART 2 -- Pitzer equations (based on Millero and Pierrot (1998))

    # Table A1 (Millero and Pierrot, 1998; after Moller, 1988 & Greenberg and Moller, 1989) valid 0 to 250degC
    param_NaCl = np.array([(1.43783204E01, 5.6076740E-3, -4.22185236E2, -2.51226677E0, 0.0, -2.61718135E-6, 4.43854508, -1.70502337),
                           (-4.83060685E-1, 1.40677470E-3, 1.19311989E2, 0.0, 0.0, 0.0, 0.0, -4.23433299),
                           (-1.00588714E-1, -1.80529413E-5, 8.61185543E0, 1.2488095E-2, 0.0, 3.41172108E-8, 6.83040995E-2, 2.93922611E-1)])
    # note that second value is changed to original ref (e-3 instead e01)
    param_NaCl = reshaper(param_NaCl, T)
    param_KCl = np.array([[2.67375563E1, 1.00721050E-2, -7.58485453E2, -4.70624175, 0.0, -3.75994338E-6, 0.0, 0.0],
                          [-7.41559626, 0.0, 3.22892989E2, 1.16438557, 0.0, 0.0, 0.0, -5.94578140],
                          [-3.30531334, -1.29807848E-3, 9.12712100E1, 5.864450181E-1, 0.0, 4.95713573E-7, 0.0, 0.0]])
    param_KCl = reshaper(param_KCl, T)
    param_K2SO4 = np.array([[4.07908797E1, 8.26906675E-3, -1.418242998E3, -6.74728848, 0.0, 0.0, 0.0, 0.0],
                            [-1.31669651E1, 2.35793239E-2, 2.06712592E3, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-1.88E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    param_K2SO4 = reshaper(param_K2SO4, T)
    param_CaCl2 = np.array([[-9.41895832E1, -4.04750026E-2, 2.34550368E3, 1.70912300E1, -9.22885841E-1, 1.51488122E-5, -1.39082000E0, 0.0],
                            [3.4787, -1.5417E-2, 0.0, 0.0, 0.0, 3.1791E-5, 0.0, 0.0],
                            [1.93056024E1, 9.77090932E-3, -4.28383748E2, -3.57996343, 8.82068538E-2, -4.62270238E-6, 9.91113465, 0.0]])
    param_CaCl2 = reshaper(param_CaCl2, T)
    # [-3.03578731e1, 1.36264728e-2, 7.64582238e2, 5.50458061e0, -3.27377782e-1, 5.69405869e-6, -5.36231106e-1, 0]])
    # param_CaCl2_Spencer = np.array([[-5.62764702e1, -3.00771997e-2, 1.05630400e-5, 3.3331626e-9, 1.11730349e3, 1.06664743e1],
    #                                    [3.4787e0, -1.5417e-2, 3.1791e-5, 0, 0, 0],
    #                                    [2.64231655e1, 2.46922993e-2, -2.48298510e-5, 1.22421864e-8, -4.18098427e2, -5.35350322e0]])
    # param_CaSO4 = np.array([[0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # corrected after Greenberg and Moller 1989 0.015 instead of 0.15
    # param_SrSO4 = param_CaSO4
    # param_CaSO4_Spencer = np.array([[0.795e-1, -0.122e-3, 0.5001e-5, 0.6704e-8, -0.15228e3, -0.6885e-2],
    #                                    [0.28945e1, 0.7434e-2, 0.5287e-5, -0.101513e-6, -0.208505e4, 0.1345e1]])

    def Equation_TabA1(T, Tinv, lnT, a):
        return (a[:, 0] + a[:, 1] * T + a[:, 2] * Tinv + a[:, 3] * lnT + a[:, 4] /
                (T - 263) + a[:, 5] * T**2 + a[:, 6] / (680 - T) + a[:, 7] / (T - 227))

    def EquationSpencer(T, lnT, q):
        return q[:, 0] + q[:, 1] * T + q[:, 2] * T * T + q[:, 3] * T**3 + q[:, 4] / T + q[:, 5] * lnT

    # Table A2 (Millero and Pierrot, 1998; after Pabalan and Pitzer, 1987) valid 25 to 200degC
    param_MgCl2 = np.array([[0.576066, -9.31654E-04, 5.93915E-07],
                            [2.60135, -0.0109438, 2.60169E-05],
                            [0.059532, -2.49949E-04, 2.41831E-07]])
    param_MgCl2 = reshaper(param_MgCl2, T)
    param_MgSO4 = np.array([[-1.0282, 8.4790E-03, -2.33667E-05, 2.1575E-08, 6.8402E-04, 0.21499],
                            [-2.9596E-01, 9.4564E-04, 0.0, 0.0, 1.1028E-02, 3.3646],
                            [4.2164E-01, -3.5726E-03, 1.0040E-05, -9.3744E-09, -3.5160E-04, 2.7972E-02]])
    param_MgSO4 = reshaper(param_MgSO4, T)
    # param_MgSO4 = np.array([[-1.0282, 8.4790E-03, -2.33667E-05, 2.1575E-08, 6.8402E-04, 0.21499],[-2.9596E-01, 9.4564E-04, 0.0, 0.0, 1.1028E-02, 3.3646], [1.0541E-01, -8.9316E-04, 2.51E-06, -2.3436E-09, -8.7899E-05, 0.006993]])  # Cparams corrected after Pabalan and Pitzer ... but note that column lists Cmx not Cphi(=4xCmx) ... MP98 is correct

    def Equation1_TabA2(T, q):
        return q[:, 0] + q[:, 1] * T + q[:, 2] * T**2

    def Equation2_TabA2(T, Tpower2, Tpower3, Tpower4, q):
        return (q[:, 0] * ((T / 2) + (88804) / (2 * T) - 298) +
                q[:, 1] * ((Tpower2 / 6) + (26463592) / (3 * T) - (88804 / 2)) +
                q[:, 2] * (Tpower3 / 12 + 88804 * 88804 / (4 * T) - 26463592 / 3) +
                q[:, 3] * ((Tpower4 / 20) + 88804 * 26463592 / (5 * T) - 88804 * 88804 / 4) +
                q[:, 4] * (298 - (88804 / T)) +
                q[:, 5])

    # Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC)
    # param_NaHSO4 = np.array([[0.030101, -0.362E-3, 0.0], [0.818686, -0.019671, 0.0], [0.0, 0.0, 0.0]])  # corrected after Pierrot et al., 1997
    param_NaHSO4 = np.array([[0.0544, -1.8478e-3, 5.3937e-5],
                             [0.3826401, -1.8431e-2, 0.0],
                             [0.003905, 0.0, 0.0]])  # corrected after Pierrot and Millero, 1997
    param_NaHSO4 = reshaper(param_NaHSO4, T)
    param_NaHCO3 = np.array([[0.028, 1.0E-3, -2.6E-5 / 2],
                             [0.044, 1.1E-3, -4.3E-5 / 2],
                             [0.0, 0.0, 0.0]])  # corrected after Peiper and Pitzer 1982
    # param_Na2SO4 = np.array([[6.536438E-3, -30.197349, -0.20084955],
    #                             [0.8742642, -70.014123, 0.2962095],
    #                             [7.693706E-3, 4.5879201, 0.019471746]])  # corrected according to Hovey et al 1993; note also that alpha = 1.7, not 2
    param_NaHCO3 = reshaper(param_NaHCO3, T)
    param_Na2SO4_Moller = np.array([[81.6920027 + 0.0301104957 * T - 2321.93726 / T - 14.3780207 * lnT - 0.666496111 / (T - 263) - 1.03923656e-05 * T**2],
                                    [1004.63018 + 0.577453682 * T - 21843.4467 / T - 189.110656 * lnT - 0.2035505488 / (T - 263) - 0.000323949532 * T**2 + 1467.72243 / (680 - T)],
                                    [-80.7816886 - 0.0354521126 * T + 2024.3883 / T + 14.619773 * lnT - 0.091697474 / (T - 263) + 1.43946005e-05 * T**2 - 2.42272049 / (680 - T)]])
    # Moller 1988 parameters as used in Excel MIAMI code !!!!!! careful this formula assumes alpha1=2 as opposed to alpha1=1.7 for the Hovey parameters
    # XXXXX - - > need to go to the calculation of beta's (to switch Hovey / Moller) and of B et al (to switch alpha1

    # param_Na2CO3 = np.array([[0.0362, 1.79E-3, 1.694E-21], [1.51, 2.05E-3, 1.626E-19], [0.0052, 0.0, 0.0]])  # Millero and Pierrot referenced to Peiper and Pitzer
    param_Na2CO3 = np.array([[0.0362, 1.79E-3, -4.22E-5 / 2],
                             [1.51, 2.05E-3, -16.8E-5 / 2],
                             [0.0052, 0.0, 0.0]])  # Peiper and Pitzer 1982
    param_Na2CO3 = reshaper(param_Na2CO3, T)
    # XXXX check below if Haynes 2003 is being used.

    param_NaBOH4 = np.array([[-0.051, 5.264E-3, 0.0],
                             [0.0961, -1.068E-2, 0.0],
                             [0.01498, -15.7E-4, 0.0]])  # corrected after Simonson et al 1987 5th param should be e-2
    param_NaBOH4 = reshaper(param_NaBOH4, T)

    def Equation_TabA3andTabA4andTabA5(T, a):
        return (a[:, 0] +
                a[:, 1] * (T - 298.15) +
                a[:, 2] * (T - 298.15) * (T - 298.15))

    # def Equation_Na2SO4_TabA3(T, ln_of_Tdiv29815, a):
    #     return (a[:, 0] + a[:, 1] * ((1 / T) - (1 / 298.15)) + a[:, 2] * ln_of_Tdiv29815)

    # Table A4 (Millero and Pierrot, 1998; after mutiple studies, at least valid 5 to 45degC)
    param_KHCO3 = np.array([[-0.0107, 0.001, 0.0],
                            [0.0478, 0.0011, 6.776E-21],
                            [0.0, 0.0, 0.0]])
    param_KHCO3 = reshaper(param_KHCO3, T)
    param_K2CO3 = np.array([[0.1288, 1.1E-3, -5.1E-6],
                            [1.433, 4.36E-3, 2.07E-5],
                            [0.0005, 0.0, 0.0]])
    param_K2CO3 = reshaper(param_K2CO3, T)
    param_KBOH4 = np.array([[0.1469, 2.881E-3, 0.0],
                            [-0.0989, -6.876E-3, 0.0],
                            [-56.43 / 1000, -9.56E-3, 0.0]])  # corrected after Simonson et al 1988
    param_KBOH4 = reshaper(param_KBOH4, T)
    # same function as TabA3 "Equation_TabA3andTabA4andTabA5(T,a)"

    # Table A5 (Millero and Pierrot, 1998; after Simonson et al, 1987b; valid 5 - 55degC
    param_MgBOH42 = np.array([[-0.623, 6.496E-3, 0.0],
                              [0.2515, -0.01713, 0.0],
                              [0.0, 0.0, 0.0]])  # corrected after Simonson et al 1988 first param is negative
    param_MgBOH42 = reshaper(param_MgBOH42, T)
    param_CaBOH42 = np.array([[-0.4462, 5.393E-3, 0.0],
                              [-0.868, -0.0182, 0.0],
                              [0.0, 0.0, 0.0]])
    param_CaBOH42 = reshaper(param_CaBOH42, T)
    param_SrBOH42 = param_CaBOH42  # see Table A6

    def Equation_TabA3andTabA4andTabA5_Simonson(T, a):
        return (a[:, 0] +
                a[:, 1] * (T - 298.15) +
                a[:, 2] * (T - 303.15) * (T - 303.15))

    # Table A7 (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC
    param_KOH = np.array([[0.1298, -0.946E-5, 9.914E-4],
                          [0.32, -2.59E-5, 11.86E-4],
                          [0.0041, 0.0638E-5, -0.944E-4]])
    param_KOH = reshaper(param_KOH, T)
    param_SrCl2 = np.array([[0.28575, -0.18367E-5, 7.1E-4],
                            [1.66725, 0.0E-5, 28.425E-4],
                            [-0.0013, 0.0E-5, 0.0E-4]])
    param_SrCl2 = reshaper(param_SrCl2, T)

    def Equation_TabA7(T, P):
        return (P[:, 0] +
                P[:, 1] * (8834524.639 - 88893.4225 * P[:, 2]) * (1 / T - (1 / 298.15)) +
                P[:, 1] / 6 * (T**2 - 88893.4225))

    # Table A8 - - - Pitzer parameters unknown; beta's known for 25degC
    Equation_KHSO4 = np.array([-0.0003, 0.1735, 0.0])
    # Equation_MgHSO42 = np.array([0.4746, 1.729, 0.0])  #  XX no Cphi #from Harvie et al 1984 as referenced in MP98
    Equation_MgHSO42 = np.array([-0.61656 - 0.00075174 * (T - 298.15), 7.716066 - 0.0164302 * (T - 298.15), 0.43026 + 0.00199601 * (T - 298.15)])  # from Pierrot and Millero 1997 as used in the Excel file

    # Equation_MgHCO32 = np.array([0.329, 0.6072, 0.0])  # Harvie et al 1984
    Equation_MgHCO32 = np.array([0.03, 0.8, 0.0])  # Millero and Pierrot redetermined after Thurmond and Millero 1982
    Equation_CaHSO42 = np.array([0.2145, 2.53, 0.0])
    Equation_CaHCO32 = np.array([0.4, 2.977, 0.0])  # np.array([0.2, 0.3, 0]) He and Morse 1993 after Pitzeretal85 np.array([0.4, 2.977, 0.0])
    Equation_CaOH2 = np.array([-0.1747, -0.2303, -5.72])  # according to Harvie84, the -5.72 should be for beta2, not Cphi (which is zero) -- but likely typo in original ref since 2:1 electrolytes don't usually have a beta2
    Equation_SrHSO42 = Equation_CaHSO42
    Equation_SrHCO32 = Equation_CaHCO32
    Equation_SrOH2 = Equation_CaOH2
    # Equation_MgOHCl = np.array([-0.1, 1.658, 0.0])
    Equation_NaOH = np.array([0.0864, 0.253, 0.0044])  # Rai et al 2002 ref to Pitzer91(CRC Press)
    Equation_CaSO4_PnM74 = np.array([0.2, 2.65, 0])  # Pitzer and Mayorga74

    # Table A9 - - - (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC
    param_HCl = np.array([[1.2859, -2.1197e-3, -142.58770],
                          [-4.4474, 8.425698E-3, 665.7882],
                          [-0.305156, 5.16E-4, 45.521540]])  # beta1 first param corrected to negative according to original reference (Campbell et al)
    param_HCl = reshaper(param_HCl, T)
    # param_HSO4 = np.array([[0.065, 0.134945, 0.022374, 7.2E-5],
    #                           [-15.009, -2.405945, 0.335839, -0.004379],
    #                           [0.008073, -0.113106, -0.003553, 3.57E-5]])  # XXXXX two equations for C
    # param_HSO4_Clegg94 = np.array([[0.0348925351, 4.97207803, 0.317555182, 0.00822580341],
    #                                  [-1.06641231, -74.6840429, -2.26268944, -0.0352968547],
    #                                  [0.00764778951, -0.314698817, -0.0211926525, 0.000586708222],
    #                                  [0.0, -0.176776695, -0.731035345, 0.0]])

    def Equation_HCl(T, a):
        return (a[:, 0] +
                a[:, 1] * T +
                a[:, 2] / T)

    def Equation_HSO4(T, a):
        return (a[:, 0] + (T - 328.15) * 1E-3 *
                (a[:, 1] + (T - 328.15) *
                 ((a[:, 2] / 2) + (T - 328.15) *
                  (a[:, 3] / 6))))

    def Equation_HSO4_Clegg94(T, a):
        return (a[:, 0] + (T - 328.15) *
                (1E-3 * a[:, 1] + (T - 328.15) *
                 ((1e-3 * a[:, 2] / 2) +
                  (T - 328.15) * 1e-3 * a[:, 3] / 6)))

    ############################################################
    # beta_0, beta_1 and C_phi values arranged into arrays
    N_cations = 6  # H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    N_anions = 7  # OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

    beta_0 = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array
    beta_1 = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array
    C_phi = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array

    # H = cation
    # [beta_0[0, 0], beta_1[0, 0], C_phi[0, 0]] = n / a
    [beta_0[0, 1], beta_1[0, 1], C_phi[0, 1]] = Equation_HCl(T, param_HCl)
    # [beta_0[0, 2], beta_1[0, 2], C_phi[0, 2]] = n / a
    # [beta_0[0, 3], beta_1[0, 3], C_phi[0, 3]] = n / a
    # [beta_0[0, 4], beta_1[0, 4], C_phi[0, 4]] = n / a
    # [beta_0[0, 5], beta_1[0, 5], C_phi[0, 5]] = n / a
    # [beta_0[0, 6], beta_1[0, 6], C_phi[0, 6]] = Equation_HSO4(T, param_HSO4)
    # [beta_0[0, 6], beta_1[0, 6], C_phi[0, 6], C1_HSO4] = Equation_HSO4_Clegg94(T, param_HSO4_Clegg94)
    C1_HSO4 = 0
    # print beta_0[0, :], beta_1[0, :]#, beta_2[0, :]

    # Na = cation
    [beta_0[1, 0], beta_1[1, 0], C_phi[1, 0]] = Equation_NaOH
    [beta_0[1, 1], beta_1[1, 1], C_phi[1, 1]] = Equation_TabA1(T, Tinv, lnT, param_NaCl)
    [beta_0[1, 2], beta_1[1, 2], C_phi[1, 2]] = Equation_TabA3andTabA4andTabA5(T, param_NaBOH4)
    [beta_0[1, 3], beta_1[1, 3], C_phi[1, 3]] = Equation_TabA3andTabA4andTabA5(T, param_NaHCO3)
    [beta_0[1, 4], beta_1[1, 4], C_phi[1, 4]] = Equation_TabA3andTabA4andTabA5(T, param_NaHSO4)
    [beta_0[1, 5], beta_1[1, 5], C_phi[1, 5]] = Equation_TabA3andTabA4andTabA5(T, param_Na2CO3)
    [beta_0[1, 6], beta_1[1, 6], C_phi[1, 6]] = param_Na2SO4_Moller  # Equation_Na2SO4_TabA3(T, ln_of_Tdiv29815, param_Na2SO4)

    # K = cation
    [beta_0[2, 0], beta_1[2, 0], C_phi[2, 0]] = Equation_TabA7(T, param_KOH)
    [beta_0[2, 1], beta_1[2, 1], C_phi[2, 1]] = Equation_TabA1(T, Tinv, lnT, param_KCl)
    [beta_0[2, 2], beta_1[2, 2], C_phi[2, 2]] = Equation_TabA3andTabA4andTabA5(T, param_KBOH4)
    [beta_0[2, 3], beta_1[2, 3], C_phi[2, 3]] = Equation_TabA3andTabA4andTabA5(T, param_KHCO3)
    [beta_0[2, 4], beta_1[2, 4], C_phi[2, 4]] = Equation_KHSO4
    [beta_0[2, 5], beta_1[2, 5], C_phi[2, 5]] = Equation_TabA3andTabA4andTabA5(T, param_K2CO3)
    [beta_0[2, 6], beta_1[2, 6], C_phi[2, 6]] = Equation_TabA1(T, Tinv, lnT, param_K2SO4)

    # Mg = cation
    # [beta_0[3, 0], beta_1[3, 0], C_phi[3, 0]] = n / a
    [beta_0[3, 1], beta_1[3, 1], C_phi[3, 1]] = Equation1_TabA2(T, param_MgCl2)
    [beta_0[3, 2], beta_1[3, 2], C_phi[3, 2]] = Equation_TabA3andTabA4andTabA5_Simonson(T, param_MgBOH42)
    [beta_0[3, 3], beta_1[3, 3], C_phi[3, 3]] = Equation_MgHCO32
    [beta_0[3, 4], beta_1[3, 4], C_phi[3, 4]] = Equation_MgHSO42
    # [beta_0[3, 5], beta_1[3, 5], C_phi[3, 5]] = n / a
    [beta_0[3, 6], beta_1[3, 6], C_phi[3, 6]] = Equation2_TabA2(T, Tpower2, Tpower3, Tpower4, param_MgSO4)
    # print beta_0[3, 6], beta_1[3, 6], C_phi[3, 6]

    # Ca = cation
    [beta_0[4, 0], beta_1[4, 0], C_phi[4, 0]] = Equation_CaOH2
    [beta_0[4, 1], beta_1[4, 1], C_phi[4, 1]] = Equation_TabA1(T, Tinv, lnT, param_CaCl2)
    [beta_0[4, 2], beta_1[4, 2], C_phi[4, 2]] = Equation_TabA3andTabA4andTabA5_Simonson(T, param_CaBOH42)
    [beta_0[4, 3], beta_1[4, 3], C_phi[4, 3]] = Equation_CaHCO32
    [beta_0[4, 4], beta_1[4, 4], C_phi[4, 4]] = Equation_CaHSO42
    # [beta_0[4, 5], beta_1[4, 5], C_phi[4, 5]] = n / a
    [beta_0[4, 6], beta_1[4, 6], C_phi[4, 6]] = Equation_CaSO4_PnM74  # Equation_TabA1(T, Tinv, lnT, param_CaSO4)

    # Sr = cation
    [beta_0[5, 0], beta_1[5, 0], C_phi[5, 0]] = Equation_SrOH2
    [beta_0[5, 1], beta_1[5, 1], C_phi[5, 1]] = Equation_TabA7(T, param_SrCl2)
    [beta_0[5, 2], beta_1[5, 2], C_phi[5, 2]] = Equation_TabA3andTabA4andTabA5_Simonson(T, param_SrBOH42)
    [beta_0[5, 3], beta_1[5, 3], C_phi[5, 3]] = Equation_SrHCO32
    [beta_0[5, 4], beta_1[5, 4], C_phi[5, 4]] = Equation_SrHSO42
    # [beta_0[5, 5], beta_1[5, 5], C_phi[5, 5]] = n / a
    [beta_0[5, 6], beta_1[5, 6], C_phi[5, 6]] = Equation_CaSO4_PnM74  # Equation_TabA1(T, Tinv, lnT, param_SrSO4)

    # for 2:2 ion pairs beta_2 is needed
    beta_2 = np.zeros((N_cations, N_anions, *T.shape))
    b2_param_MgSO4 = np.array([-13.764, 0.12121, -2.7642e-4, 0, -0.21515, -32.743])

    def Eq_b2_MgSO4(T, Tpower2, Tpower3, Tpower4, q):
        return (q[0] * ((T / 2) + (88804) / (2 * T) - 298) +
                q[1] * ((Tpower2 / 6) + (26463592) / (3 * T) - (88804 / 2)) +
                q[2] * (Tpower3 / 12 + 88804 * 88804 / (4 * T) - 26463592 / 3) +
                q[3] * ((Tpower4 / 20) + 88804 * 26463592 / (5 * T) - 88804 * 88804 / 4) +
                q[4] * (298 - (88804 / T)) +
                q[5])

    b2_param_MgBOH42 = np.array([-11.47, 0.0, -3.24e-3])
    b2_param_CaBOH42 = np.array([-15.88, 0.0, -2.858e-3])

    def Eq_b2_MgANDCaBOH42(T, a):
        return a[0] + a[1] * (T - 298.15) + a[2] * (T - 303.15) * (T - 303.15)

    b2_param_CaSO4 = np.array([-55.7, 0])  # Pitzer and Mayorga74 # [-1.29399287e2, 4.00431027e-1]) Moller88

    def Eq_b2_CaSO4(T, a):
        return a[0] + a[1] * T

    beta_2[3, 6] = Eq_b2_MgSO4(T, Tpower2, Tpower3, Tpower4, b2_param_MgSO4)
    beta_2[3, 2] = Eq_b2_MgANDCaBOH42(T, b2_param_MgBOH42)
    beta_2[4, 2] = Eq_b2_MgANDCaBOH42(T, b2_param_CaBOH42)
    beta_2[4, 6] = Eq_b2_CaSO4(T, b2_param_CaSO4)
    beta_2[5, 2] = beta_2[4, 2]

    #############################################################################
    #############################################################################
    # Data and T - based calculations to create arrays holding Theta and Phi values
    # based on Table A10 and A11

    # Theta of positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    Theta_positive = np.zeros((6, 6, *T.shape))  # Array to hold Theta values between ion two ions (for numbering see list above)

    # H - Sr
    Theta_positive[0, 5] = 0.0591 + 4.5 * 1E-4 * (T - 298.15)
    Theta_positive[5, 0] = Theta_positive[0, 5]

    # H - Na
    Theta_positive[0, 1] = 0.03416 - 2.09 * 1E-4 * (T - 298.15)
    Theta_positive[1, 0] = Theta_positive[0, 1]

    # H - K
    Theta_positive[0, 2] = 0.005 - 2.275 * 1E-4 * (T - 298.15)
    Theta_positive[2, 0] = Theta_positive[0, 2]

    # H - Mg
    Theta_positive[0, 3] = 0.062 + 3.275 * 1E-4 * (T - 298.15)
    Theta_positive[3, 0] = Theta_positive[0, 3]

    # H - Ca
    Theta_positive[0, 4] = 0.0612 + 3.275 * 1E-4 * (T - 298.15)
    Theta_positive[4, 0] = Theta_positive[0, 4]

    # Na - K
    Theta_positive[1, 2] = -5.02312111E-2 + 14.0213141 / T
    Theta_positive[2, 1] = Theta_positive[1, 2]

    # Na - Mg
    Theta_positive[1, 3] = 0.07
    Theta_positive[3, 1] = 0.07

    # Na - Ca
    Theta_positive[1, 4] = 0.05
    Theta_positive[4, 1] = 0.05

    # K - Mg
    Theta_positive[2, 3] = 0.0
    Theta_positive[3, 2] = 0.0

    # K - Ca
    Theta_positive[2, 4] = 0.1156
    Theta_positive[4, 2] = 0.1156

    # Sr - Na
    Theta_positive[5, 1] = 0.07
    Theta_positive[1, 5] = 0.07

    # Sr - K
    Theta_positive[5, 2] = 0.01
    Theta_positive[2, 5] = 0.01

    # Mg - Ca
    Theta_positive[3, 4] = 0.007
    Theta_positive[4, 3] = 0.007
    # print 5.31274136 - 6.3424248e-3 * T - 9.83113847e2 / T, "ca - mg" #Spencer et al 1990

    # Theta of negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
    Theta_negative = np.zeros((7, 7, *T.shape))  # Array to hold Theta values between ion two ions (for numbering see list above)

    # Cl - SO4
    Theta_negative[1, 6] = 0.07
    Theta_negative[6, 1] = 0.07

    # Cl - CO3
    Theta_negative[1, 5] = -0.092  # corrected after Pitzer and Peiper 1982
    Theta_negative[5, 1] = -0.092  # corrected after Pitzer and Peiper 1982

    # Cl - HCO3
    Theta_negative[1, 3] = 0.0359
    Theta_negative[3, 1] = 0.0359

    # Cl - BOH4
    Theta_negative[1, 2] = -0.0323 - 0.42333 * 1E-4 * (T - 298.15) - 21.926 * 1E-6 * (T - 298.15) * (T - 298.15)
    Theta_negative[2, 1] = Theta_negative[1, 2]

    # CO3 - HCO3
    Theta_negative[3, 5] = 0.0
    Theta_negative[5, 3] = 0.0

    # SO4 - HSO4
    Theta_negative[4, 6] = 0.0
    Theta_negative[6, 4] = 0.0

    # OH - Cl
    Theta_negative[0, 1] = -0.05 + 3.125 * 1E-4 * (T - 298.15) - 8.362 * 1E-6 * (T - 298.15) * (T - 298.15)
    Theta_negative[1, 0] = Theta_negative[0, 1]

    # SO4 - CO3
    Theta_negative[5, 6] = 0.02
    Theta_negative[6, 5] = 0.02

    # SO4 - HCO3
    Theta_negative[3, 6] = 0.01
    Theta_negative[6, 3] = 0.01

    # SO4 - BOH4
    Theta_negative[2, 6] = -0.012
    Theta_negative[6, 2] = -0.012

    # HSO4 - Cl
    Theta_negative[1, 4] = -0.006
    Theta_negative[4, 1] = -0.006

    # OH - SO4
    Theta_negative[0, 6] = -0.013
    Theta_negative[6, 0] = -0.013

    # CO3 - OH #http: / /www.aim.env.uea.ac.uk / aim / accent4 / parameters.html
    Theta_negative[3, 0] = 0.1
    Theta_negative[0, 3] = 0.1

    # Phi
    # positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    # negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

    # Phi_PPN holds the values for cation - cation - anion
    Phi_PPN = np.zeros((6, 6, 7, *T.shape))  # Array to hold Theta values between ion two ions (for numbering see list above)

    # Na - K-Cl
    Phi_PPN[1, 2, 1] = 1.34211308E-2 - 5.10212917 / T
    Phi_PPN[2, 1, 1] = Phi_PPN[1, 2, 1]

    # Na - K-SO4
    Phi_PPN[1, 2, 6] = 3.48115174E-2 - 8.21656777 / T
    Phi_PPN[2, 1, 6] = Phi_PPN[1, 2, 6]

    # Na - Mg - Cl
    Phi_PPN[1, 3, 1] = 0.0199 - 9.51 / T
    Phi_PPN[3, 1, 1] = Phi_PPN[1, 3, 1]

    # Na - Ca - Cl
    Phi_PPN[1, 4, 1] = -7.6398 - 1.2990e-2 * T + 1.1060e-5 * T**2 + 1.8475 * lnT  # Spencer et al 1990 # -0.003
    Phi_PPN[4, 1, 1] = Phi_PPN[1, 4, 1]
    # print -7.6398 -1.2990e-2 * T + 1.1060e-5 * T*T + 1.8475 * lnT

    # Na - Ca - SO4
    Phi_PPN[1, 4, 6] = -0.012
    Phi_PPN[4, 1, 6] = Phi_PPN[1, 4, 6]

    # K - Mg - Cl
    Phi_PPN[2, 3, 1] = 0.02586 - 14.27 / T
    Phi_PPN[3, 2, 1] = Phi_PPN[2, 3, 1]

    # K - Ca - Cl
    Phi_PPN[2, 4, 1] = 0.047627877 - 27.0770507 / T
    Phi_PPN[4, 2, 1] = Phi_PPN[2, 4, 1]

    # K - Ca - SO4
    Phi_PPN[2, 4, 6] = 0.0
    Phi_PPN[4, 2, 6] = 0.0

    # H - Sr - Cl
    Phi_PPN[0, 5, 1] = 0.0054 - 2.1 * 1E-4 * (T - 298.15)
    Phi_PPN[5, 0, 1] = Phi_PPN[0, 5, 1]

    # H - Mg - Cl
    Phi_PPN[0, 3, 1] = 0.001 - 7.325 * 1E-4 * (T - 298.15)
    Phi_PPN[3, 0, 1] = Phi_PPN[0, 3, 1]

    # H - Ca - Cl
    Phi_PPN[0, 4, 1] = 0.0008 - 7.25 * 1E-4 * (T - 298.15)
    Phi_PPN[4, 0, 1] = Phi_PPN[0, 4, 1]

    # Sr - Na - Cl
    Phi_PPN[5, 1, 1] = -0.015
    Phi_PPN[1, 5, 1] = -0.015

    # Sr - K-Cl
    Phi_PPN[5, 2, 1] = -0.015
    Phi_PPN[2, 5, 1] = -0.015

    # Na - Mg - SO4
    Phi_PPN[1, 3, 6] = -0.015
    Phi_PPN[3, 1, 6] = -0.015

    # K - Mg - SO4
    Phi_PPN[2, 3, 6] = -0.048
    Phi_PPN[3, 2, 6] = -0.048

    # Mg - Ca - Cl
    Phi_PPN[3, 4, 1] = 4.15790220e1 + 1.30377312e-2 * T - 9.81658526e2 / T - 7.4061986 * lnT  # Spencer et al 1990 # - 0.012
    Phi_PPN[4, 3, 1] = Phi_PPN[3, 4, 1]
    # print 4.15790220e1 + 1.30377312e-2 * T -9.81658526e2 / T -7.4061986 * lnT

    # Mg - Ca - SO4
    Phi_PPN[3, 4, 6] = 0.024
    Phi_PPN[4, 3, 6] = 0.024

    # H - Na - Cl
    Phi_PPN[0, 1, 1] = 0.0002
    Phi_PPN[1, 0, 1] = 0.0002

    # H - Na - SO4
    Phi_PPN[0, 1, 6] = 0.0
    Phi_PPN[1, 0, 6] = 0.0

    # H - K-Cl
    Phi_PPN[0, 2, 1] = -0.011
    Phi_PPN[2, 0, 1] = -0.011

    # H - K-SO4
    Phi_PPN[0, 2, 1] = 0.197
    Phi_PPN[2, 0, 1] = 0.197

    # Phi_PPN holds the values for anion - anion - cation
    Phi_NNP = np.zeros((7, 7, 6, *T.shape))  # Array to hold Theta values between ion two ions (for numbering see list above)

    # Cl - SO4 - Na
    Phi_NNP[1, 6, 1] = -0.009
    Phi_NNP[6, 1, 1] = -0.009

    # Cl - SO4 - K
    Phi_NNP[1, 6, 2] = -0.21248147 + 37.5619614 / T + 2.8469833 * 1E-3 * T
    Phi_NNP[6, 1, 2] = Phi_NNP[1, 6, 2]

    # Cl - SO4 - Ca
    Phi_NNP[1, 6, 4] = -0.018
    Phi_NNP[6, 1, 4] = -0.018

    # Cl - CO3 - Ca
    Phi_NNP[1, 5, 4] = 0.016
    Phi_NNP[5, 1, 4] = 0.016

    # Cl - HCO3 - Na
    Phi_NNP[1, 3, 1] = -0.0143
    Phi_NNP[3, 1, 1] = -0.0143

    # Cl - BOH4 - Na
    Phi_NNP[1, 2, 1] = -0.0132
    Phi_NNP[2, 1, 1] = -0.0132

    # Cl - BOH4 - Mg
    Phi_NNP[1, 2, 3] = -0.235
    Phi_NNP[2, 1, 3] = -0.235

    # Cl - BOH4 - Ca
    Phi_NNP[1, 2, 4] = -0.8
    Phi_NNP[2, 1, 4] = -0.8

    # HSO4 - SO4 - Na
    Phi_NNP[4, 6, 1] = 0.0
    Phi_NNP[6, 4, 1] = 0.0

    # CO3 - HCO3 - Na
    Phi_NNP[3, 5, 1] = 0.0
    Phi_NNP[5, 3, 1] = 0.0

    # CO3 - HCO3 - K
    Phi_NNP[3, 5, 2] = 0.0
    Phi_NNP[5, 3, 2] = 0.0

    # Cl - SO4 - Mg
    Phi_NNP[1, 6, 3] = -0.004
    Phi_NNP[6, 1, 3] = -0.004

    # Cl - HCO3 - Mg
    Phi_NNP[1, 3, 3] = -0.0196
    Phi_NNP[3, 1, 3] = -0.0196

    # SO4 - CO3 - Na
    Phi_NNP[6, 5, 1] = -0.005
    Phi_NNP[5, 6, 1] = -0.005

    # SO4 - CO3 - K
    Phi_NNP[6, 5, 2] = -0.009
    Phi_NNP[5, 6, 2] = -0.009

    # SO4 - HCO3 - Na
    Phi_NNP[6, 3, 1] = -0.005
    Phi_NNP[3, 6, 1] = -0.005

    # SO4 - HCO3 - Mg
    Phi_NNP[6, 3, 3] = -0.161
    Phi_NNP[3, 6, 3] = -0.161

    # HSO4 - Cl - Na
    Phi_NNP[4, 1, 1] = -0.006
    Phi_NNP[1, 4, 1] = -0.006

    # HSO4 - SO4 - K
    Phi_NNP[4, 6, 2] = -0.0677
    Phi_NNP[6, 4, 2] = -0.0677

    # OH - Cl - Na
    Phi_NNP[0, 1, 1] = -0.006
    Phi_NNP[1, 0, 1] = -0.006

    # OH - Cl - K
    Phi_NNP[0, 1, 2] = -0.006
    Phi_NNP[1, 0, 2] = -0.006

    # OH - Cl - Ca
    Phi_NNP[0, 1, 4] = -0.025
    Phi_NNP[1, 0, 4] = -0.025

    # OH - SO4 - Na
    Phi_NNP[0, 6, 1] = -0.009
    Phi_NNP[6, 0, 1] = -0.009

    # OH - SO4 - K
    Phi_NNP[0, 6, 2] = -0.05
    Phi_NNP[6, 0, 2] = -0.05

    return beta_0, beta_1, beta_2, C_phi, Theta_negative, Theta_positive, Phi_NNP, Phi_PPN, C1_HSO4


# Functions from K_HSO4_thermo.py
# --------------------------------------
def supplyKHSO4(T, I):
    """
    Calculate KHSO4 for given temperature and salinity
    """
    I = pow(I, 1)
    # param_HSO4 = np.array([562.69486, -13273.75, -102.5154, 0.2477538, -1.117033e-4]) #Clegg et al. 1994
    # K_HSO4 = np.power(10,param_HSO4[0] + param_HSO4[1]/T + param_HSO4[2]*np.log(T) + param_HSO4[3]*T + param_HSO4[4]*T*T)

    param_HSO4 = np.array([141.411, -4340.704, -23.4825, 0.016637])  # Campbell et al. 1993
    # param_HSO4 = np.array([141.328, -4276.1, -23.093, 0]) #Dickson 1990
    # param_HSO4 = np.array([141.411, -4340.704, -23.4825, 0.016637])
    K_HSO4 = np.power(10, (param_HSO4[0] +
                           param_HSO4[1] / T +
                           param_HSO4[2] * np.log(T) +
                           param_HSO4[3] * T))

    param_HSO4_cond = np.array([141.328, -4276.1, -23.093, 324.57, -13856, -47.986, -771.54, 35474, 114.723, -2698, 1776])  # Dickson 1990
    K_HSO4_cond = np.exp(param_HSO4_cond[0] +
                         param_HSO4_cond[1] / T +
                         param_HSO4_cond[2] * np.log(T) + np.sqrt(I) *
                         (param_HSO4_cond[3] +
                          param_HSO4_cond[4] / T +
                          param_HSO4_cond[5] * np.log(T)) + I *
                         (param_HSO4_cond[6] +
                          param_HSO4_cond[7] / T +
                          param_HSO4_cond[8] * np.log(T)) +
                         param_HSO4_cond[9] / T * I * np.sqrt(I) +
                         param_HSO4_cond[10] / T * I * I)

    return [K_HSO4_cond, K_HSO4]


# Functions from K_HF_cond.py
# --------------------------------------
def supplyKHF(T, sqrtI):
    return np.exp(1590.2 / T - 12.641 + 1.525 * sqrtI)


# Functions from gammaANDalpha.py
# --------------------------------------
def CalculateGammaAndAlphas(Tc, S, I, m_cation, m_anion):
    # Testbed case T=25C, I=0.7, seawatercomposition
    T = Tc + 273.15
    sqrtI = np.sqrt(I)

    Z_cation = np.zeros((6, 1, 1))
    Z_cation[0] = 1
    Z_cation[1] = 1
    Z_cation[2] = 1
    Z_cation[3] = 2
    Z_cation[4] = 2
    Z_cation[5] = 2

    Z_anion = np.zeros((7, 1, 1))
    Z_anion[0] = -1
    Z_anion[1] = -1
    Z_anion[2] = -1
    Z_anion[3] = -1
    Z_anion[4] = -1
    Z_anion[5] = -2
    Z_anion[6] = -2

    ##########################################################################
    [beta_0, beta_1, beta_2, C_phi, Theta_negative, Theta_positive,
        Phi_NNP, Phi_PPN, C1_HSO4] = SupplyParams(T)

    A_phi = 3.36901532E-01 - 6.32100430E-04 * T + 9.14252359 / T - 1.35143986E-02 * np.log(T) + 2.26089488E-03 / (
        T - 263) + 1.92118597E-6 * T * T + 4.52586464E+01 / (680 - T)  # note correction of last parameter, E + 1 instead of E-1
    # A_phi = 8.66836498e1 + 8.48795942e-2 * T - 8.88785150e-5 * T * T +
    # 4.88096393e-8 * T * T * T -1.32731477e3 / T - 1.76460172e1 * np.log(T)
    # # Spencer et al 1990

    f_gamma = -A_phi * (sqrtI / (1 + 1.2 * sqrtI) +
                        (2 / 1.2) * np.log(1 + 1.2 * sqrtI))

    # E_cat = sum(m_cation * Z_cation)
    E_an = -sum(m_anion * Z_anion)
    E_cat = -E_an

    # BMX_phi
    BMX_phi = np.zeros((6, 7, *Tc.shape))
    BMX = np.zeros((6, 7, *Tc.shape))
    BMX_apostroph = np.zeros((6, 7, *Tc.shape))
    CMX = np.zeros((6, 7, *Tc.shape))

    for cat in range(0, 6):
        for an in range(0, 7):
            BMX_phi[cat, an] = beta_0[cat, an] + \
                beta_1[cat, an] * np.exp(-2 * sqrtI)
            BMX[cat, an] = beta_0[cat, an] + \
                (beta_1[cat, an] / (2 * I)) * \
                (1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI))
            BMX_apostroph[cat, an] = (beta_1[cat, an] / (2 * I * I)) * \
                (-1 + (1 + (2 * sqrtI) + (2 * sqrtI)) * np.exp(-2 * sqrtI))
            CMX[cat, an] = C_phi[cat, an] / \
                (2 * np.sqrt(-Z_anion[an] * Z_cation[cat]))

    # BMX* and CMX are calculated differently for 2:2 ion pairs, corrections
    # below  # § alpha2= 6 for borates ... see Simonson et al 1988
    cat = 3
    an = 2  # MgBOH42
    BMX_phi[cat, an] = beta_0[cat, an] + beta_1[cat, an] * \
        np.exp(-1.4 * sqrtI) + beta_2[cat, an] * np.exp(-6 * sqrtI)
    BMX[cat, an] = beta_0[cat, an] + (beta_1[cat, an] / (0.98 * I)) * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI)) + (
        beta_2[cat, an] / (18 * I)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I * I)) * (-1 + (1 + 1.4 * sqrtI + 0.98 * I) *
                                                                   np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (18 * I)) * (-1 - (1 + 6 * sqrtI + 18 * I) * np.exp(-6 * sqrtI))
    cat = 3
    an = 6  # MgSO4
    BMX_phi[cat, an] = beta_0[cat, an] + beta_1[cat, an] * \
        np.exp(-1.4 * sqrtI) + beta_2[cat, an] * np.exp(-12 * sqrtI)
    BMX[cat, an] = beta_0[cat, an] + (beta_1[cat, an] / (0.98 * I)) * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI)) + (
        beta_2[cat, an] / (72 * I)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I * I)) * (-1 + (1 + 1.4 * sqrtI + 0.98 * I) * np.exp(-1.4 * sqrtI)
                                                                   ) + (beta_2[cat, an] / (72 * I * I)) * (-1 - (1 + 12 * sqrtI + 72 * I) * np.exp(-12 * sqrtI))
    # BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I)) * (-1 + (1 + 1.4
    # * sqrtI + 0.98 * I) * np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (72 *
    # I)) * (-1-(1 + 12 * sqrtI + 72 * I) * np.exp(-12 * sqrtI)) # not 1 /
    # (0.98 * I * I) ... compare M&P98 equation A17 with Pabalan and Pitzer
    # 1987 equation 15c / 16b
    cat = 4
    an = 2  # CaBOH42
    BMX_phi[cat, an] = beta_0[cat, an] + beta_1[cat, an] * \
        np.exp(-1.4 * sqrtI) + beta_2[cat, an] * np.exp(-6 * sqrtI)
    BMX[cat, an] = beta_0[cat, an] + (beta_1[cat, an] / (0.98 * I)) * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI)) + (
        beta_2[cat, an] / (18 * I)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I * I)) * (-1 + (1 + 1.4 * sqrtI + 0.98 * I) *
                                                                   np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (18 * I)) * (-1 - (1 + 6 * sqrtI + 18 * I) * np.exp(-6 * sqrtI))
    cat = 4
    an = 6  # CaSO4
    BMX_phi[cat, an] = beta_0[cat, an] + beta_1[cat, an] * \
        np.exp(-1.4 * sqrtI) + beta_2[cat, an] * np.exp(-12 * sqrtI)
    BMX[cat, an] = beta_0[cat, an] + (beta_1[cat, an] / (0.98 * I)) * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI)) + (
        beta_2[cat, an] / (72 * I)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I * I)) * (-1 + (1 + 1.4 * sqrtI + 0.98 * I) * np.exp(-1.4 * sqrtI)
                                                                   ) + (beta_2[cat, an] / (72 * I)) * (-1 - (1 + 12 * sqrtI + 72 * I) * np.exp(-12 * sqrtI))

    cat = 5
    an = 2  # SrBOH42
    BMX_phi[cat, an] = beta_0[cat, an] + beta_1[cat, an] * \
        np.exp(-1.4 * sqrtI) + beta_2[cat, an] * np.exp(-6 * sqrtI)
    BMX[cat, an] = beta_0[cat, an] + (beta_1[cat, an] / (0.98 * I)) * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI)) + (
        beta_2[cat, an] / (18 * I)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * I * I)) * (-1 + (1 + 1.4 * sqrtI + 0.98 * I) *
                                                                   np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (18 * I)) * (-1 - (1 + 6 * sqrtI + 18 * I) * np.exp(-6 * sqrtI))

    # BMX* is calculated with T-dependent alpha for H-SO4; see Clegg et al.,
    # 1994 --- Millero and Pierrot are completly off for this ion pair
    xClegg = (2 - 1842.843 * (1 / T - 1 / 298.15)) * sqrtI
    # xClegg = (2) * sqrtI
    gClegg = 2 * (1 - (1 + xClegg) * np.exp(-xClegg)) / (xClegg * xClegg)
    # alpha = (2 - 1842.843 * (1 / T - 1 / 298.15)) see Table 6 in Clegg et al
    # 1994
    BMX[0, 6] = beta_0[0, 6] + beta_1[0, 6] * gClegg
    BMX_apostroph[0, 6] = beta_1[0, 6] / I * (np.exp(-xClegg) - gClegg)

    CMX[0, 6] = C_phi[0, 6] + 4 * C1_HSO4 * (6 - (6 + 2.5 * sqrtI * (6 + 3 * 2.5 * sqrtI + 2.5 * sqrtI * 2.5 * sqrtI)) *
                                             np.exp(-2.5 * sqrtI)) / (2.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI)  # w = 2.5 ... see Clegg et al., 1994

    # unusual alpha=1.7 for Na2SO4
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (2.89 * I)) * 2 * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (1.7 * I)) * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))

    # BMX[4, 6] =BMX[4, 6] * 0  # knock out Ca-SO4

    R = 0
    S = 0
    for cat in range(0, 6):
        for an in range(0, 7):
            R = R + m_anion[an] * m_cation[cat] * BMX_apostroph[cat, an]
            S = S + m_anion[an] * m_cation[cat] * CMX[cat, an]

    gamma_anion = np.zeros((7, *Tc.shape))
    ln_gamma_anion = np.zeros((7, *Tc.shape))
    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S
    # print (np.exp(ln_gammaCl), ln_gammaCl)
    XX = 99

    for an in range(0, 7):
        ln_gamma_anion[an] = Z_anion[an] * \
            Z_anion[an] * (f_gamma + R) + Z_anion[an] * S
        if an == XX:
            print (ln_gamma_anion[an], "init")
        for cat in range(0, 6):
            ln_gamma_anion[an] = ln_gamma_anion[an] + 2 * \
                m_cation[cat] * (BMX[cat, an] + E_cat * CMX[cat, an])
            if an == XX:
                print (ln_gamma_anion[an], cat)
        for an2 in range(0, 7):
            ln_gamma_anion[an] = ln_gamma_anion[an] + \
                m_anion[an2] * (2 * Theta_negative[an, an2])
            if an == XX:
                print (ln_gamma_anion[an], an2)
        for an2 in range(0, 7):
            for cat in range(0, 6):
                ln_gamma_anion[an] = ln_gamma_anion[an] + \
                    m_anion[an2] * m_cation[cat] * Phi_NNP[an, an2, cat]
                if an == XX:
                    print (ln_gamma_anion[an], an2, cat)
        for cat in range(0, 6):
            for cat2 in range(cat + 1, 6):
                ln_gamma_anion[an] = ln_gamma_anion[an] + m_cation[cat] * m_cation[cat2] * Phi_PPN[cat, cat2, an]
                if an == XX:
                    print (ln_gamma_anion[an], cat, cat2)

    gamma_cation = np.zeros((6, *Tc.shape))
    ln_gamma_cation = np.zeros((6, *Tc.shape))
    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S
    # print (np.exp(ln_gammaCl), ln_gammaCl)
    XX = 99
    for cat in range(0, 6):
        ln_gamma_cation[cat] = Z_cation[cat] * \
            Z_cation[cat] * (f_gamma + R) + Z_cation[cat] * S
        if cat == XX:
            print (ln_gamma_cation[cat], "init")
        for an in range(0, 7):
            ln_gamma_cation[cat] = ln_gamma_cation[cat] + 2 * \
                m_anion[an] * (BMX[cat, an] + E_cat * CMX[cat, an])
            if cat == XX:
                print (ln_gamma_cation[cat], an, BMX[cat, an], E_cat * CMX[cat, an])
        for cat2 in range(0, 6):
            ln_gamma_cation[cat] = ln_gamma_cation[cat] + \
                m_cation[cat2] * (2 * Theta_positive[cat, cat2])
            if cat == XX:
                print (ln_gamma_cation[cat], cat2)
        for cat2 in range(0, 6):
            for an in range(0, 7):
                ln_gamma_cation[cat] = ln_gamma_cation[cat] + m_cation[cat2] * m_anion[an] * Phi_PPN[cat, cat2, an]
                if cat == XX:
                    print (ln_gamma_cation[cat], cat2, an)
        for an in range(0, 7):
            for an2 in range(an + 1, 7):
                ln_gamma_cation[cat] = ln_gamma_cation[cat] + \
                    m_anion[an] * m_anion[an2] * Phi_NNP[an, an2, cat]
                if cat == XX:
                    print (ln_gamma_cation[cat], an, an2)

    gamma_anion = np.exp(ln_gamma_anion)
    gamma_cation = np.exp(ln_gamma_cation)

    # choice of pH-scale = total pH-scale [H]T = [H]F + [HSO4]
    # so far gamma_H is the [H]F activity coefficient (= free-H pH-scale)
    # thus, conversion is required
    # * (gamma_anion[4] / gamma_anion[6] / gamma_cation[0])
    [K_HSO4_conditional, K_HSO4] = supplyKHSO4(T, I)
    # print (K_HSO4_conditional)
    # print (gamma_anion[4], gamma_anion[6], gamma_cation[0])
    # alpha_H = 1 / (1+ m_anion[6] / K_HSO4_conditional + 0.0000683 / (7.7896E-4 * 1.1 / 0.3 / gamma_cation[0]))
    alpha_Hsws = 1 / (1 + m_anion[6] / K_HSO4_conditional +
                      0.0000683 / (supplyKHF(T, sqrtI)))
    alpha_Ht = 1 / (1 + m_anion[6] / K_HSO4_conditional)
    # alpha_H = 1 / (1+ m_anion[6] / K_HSO4_conditional)

    # A number of ion pairs are calculated explicitly: MgOH, CaCO3, MgCO3, SrCO3
    # since OH and CO3 are rare compared to the anions the anion alpha (free /
    # total) are assumed to be unity
    gamma_MgCO3 = 1
    gamma_CaCO3 = gamma_MgCO3
    gamma_SrCO3 = gamma_MgCO3

    b0b1CPhi_MgOH = np.array([-0.1, 1.658, 0, 0.028])
    BMX_MgOH = b0b1CPhi_MgOH[
        0] + (b0b1CPhi_MgOH[1] / (2 * I)) * (1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI))
    ln_gamma_MgOH = 1 * (f_gamma + R) + (1) * S
    # interaction between MgOH-Cl affects MgOH gamma
    ln_gamma_MgOH = ln_gamma_MgOH + 2 * \
        m_anion[1] * (BMX_MgOH + E_cat * b0b1CPhi_MgOH[2])
    # interaction between MgOH-Mg-OH affects MgOH gamma
    ln_gamma_MgOH = ln_gamma_MgOH + m_cation[3] * m_anion[1] * b0b1CPhi_MgOH[3]
    gamma_MgOH = np.exp(ln_gamma_MgOH)

    K_MgOH = np.power(10, -(3.87 - 501.6 / T)) / \
        (gamma_cation[3] * gamma_anion[0] / gamma_MgOH)
    K_MgCO3 = np.power(10, -(1.028 + 0.0066154 * T)) / \
        (gamma_cation[3] * gamma_anion[5] / gamma_MgCO3)
    K_CaCO3 = np.power(10, -(1.178 + 0.0066154 * T)) / \
        (gamma_cation[4] * gamma_anion[5] / gamma_CaCO3)
    # K_CaCO3 = np.power(10, (-1228.732 - 0.299444 * T + 35512.75 / T +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] / gamma_CaCO3) # Plummer and Busenberg82
    # K_MgCO3 = np.power(10, (-1228.732 +(0.15) - 0.299444 * T + 35512.75 / T
    # +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] /
    # gamma_CaCO3)# Plummer and Busenberg82
    K_SrCO3 = np.power(10, -(1.028 + 0.0066154 * T)) / \
        (gamma_cation[5] * gamma_anion[5] / gamma_SrCO3)

    alpha_OH = 1 / (1 + (m_cation[3] / K_MgOH))
    alpha_CO3 = 1 / (1 + (m_cation[3] / K_MgCO3) +
                     (m_cation[4] / K_CaCO3) + (m_cation[5] / K_SrCO3))

    return gamma_cation, gamma_anion, alpha_Hsws, alpha_Ht, alpha_OH, alpha_CO3


# Functions from gammaNeutral.py
# --------------------------------------
def gammaCO2_fn(Tc, m_an, m_cat):
    T = Tc + 273.15
    lnT = np.log(T)

    m_ion = np.array([m_cat[0], m_cat[1], m_cat[2], m_cat[3], m_cat[4], m_an[1], m_an[6]])

    param_lamdaCO2 = np.zeros([7, 5])
    param_lamdaCO2[0, :] = [0, 0, 0, 0, 0]  # H
    param_lamdaCO2[1, :] = [-5496.38465, -3.326566, 0.0017532, 109399.341, 1047.021567]  # Na
    param_lamdaCO2[2, :] = [2856.528099, 1.7670079, -0.0009487, -55954.1929, -546.074467]  # K
    param_lamdaCO2[3, :] = [-479.362533, -0.541843, 0.00038812, 3589.474052, 104.3452732]  # Mg
    # param_lamdaCO2[3, :] = [9.03662673e+03, 5.08294701e+00, -2.51623005e-03, -1.88589243e+05, -1.70171838e+03]  # Mg refitted
    param_lamdaCO2[4, :] = [-12774.6472, -8.101555, 0.00442472, 245541.5435, 2452.50972]  # Ca
    # param_lamdaCO2[4, :] = [-8.78153999e+03, -5.67606538e+00, 3.14744317e-03, 1.66634223e+05, 1.69112982e+03]  # Ca refitted
    param_lamdaCO2[5, :] = [1659.944942, 0.9964326, -0.00052122, -33159.6177, -315.827883]  # Cl
    param_lamdaCO2[6, :] = [2274.656591, 1.8270948, -0.00114272, -33927.7625, -457.015738]  # SO4

    param_zetaCO2 = np.zeros([2, 6, 5])
    param_zetaCO2[0, 0, :] = [-804.121738, -0.470474, 0.000240526, 16334.38917, 152.3838752]  # Cl & H
    param_zetaCO2[0, 1, :] = [-379.459185, -0.258005, 0.000147823, 6879.030871, 73.74511574]  # Cl & Na
    param_zetaCO2[0, 2, :] = [-379.686097, -0.257891, 0.000147333, 6853.264129, 73.79977116]  # Cl & K
    param_zetaCO2[0, 3, :] = [-1342.60256, -0.772286, 0.000391603, 27726.80974, 253.62319406]  # Cl & Mg
    param_zetaCO2[0, 4, :] = [-166.06529, -0.018002, -0.0000247349, 5256.844332, 27.377452415]  # Cl & Ca
    param_zetaCO2[1, 1, :] = [67030.02482, 37.930519, -0.0189473, -1399082.37, -12630.27457]  # SO4 & Na
    param_zetaCO2[1, 2, :] = [-2907.03326, -2.860763, 0.001951086, 30756.86749, 611.37560512]  # SO4 & K
    param_zetaCO2[1, 3, :] = [-7374.24392, -4.608331, 0.002489207, 143162.6076, 1412.302898]  # SO4 & Mg

    lamdaCO2 = np.zeros((7, *Tc.shape))
    for ion in range(0, 7):
        lamdaCO2[ion] = param_lamdaCO2[ion, 0] + param_lamdaCO2[ion, 1] * T + param_lamdaCO2[ion, 2] * T**2 + param_lamdaCO2[ion, 3] / T + param_lamdaCO2[ion, 4] * lnT

    zetaCO2 = np.zeros([2, 5, *Tc.shape])
    for ion in range(0, 5):
        zetaCO2[0, ion] = param_zetaCO2[0, ion, 0] + param_zetaCO2[0, ion, 1] * T + param_zetaCO2[0, ion, 2] * T**2 + param_zetaCO2[0, ion, 3] / T + param_zetaCO2[0, ion, 4] * lnT
    for ion in range(1, 4):
        zetaCO2[1, ion] = param_zetaCO2[1, ion, 0] + param_zetaCO2[1, ion, 1] * T + param_zetaCO2[1, ion, 2] * T**2 + param_zetaCO2[1, ion, 3] / T + param_zetaCO2[1, ion, 4] * lnT

    ln_gammaCO2 = 0
    for ion in range(0, 7):
        ln_gammaCO2 = ln_gammaCO2 + m_ion[ion] * 2 * lamdaCO2[ion]
    # for cat in range(0, 5):
    # ln_gammaCO2 = ln_gammaCO2 + m_ion[5] * m_ion[cat] * zetaCO2[0, cat] + m_ion[6] * m_ion[cat] * zetaCO2[1, cat]

    gammaCO2 = np.exp(ln_gammaCO2)  # as according to He and Morse 1993
    # gammaCO2 = np.power(10, ln_gammaCO2) # pK1 is "correct if log-base 10 is assumed

    gammaCO2gas = np.exp(1 / (8.314462175 * T * (0.10476 - 61.0102 / T - 660000 / T / T / T - 2.47E27 / np.power(T, 12))))

    ##########################
    # CALCULATION OF gammaB

    lamdaB = np.array([0, -0.097, -0.14, 0, 0, 0.091, 0.018])  # Felmy and Wear 1986
    # lamdaB = np.array([0.109, 0.028, -0.026, 0.191, 0.165, 0, -0.205]) #Chanson and Millero 2006
    ln_gammaB = m_ion[1] * m_ion[6] * 0.046  # tripple ion interaction Na-SO4
    for ion in range(0, 7):
        ln_gammaB = ln_gammaB + m_ion[ion] * 2 * lamdaB[ion]

    gammaB = np.exp(ln_gammaB)  # as according to Felmy and Wear 1986
    # print gammaB

    return gammaCO2, gammaCO2gas, gammaB


# Functions from pKs.py
# --------------------------------------
def calculate_gKs(Tc, Sal, mCa, mMg):
    I = 19.924 * Sal / (1000 - 1.005 * Sal)

    m_cation = np.zeros((6, *Tc.shape))
    m_cation[0] = (0.00000001 * Sal / 35.)  # H ion; pH of about 8
    # Na Millero et al., 2008; Dickson OA-guide
    m_cation[1] = (0.4689674 * Sal / 35.)
    # K Millero et al., 2008; Dickson OA-guide
    m_cation[2] = (0.0102077 * Sal / 35.)
    m_cation[3] = (mMg * Sal / 35.)  # Mg Millero et al., 2008; Dickson OA-guide
    m_cation[4] = (mCa * Sal / 35.)  # Ca Millero et al., 2008; Dickson OA-guide
    # Sr Millero et al., 2008; Dickson OA-guide
    m_cation[5] = (0.0000907 * Sal / 35.)

    m_anion = np.zeros((7, *Tc.shape))
    m_anion[0] = (0.0000010 * Sal / 35.)  # OH ion; pH of about 8
    # Cl Millero et al., 2008; Dickson OA-guide
    m_anion[1] = (0.5458696 * Sal / 35.)
    # BOH4 Millero et al., 2008; Dickson OA-guide; pH of about 8 -- borate,
    # not Btotal
    m_anion[2] = (0.0001008 * Sal / 35.)
    # HCO3 Millero et al., 2008; Dickson OA-guide
    m_anion[3] = (0.0017177 * Sal / 35.)
    # HSO4 Millero et al., 2008; Dickson OA-guide
    m_anion[4] = (0.0282352 * 1e-6 * Sal / 35.)
    # CO3 Millero et al., 2008; Dickson OA-guide
    m_anion[5] = (0.0002390 * Sal / 35.)
    # SO4 Millero et al., 2008; Dickson OA-guide
    m_anion[6] = (0.0282352 * Sal / 35.)

    [gamma_cation, gamma_anion, alpha_Hsws, alpha_Ht, alpha_OH,
        alpha_CO3] = CalculateGammaAndAlphas(Tc, Sal, I, m_cation, m_anion)

    gammaT_OH = gamma_anion[0] * alpha_OH
    gammaT_BOH4 = gamma_anion[2]
    gammaT_HCO3 = gamma_anion[3]
    gammaT_CO3 = gamma_anion[5] * alpha_CO3

    # gammaT_Hsws = gamma_cation[0] * alpha_Hsws
    gammaT_Ht = gamma_cation[0] * alpha_Ht
    gammaT_Ca = gamma_cation[4]

    [gammaCO2, gammaCO2gas, gammaB] = gammaCO2_fn(Tc, m_anion, m_cation)

    gKspC = 1 / gammaT_Ca / gammaT_CO3
    gKspA = 1 / gammaT_Ca / gammaT_CO3
    gK1 = 1 / gammaT_Ht / gammaT_HCO3 * gammaCO2
    gK2 = 1 / gammaT_Ht / gammaT_CO3 * gammaT_HCO3
    gKW = 1 / gammaT_Ht / gammaT_OH
    gKB = 1 / gammaT_BOH4 / gammaT_Ht * (gammaB)
    gK0 = 1 / gammaCO2 * gammaCO2gas
    gKHSO4 = 1 / gamma_anion[6] / gammaT_Ht * gamma_anion[4]

    return gKspC, gK1, gK2, gKW, gKB, gKspA, gK0, gKHSO4


# Helper functions
# ----------------
def reshaper(orig, target):
    """
    Adds additional dimensions to orig so it can be broadcast on target.
    """
    on = orig.copy()
    while on.ndim < (target.ndim + orig.ndim):
        on = np.expand_dims(on, -1)
    return on


start_params = {'K0': np.array([-60.2409, 93.4517, 23.3585, 0.023517,
                                -0.023656, 0.0047036]),
                'K1': np.array([61.2172, -3633.86, -9.6777,
                                0.011555, -0.0001152]),
                'K2': np.array([-25.9290, -471.78, 3.16967,
                                0.01781, -0.0001122]),
                'KB': np.array([148.0248, 137.1942, 1.62142, -8966.90, -2890.53, -77.942, 1.728, -0.0996, -24.4344, -25.085, -0.2474, 0.053105]),
                'KW': np.array([148.9652, -13847.26, -23.6521, 118.67,
                                -5.977, 1.0495, -0.01615]),
                'KspC': np.array([-171.9065, -0.077993, 2839.319, 71.595, -0.77712,
                                  0.0028426, 178.34, -0.07711, 0.0041249]),
                'KspA': np.array([-171.945, -0.077993, 2903.293, 71.595, -0.068393,
                                  0.0017276, 88.135, -0.10018, 0.0059415]),
                'KSO4': np.array([141.328, -4276.1, -23.093, -13856, 324.57, -47.986, 35474,
                                  -771.54, 114.723, -2698, 1776])}


# K Functions
# Unless otherwise stated, all are from Dickson, Sabine and Christian 2007 handbook.
def func_K0(TKS, a, b, c, d, e, f):
    TempK, Sal = TKS
    return np.exp(a +
                  b * 100 / TempK +
                  c * np.log(TempK / 100) +
                  Sal * (d + e * TempK / 100 +
                         f * TempK / 100 * TempK / 100))


def func_K1(TKS, a, b, c, d, e):
    TempK, Sal = TKS
    return np.power(10, (a +
                         b / TempK +
                         c * np.log(TempK) +
                         d * Sal +
                         e * Sal * Sal))


def func_K2(TKS, a, b, c, d, e):
    TempK, Sal = TKS
    return np.power(10, (a +
                         b / TempK +
                         c * np.log(TempK) +
                         d * Sal +
                         e * Sal * Sal))


def func_KB(TKS, a, b, c, d, e, f, g, h, i, j, k, l):
    TempK, Sal = TKS
    sqrtSal = np.sqrt(Sal)
    lnTempK = np.log(TempK)
    return np.exp(a +
                  b * sqrtSal +
                  c * Sal + (1 / TempK) *
                  (d +
                   e * sqrtSal +
                   f * Sal +
                   g * Sal * sqrtSal +
                   h * Sal * Sal) + lnTempK *
                  (i +
                   j * sqrtSal +
                   k * Sal) +
                  l * sqrtSal * TempK)


def func_KW(TKS, a, b, c, d, e, f, g):
    TempK, Sal = TKS
    sqrtSal = np.sqrt(Sal)
    lnTempK = np.log(TempK)
    return np.exp(a +
                  b / TempK +
                  c * lnTempK + sqrtSal *
                  (d / TempK +
                   e +
                   f * lnTempK) +
                  g * Sal)


# From Zeebe and Wolf-Gladrow, Appendix A.10.
def func_KspC(TKS, a, b, c, d, e, f, g, h, i):
    TempK, Sal = TKS
    log10TempK = np.log10(TempK)
    return np.power(10, (a +
                         b * TempK +
                         c / TempK +
                         d * log10TempK + Sal**0.5 *
                         (e +
                          f * TempK +
                          g / TempK) +
                         h * Sal +
                         i * Sal**1.5))

# From Zeebe and Wolf-Gladrow, Appendix A.10.
def func_KspA(TKS, a, b, c, d, e, f, g, h, i):
    TempK, Sal = TKS
    sqrtSal = np.sqrt(Sal)
    log10TempK = np.log10(TempK)
    return np.power(10, (a +
                         b * TempK +
                         c / TempK +
                         d * log10TempK + Sal**0.5 *
                         (e +
                          f * TempK +
                          g / TempK) +
                         h * Sal +
                         i * Sal**1.5))


def func_KSO4(TKS, a, b, c, d, e, f, g, h, i, j, k):
    TempK, Sal = TKS
    I = 19.924 * Sal / (1000 - 1.005 * Sal)
    sqrtI = np.sqrt(I)
    lnTempK = np.log(TempK)
    return np.exp(a +
                  b / TempK +
                  c * lnTempK + sqrtI *
                  (d / TempK +
                   e +
                   f * lnTempK) + I *
                  (g / TempK +
                   h +
                   i * lnTempK) +
                  j / TempK * I * sqrtI +
                  k / TempK * I * I + np.log(1 - 0.001005 * Sal))


fn_dict = {'K0': func_K0,
           'K1': func_K1,
           'K2': func_K2,
           'KB': func_KB,
           'KW': func_KW,
           'KspC': func_KspC,
           'KspA': func_KspA,
           'KSO4': func_KSO4}


# fit Functions
def fitfunc_K0(TKS, a, b, c, d, e, f):
    ps = start_params['K0'] * (a, b, c, d, e, f)
    return func_K0(TKS, *ps)


def fitfunc_K1(TKS, a, b, c, d, e):
    ps = start_params['K1'] * (a, b, c, d, e)
    return func_K1(TKS, *ps)


def fitfunc_K2(TKS, a, b, c, d, e):
    ps = start_params['K2'] * (a, b, c, d, e)
    return func_K2(TKS, *ps)


def fitfunc_KB(TKS, a, b, c, d, e, f, g, h, i, j, k, l):
    ps = start_params['KB'] * (a, b, c, d, e, f, g, h, i, j, k, l)
    return func_KB(TKS, *ps)


def fitfunc_KW(TKS, a, b, c, d, e, f, g):
    ps = start_params['KW'] * (a, b, c, d, e, f, g)
    return func_KW(TKS, *ps)


def fitfunc_KspC(TKS, a, b, c, d, e, f, g, h, i):
    ps = start_params['KspC'] * (a, b, c, d, e, f, g, h, i)
    return func_KspC(TKS, *ps)


def fitfunc_KspA(TKS, a, b, c, d, e, f, g, h, i):
    ps = start_params['KspA'] * (a, b, c, d, e, f, g, h, i)
    return func_KspA(TKS, *ps)


def fitfunc_KSO4(TKS, a, b, c, d, e, f, g, h, i, j, k):
    ps = start_params['KSO4'] * (a, b, c, d, e, f, g, h, i, j, k)
    return func_KSO4(TKS, *ps)


fitfn_dict = {'K0': fitfunc_K0,
              'K1': fitfunc_K1,
              'K2': fitfunc_K2,
              'KB': fitfunc_KB,
              'KW': fitfunc_KW,
              'KspC': fitfunc_KspC,
              'KspA': fitfunc_KspA,
              'KSO4': fitfunc_KSO4}


# Main (new) functions
# --------------------------------------
def MyAMI_params(XmCa=0.0102821, XmMg=0.0528171):
    """
    Calculate equilibrium constant parameters using MyAMI model.

    Parameters
    ----------
    XmCa : float
        Ca concentration in mol/kgSW.
    XmMg : float
        Mg concentration in mol/kgSW
    P : float or array-like
        Pressure in bar.

    Returns
    -------
    dict of params
    """

    # Apply PITZER model to predict conditional constants across a Temp / Sal range
    # for the specified [Ca] and [Mg]

    # Modern (M) concentration (m) of Ca and Mg case T=25C, I=0.7, seawatercomposition
    MmMg = 0.0528171  # Mg Millero et al., 2008; Dickson OA-guide
    MmCa = 0.0102821  # Ca Millero et al., 2008; Dickson OA-guide

    # number of Temp and Sal steps used as the basis dataset for the fitting of the pK's
    n = 21  # number Temp and Sal levels

    # create list of Temp's and Sal's defining the grid for fitting pK's
    TempC = np.linspace(0, 40, n)  # 0-40degC in N steps
    Sal = np.linspace(30, 40, n)  # 30-40 Sal
    TempC_M, Sal_M = np.meshgrid(TempC, Sal)  # generate grid in matrix form
    TempK = TempC + 273.15
    TempK_M = TempC_M + 273.15

    # Calculate K's for modern seawater composition
    KspC_mod, K1_mod, K2_mod, KW_mod, KB_mod, KspA_mod, K0_mod, KSO4_mod = CalculateKcond(TempC_M, Sal_M)

    # Calculate gK's for modern (mod) and experimental (x) seawater composition
    gKspC_mod, gK1_mod, gK2_mod, gKW_mod, gKB_mod, gKspA_mod, gK0_mod, gKSO4_mod = calculate_gKs(TempC_M, Sal_M, MmCa, MmMg)
    gKspC_X, gK1_X, gK2_X, gKW_X, gKB_X, gKspA_X, gK0_X, gKSO4_X = calculate_gKs(TempC_M, Sal_M, XmCa, XmMg)

    # Calculate conditional K's predicted for seawater composition X
    X_dict = {'K0': K0_mod * gK0_X / gK0_mod,
              'K1': K1_mod * gK1_X / gK1_mod,
              'K2': K2_mod * gK2_X / gK2_mod,
              'KB': KB_mod * gKB_X / gKB_mod,
              'KW': KW_mod * gKW_X / gKW_mod,
              'KspC': KspC_mod * gKspC_X / gKspC_mod,
              'KspA': KspA_mod * gKspA_X / gKspA_mod,
              'KSO4': KSO4_mod * gKSO4_X / gKSO4_mod}

    maxfevN = 2000000  # number of optimiztion timesteps allowed to reach convergence

    param_dict = Bunch()
    for k in fitfn_dict.keys():
        p0 = np.ones(len(start_params[k]))
        p, cov = curve_fit(fitfn_dict[k],
                           (TempK_M.ravel(), Sal_M.ravel()),
                           X_dict[k].ravel(),
                           p0=p0, maxfev=maxfevN)

        param_dict[k] = p * start_params[k]

    return param_dict


def MyAMI_K_calc(TempC=25., Sal=35., Ca=0.0102821, Mg=0.0528171, P=None, param_dict=None):
    """
    Calculate K constants at given salinities and temperatures.

    Note: if both inputs are array-like, they must be the same
    length.

    Parameters
    ----------
    TempC : float or array-like
        Temperature in centigrade.
    Sal : float or array-like
        Salinity in psu.
    Ca : float or array-like
        Ca concentration in mol/kgSW.
    Mg : float or array-like
        Mg concentration in mol/kgSW.
    P : float or array-like
        Pressure in bar.
    param_dict : dict
        Dictionary of paramters calculated by MyAMI_params.

    Returns
    -------
    dict of K values
    """
    if param_dict is None:
        if Ca == 0.0102821 and Mg == 0.0528171:
            # MyAMI parameters for s=35, t=25. Specified in full rather than
            # calculated, for speed.
            param_dict = {'K0': np.array([-6.02409000e+01, 9.34517000e+01, 2.33585000e+01,
                                          2.35170000e-02, -2.36560000e-02, 4.70360000e-03]),
                          'K1': np.array([6.12172000e+01, -3.63386000e+03, -9.67770000e+00,
                                          1.15550000e-02, -1.15200000e-04]),
                          'K2': np.array([-2.59290000e+01, -4.71780000e+02, 3.16967000e+00,
                                          1.78100000e-02, -1.12200000e-04]),
                          'KSO4': np.array([1.41328000e+02, -4.27610000e+03, -2.30930000e+01,
                                            -1.38560000e+04, 3.24570000e+02, -4.79860000e+01,
                                            3.54740000e+04, -7.71540000e+02, 1.14723000e+02,
                                            -2.69800000e+03, 1.77600000e+03]),
                          'KB': np.array([1.48024800e+02, 1.37194200e+02, 1.62142000e+00,
                                          -8.96690000e+03, -2.89053000e+03, -7.79420000e+01,
                                          1.72800000e+00, -9.96000000e-02, -2.44344000e+01,
                                          -2.50850000e+01, -2.47400000e-01, 5.31050000e-02]),
                          'KspA': np.array([-1.71945000e+02, -7.79930000e-02, 2.90329300e+03,
                                            7.15950000e+01, -6.83930000e-02, 1.72760000e-03,
                                            8.81350000e+01, -1.00180000e-01, 5.94150000e-03]),
                          'KspC': np.array([-1.71906500e+02, -7.79930000e-02, 2.83931900e+03,
                                            7.15950000e+01, -7.77120000e-01, 2.84260000e-03,
                                            1.78340000e+02, -7.71100000e-02, 4.12490000e-03]),
                          'KW': np.array([1.48965200e+02, -1.38472600e+04, -2.36521000e+01,
                                          1.18670000e+02, -5.97700000e+00, 1.04950000e+00,
                                          -1.61500000e-02])}
        else:
            param_dict = MyAMI_params(Ca, Mg)

    TempC, Sal = [np.array(p) for p in (TempC, Sal)]
    TempK = TempC + 273.15
    Ks = Bunch()
    for k in param_dict.keys():
        Ks[k] = fn_dict[k]((TempK, Sal),
                           *param_dict[k])

    # Pressure correction
    if P is not None:
        # parameters from Table 5 of Millero 2007 (doi:10.1021/cr0503557)
        ppar = {'K1': [-25.50, 0.1271, 0, -3.08, 0.0877],
                'K2': [-15.82, -0.0219, 0, 1.13, -0.1475],
                'KB': [-29.48, 0.1622, 2.608e-3, -2.84, 0],
                'KW': [-25.60, 0.2324, -3.6246e-3, -5.13, 0.0794],
                'KSO4': [-18.03, 0.0466, 0.316e-3, -4.53, 0.0900],
                'KHF': [-9.78, -0.0090, -0.942e-3, -3.91, 0.054],
                'KH2S': [-14.80, 0.0020, -0.400e-3, 2.89, 0.054],
                'KNH4': [-26.43, 0.0889, -0.905e-3, -5.03, 0.0814],
                'KH3PO4': [-14.51, 0.1211, -0.321e-3, -2.67, 0.0427],
                'KH2PO4': [-23.12, 0.1758, -2.647e-3, -5.15, 0.09],
                'KHPO42': [-26.57, 0.2020, -3.042e-3, -4.08, 0.0714],
                'KspC': [-48.76, 0.5304, 0, -11.76, 0.3692],
                'KspA': [-35, 0.5304, 0, -11.76, 0.3692]}

        for k in ['K1', 'K2', 'KW', 'KB', 'KspA', 'KspC', 'KSO4']:
            Ks[k] *= prescorr(P, TempC, *ppar[k])

    return Ks


def MyAMI_K_calc_multi(T=25., S=35., Ca=0.0102821, Mg=0.0528171, P=None):
    """
    Calculate MyAMI equilibrium constants for multiple T, S and Mg and Ca conditions.

    Inputs should be array-like or a single value. If more than one is array-like,
    they should be the same length.

    Parameters
    ----------
    T : float or array-like
        Temperature in centigrade.
    S : float or array-like
        Salinity in psu.
    Ca : float or array-like
        Ca concentration in mol/kgSW.
    Mg : float or array-like
        Mg concentration in mol/kgSW.
    P : float or array-like
        Pressure in bar.

    Returns
    -------
    dict of dicts of params, where keys are (Ca, Mg) pairs.
    """
    # package data in a bunch of 1D arrays.
    d = Bunch()
    d.T = np.array(T, ndmin=1)
    d.S = np.array(S, ndmin=1)
    d.Ca = np.array(Ca, ndmin=1)
    d.Mg = np.array(Mg, ndmin=1)
    d.P = np.array(P, ndmin=1)

    # make all shorter arrays repeat to length of longest
    mL = max(d, key=lambda k: d[k].size)  # ID longest array
    L = d[mL].size  # get size of longest
    for k, v in d.items():
        if k != mL:
            d[k] = itertools.cycle(v)  # turn all shorter arrays into itertools.cycle objects

    zd = np.array(list(zip(d.T, d.S, d.P, d.Ca, d.Mg)), dtype=np.float64)  # make a 4xL array of parameters

    # identify Ca-Mg pairs
    CaMg = set(zip(*zd[:, -2:].T))

    # set up empty K Bunch
    Ks = Bunch({k: np.zeros(L) for k in ['K0', 'K1', 'K2', 'KSO4', 'KB', 'KspA', 'KspC', 'KW']})

    # calculate T and S specific Ks for each Ca-Mg pair.
    for (ca, mg) in tqdm(CaMg, desc='Calculating MyAMI Constants'):
        # par = MyAMI_params(ca, mg)  # calculate parameters for Ca-Mg conditions

        ind = (zd[:, -2] == ca) & (zd[:, -1] == mg)
        t = zd[ind, 0]
        s = zd[ind, 1]
        p = zd[ind, 2]
        # remove Nans (bodge)
        p[np.isnan(p)] = 0.

        Ks_tmp = MyAMI_K_calc(t, s, ca, mg, p)

        for k in Ks.keys():
            Ks[k][ind] = Ks_tmp[k]

    return Ks


def MyAMI_pK_calc(TempC=25., Sal=35., Ca=0.0102821, Mg=0.0528171, P=0., param_dict=None):
    """
    Calculate pK constants at given salinities and temperatures.

    Note: if both inputs are array-like, they must be the same
    length.

    Parameters
    ----------
    TempC : float or array-like
        Temperature in centigrade.
    Sal : float or array-like
        Salinity in psu.
    Ca : float or array-like
        Ca concentration in mol/kgSW.
    Mg : float or array-like
        Mg concentration in mol/kgSW.
    P : float or array-like
        Pressure in bar.
    param_dict : dict
        Dictionary of paramters calculated by MyAMI_params.

    Returns
    -------
    dict of pK values
    """
    Ks = MyAMI_K_calc(TempC, Sal, Ca, Mg, P, param_dict)
    pKs = Bunch()
    for k, v in Ks.items():
        pKs[k] = -np.log10(v)
    return pKs


# Test function
# -------------
def MyAMI_test():
    """
    Make sure V2 is working correctly.

    Checks calculated paramters against MyAMI_V1 and Dickson values.

    Dickson values from:
    Dickson, Sabine & Christian (2007), Guide to best practises
    for ocean CO2 measurements.
    """

    # parameters calculated by MyAMI_V1.py
    MyAMI_orig = {'K0': np.array([-60.240900000000003, 93.451700000000002, 23.358499999999999, 0.023517, -0.023656, 0.0047035999999999996]),
                  'K1': np.array([61.217199999999998, -3633.8600000000001, -9.6776999999999997, 0.011554999999999999, -0.00011519999999999999]),
                  'K2': np.array([-25.928999999999998, -471.77999999999997, 3.16967, 0.017809999999999999, -0.0001122]),
                  'KB': np.array([148.0248, 137.1942, 1.6214200000000001, -8966.8999999999996, -2890.5300000000002, -77.941999999999993, 1.728, -0.099599999999999994, -24.4344, -25.085000000000001, -0.24740000000000001, 0.053104999999999999]),
                  'KW': np.array([148.96520000000001, -13847.26, -23.652100000000001, 118.67, -5.9770000000000003, 1.0495000000000001, -0.016150000000000001]),
                  'KspC': np.array([-171.90649999999999, -0.077993000000000007, 2839.319, 71.594999999999999, -0.77712000000000003, 0.0028425999999999998, 178.34, -0.077109999999999998, 0.0041248999999999999]),
                  'KspA': np.array([-171.94499999999999, -0.077993000000000007, 2903.2930000000001, 71.594999999999999, -0.068392999999999995, 0.0017275999999999999, 88.135000000000005, -0.10018000000000001, 0.0059414999999999997]),
                  'KSO4': np.array([141.328, -4276.1000000000004, -23.093, -13856.0, 324.56999999999999, -47.985999999999997, 35474.0, -771.53999999999996, 114.723, -2698.0, 1776.0])}

    MyAMI_lowMgnormCa = {'K0': np.array([-58.835988024241153, 91.481812147562721, 22.67747869243458, 0.015798688488939287, -0.018255877435858551, 0.0038003679139008897]),
                         'K1': np.array([58.062087886559837, -3494.0233318473447, -9.2038164315804138, 0.011168501289914329, -0.00011365376326036976]),
                         'K2': np.array([-31.434097349400538, -162.75662343373008, 3.9109079889323324, 0.017704525744139957, -0.00011016412477939389]),
                         'KB': np.array([156.24940221168944, 128.25079663389661, 1.2732109192923926, -9381.59308693192, -2723.8280995436671, -66.130468335345725, 2.0505262462401235, -0.11549440220126056, -25.63858301431771, -23.400388268842654, -0.19484439742973533, 0.04906551592740143]),
                         'KW': np.array([167.00613348978843, -14400.583448847821, -26.52650914480521, 338.16897225894417, -10.913107565225641, 1.7768705047133142, -0.018636421994989119]),
                         'KspC': np.array([-82.768891656782529, -0.0510250425127734, 653.61741054522099, 35.217868927106146, -0.88625140484347653, 0.0030161168011013243, 188.12909155930794, -0.074743157746358937, 0.0040202923717078919]),
                         'KspA': np.array([-81.943776995027889, -0.050733308492461167, 697.26556788488676, 34.861317352102368, -0.17500557330551639, 0.0018967287129921546, 97.551225643759196, -0.097813597252809872, 0.0058373362605226046]),
                         'KSO4': np.array([144.01723254985009, -4387.5774837397103, -23.495881238664886, -14072.430144624854, 329.36367918201313, -48.71841105264248, 35279.449308067575, -770.95513413049468, 114.68004975234182, -2574.4649294352289, 1744.9018046633666])}

    MyAMI_highMgnormCa = {'K0': np.array([-61.810571613553734, 95.652592105230966, 24.119389775679647, 0.031633200121371587, -0.029340456577593793, 0.0056540258038219647]),
                          'K1': np.array([64.646362988472191, -3785.8006010138361, -    10.192756962536393, 0.011991951216023209, -    0.00011695382574553078]),
                          'K2': np.array([-22.608853013802992, -640.62695443600217, 2.7186241756853655, 0.017134698149918966, -0.00010846862856783123]),
                          'KB': np.array([135.80210783355605, 148.02810999968719, 1.9073200904360372, -8397.0531700794145, -3102.8610734941999, -91.196260901984544, 1.8875655121628416, -0.10394558858478999, -22.634895306112007, -27.082927065489599, -    0.29068978419138441, 0.057501272476377069]),
                          'KW': np.array([152.35156939395085, -14181.651163426186, -    24.014243686459675, 94.885045534960184, -5.0639203227863989, 0.91026133561441924, -0.017428588478068725]),
                          'KspC': np.array([-204.75418064702887, -0.088662754155618279, 3593.6747830384475, 85.188432139333557, -0.77303216591690571, 0.0028322646621595665, 180.47102892397942, -0.077798210121014622, 0.0041609885276488552]),
                          'KspA': np.array([-205.27704739574395, -0.088822477794205643, 3669.2749301858271, 85.387619157326156, -0.065400426582049809, 0.0017191937785523521, 90.430831723766886, -0.10086886021110948, 0.0059773450025442203]),
                          'KSO4': np.array([138.11090558191418, -4143.2416843287729, - 22.610868816926811, -13639.732449040992, 319.889250580885, -47.26969380854311, 35696.914533285184, -772.52584556489103, 114.82170476859116, -2829.3346454669118, 1808.8923394974406])}

    MyAMI_normMglowCa = {'K0': np.array([-60.354861415114186, 93.6114901661186, 23.413741974192469, 0.019471443043081429, -0.020880580475490663, 0.0042360614216956959]),
                         'K1': np.array([60.571282498053101, -3604.851785021769, -9.5808093449178067, 0.011627611432340741, -0.00011554542195645259]),
                         'K2': np.array([-27.797620974421175, -379.10039640360452, 3.4344772188147248, 0.017795324243456347, -0.00011189730601039271]),
                         'KB': np.array([149.73578915124264, 135.51878754422717, 1.5697190283621318, -9047.2694347882134, -2859.3270820038338, -75.691260952752486, 1.7275714796068642, -0.10040470700438676, -24.686290139620521, -24.773314274598235, -0.23961001755283762, 0.052391979614480146]),
                         'KW': np.array([149.5757384358902, -13863.557270210315, -23.750438094274756, 129.08418248409774, -6.2195684718423587, 1.084610305608863, -0.015439222265140332]),
                         'KspC': np.array([-211.52170472418692, -0.090199167158515448, 3856.1688808669751, 87.682933659918803, -0.85761852467556132, 0.0029765579810714267, 189.36084847727781, -0.076643561611619324, 0.0041047729754152317]),
                         'KspA': np.array([-212.32447841169937, -0.090444362804512873, 3938.9176464157658, 87.995927923836049, -0.1501097667611383, 0.0018636059614886204, 99.328910907417679, -0.099710661830045427, 0.0059212989088641534]),
                         'KSO4': np.array([141.1770439414571, -4273.0265734374207, -23.069475159874695, -13881.516800613268, 325.4259752983441, -48.119398949498262, 35445.935681241004, -771.56704443236777, 114.73121075935127, -2676.1174771340106, 1770.1552350112891])}

    MyAMI_normMghighCa = {'K0': np.array([-58.590903344471016, 91.138226208152034, 22.558657775967909, 0.049708549006778388, -0.041565798253250033, 0.0077243125633299553]),
                          'K1': np.array([64.31775412475875, -3774.150593504583, -10.142460932422811, 0.010789364340744423, -0.00011172101044368213]),
                          'K2': np.array([-18.1184401789505, -847.60828395037345, 2.0555415100456993, 0.018079843009880359, -0.00011364444693652285]),
                          'KB': np.array([141.5124421720904, 144.20233560051338, 1.8668270247130392, -8660.6047342969923, -3013.252222111963, -88.573923681298325, 1.6904831429939082, -0.093069165984674337, -23.475935776590447, -26.399150028099069, -0.28426489312070991, 0.056213603865328497]),
                          'KW': np.array([145.10404153248484, -13633.01312911182, -23.109609760314786, 80.944017507293069, -5.2690814493605655, 0.95142669086974152, -0.020499466300191754]),
                          'KspC': np.array([41.56845663058197, -0.012494837547226051, -2621.270517368956, -15.106275506980497, -0.42490307162652657, 0.0022546429971244037, 128.59267314835913, -0.078567023251951862, 0.004187021665224497]),
                          'KspA': np.array([45.46883333289923, -0.011231833768196238, -2654.0719220492115, -16.719120961494937, 0.28994297991389995, 0.0011290794887153634, 37.493189951682702, -0.10162567004805068, 0.0060026679175019422]),
                          'KSO4': np.array([144.299587462815, -4382.0135712955152, -23.545133256925382, -13723.014934595181, 320.04593550412079, -47.277080818942601, 35505.166190487507, -770.55784024737886, 114.56947002838029, -2752.9698393649651, 1790.6413670134023])}

    MyAMI_lowMglowCa = {'K0': np.array([-59.031957809407686, 91.756584913758871, 22.772475270805277, 0.012439048955401214, -0.015957562302394309, 0.0034127941095757714]),
                        'K1': np.array([57.645653166592687, -3475.20508087085, -9.1413869167198492, 0.011261638932758035, -0.00011407852450762963]),
                        'K2': np.array([-33.797052214730329, -34.687019726678315, 4.2408857775382689, 0.017505028643035214, -0.00010825154698773285]),
                        'KB': np.array([156.5145262978281, 127.50333419476038, 1.2215711713924888, -9399.3466614485496, -2714.0159959999696, -64.48136298090111, 2.1233847822014575, -0.11922787349734948, -25.676139876177906, -23.252155109817927, -0.18709894308737055, 0.04864009697090256]),
                        'KW': np.array([167.10645326206787, -14407.78610729866, -26.535372869940193, 339.51112629715283, -10.93693969267213, 1.7782409980790552, -0.017212537882267356]),
                        'KspC': np.array([-133.24446950415958, -0.067315974817180324, 1914.5973496967124, 55.857418991475782, -1.0537934906898849, 0.0032909347910871382, 211.26771344577173, -0.07375599800265277, 0.0039795671790368381]),
                        'KspA': np.array([-133.57641396700726, -0.067394420420986825, 1986.7281617372416, 55.974478381552849, -0.34424060172868359, 0.002174466671642204, 120.936193025869, -0.096829351862153537, 0.0057968937363111032]),
                        'KSO4': np.array([142.87795647594166, -4336.8363580664582, -23.324131972826077, -14191.380110202912, 331.82738359227761, -49.092005161740197, 35356.825240910941, -772.1635449423793, 114.86252820078565, -2594.6365722656565, 1751.0699498051545])}

    MyAMI_highMghighCa = {'K0': np.array([-60.242558923531988, 93.454026697777721, 23.35930393085005, 0.058514324588747689, -0.047729633422100563, 0.0087550495701775415]),
                          'K1': np.array([67.975551223686139, -3936.2440524408544, -10.691852254750499, 0.011246896099572095, -0.00011355427640403392]),
                          'K2': np.array([-17.015715892221774, -902.20421770611688, 1.9199531113654376, 0.017212262772804524, -0.00010799407526682958]),
                          'KB': np.array([132.80183421700798, 154.26130412176894, 2.2945018159380695, -8235.6462474105119, -3180.0374178989064, -106.69605945646256, 1.6982781679275936, -0.090838190573174118, -22.203197266497337, -28.305328913879592, -0.34878219590073556, 0.060890052818741602]),
                          'KW': np.array([148.59341506108012, -13995.447057143834, -23.46484882933628, 54.475141138360499, -4.2384008335651258, 0.79103107136328821, -0.020072375108112162]),
                          'KspC': np.array([-32.141860679870902, -0.035931500758019999, -823.93579550223012, 15.115652324166364, -0.53054878823398877, 0.0024230088434383269, 145.65725450322446, -0.078466431425245023, 0.0041921237816589232]),
                          'KspA': np.array([-29.574105962378695, -0.035101253030014816, -824.33399730398355, 14.049494193893432, 0.18187006354626084, 0.0013017899807812074, 54.926624770335998, -0.1015408905334869, 0.0060085347187136574]),
                          'KSO4': np.array([141.44308527847932, -4263.2391572297274, -23.117231632134168, -13446.393136941231, 313.79318037915908, -46.323718708827585, 35723.642314394208, -770.85435112916298, 114.56164858754464, -2903.5083191909271, 1828.8097340479994])}

    # log10(K) values from Dickson, Sabine & Christian (2007) Chapter 5
    Dickson_ref = {'K0': -np.log10(np.exp(-3.5617)),
                   'K1': 5.8472,
                   'K2': 8.9660,
                   'KB': -np.log10(np.exp(-19.7964)),
                   'KW': -np.log10(np.exp(-30.434)),
                   'KspC': np.nan,
                   'KspA': np.nan,
                   'KSO4': -np.log10(np.exp(-2.3))}

    par = ['K0', 'K1', 'K2', 'KB', 'KW', 'KSO4', 'KspC', 'KspA']
    TKS = (25 + 273.15, 35.)

    def pcomp(params, Kds):
        print(''.join(['{:10}'.format(p) for p in par]))
        print(''.join(['{:<10.3f}'.format(Kd) for Kd in Kds]))
#         print(''.join(['{:<10.1e}'.format(Kd) for Kd in Kds]))
        print('\n')

    def Kdcalc(params, V1_params, par):
        Kds = []
        for k in par:
            V1 = fn_dict[k](TKS, *V1_params[k])
            V2 = fn_dict[k](TKS, *params[k])
            Kds.append(100 * (V1 - V2) / V1)
        return Kds

    # calculate params using MyAMI_V2
    params = MyAMI_params(0.0102821, 0.0528171)
    Dds = []
    for k in par:
        V2 = -np.log10(fn_dict[k](TKS, *params[k]))
        Dds.append(100 * (Dickson_ref[k] - V2) / Dickson_ref[k])

    Kds = Kdcalc(params, MyAMI_orig, par)

    print('Testing MyAMI_V2')
    print('----------------')
    print()
    print('Normal Seawater')
    print('---------------')
    print('% Difference between MyAMI_V1 and MyAMI_V2 K values:')
    print(''.join(['{:10}'.format(p) for p in par]))
    print(''.join(['{:<10.3f}'.format(Kd) for Kd in Kds]))
    print()
    print('% Difference between Dickson values and MyAMI_V2 pK values')
    print(''.join(['{:10}'.format(p) for p in par]))
    print(''.join(['{:<10.3f}'.format(Kd) for Kd in Dds]))
    print('\n\n')

    print('% Difference from MyAMI_V1 K values:')

    # low Mg
    params = MyAMI_params(0.01, 0.01)
    Kds = Kdcalc(params, MyAMI_lowMgnormCa, par)
    print('Low Mg (0.01), Normal Ca (0.01)')
    print('-------------------------------')
    pcomp(params, Kds)

    # high Mg
    params = MyAMI_params(0.01, 0.1)
    Kds = Kdcalc(params, MyAMI_highMgnormCa, par)
    print('High Mg (0.1), Normal Ca (0.01)')
    print('-------------------------------')
    pcomp(params, Kds)

    # low Ca
    params = MyAMI_params(0.005, 0.05)
    Kds = Kdcalc(params, MyAMI_normMglowCa, par)
    print('Normal Mg (0.05), Low Ca (0.005)')
    print('--------------------------------')
    pcomp(params, Kds)

    # high Ca
    params = MyAMI_params(0.05, 0.05)
    Kds = Kdcalc(params, MyAMI_normMghighCa, par)
    print('Normal Mg (0.05), High Ca (0.05)')
    print('--------------------------------')
    pcomp(params, Kds)

    # both high
    params = MyAMI_params(0.05, 0.1)
    Kds = Kdcalc(params, MyAMI_highMghighCa, par)
    print('High Mg (0.1), High Ca (0.05)')
    print('-----------------------------')
    pcomp(params, Kds)

    # both low
    params = MyAMI_params(0.005, 0.01)
    Kds = Kdcalc(params, MyAMI_lowMglowCa, par)
    print('Low Mg (0.01), Low Ca (0.005)')
    print('-----------------------------')
    pcomp(params, Kds)

    return
