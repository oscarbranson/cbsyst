import numpy as np
import pandas as pd
from .helpers import MyAMI_resource_file, expand_dims, match_dims, load_params

# dictionaries of valid ions containing their matrix indices
# positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
Pind = {
    'H': 0,
    'Na': 1,
    'K': 2,
    'Mg': 3,
    'Ca': 4,
    'Sr': 5
}

# negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
Nind = {
    'OH': 0,
    'Cl': 1,
    'B(OH)4': 2,
    'HCO3': 3,
    'HSO4': 4,
    'CO3': 5,
    'SO4': 6
}

# dictionary containing all valid ions
Iind = Pind.copy()
Iind.update(Nind)

# helper functions for converting tables into calculationg matrices

def filter_terms(tab, valid_ions):
    include = []
    for ions in tab.Parameter.str.split('-'):
        include.append(~np.any([i not in valid_ions for i in ions]))

    return tab.loc[include]

def get_ion_index(ions):
    return tuple([Iind[k] for k in ions.split('-')])

def EqA10(a, TK):
    """
    Calculate Phi and Theta parameters as a function of TK accoring to 
    """
    # a1 + a2 / T + a3 * T + a4 * (T - 298.15) + a5 * (T - 298.15)**2
    return a[0] + a[1] / TK + a[2] * 1e-4 * TK + a[3] * 1e-4 * (TK - 298.15) * a[4] * 1e-6 * (TK - 298.15)**2


# Load Tables A19 and A11
TABA11 = pd.read_csv(MyAMI_resource_file('TabA11.csv'), comment='#')
TABA10 = pd.read_csv(MyAMI_resource_file('TabA10.csv'), comment='#')

TABA11 = filter_terms(TABA11, Iind)
TABA10 = filter_terms(TABA10, Iind)
TABA10.fillna(0, inplace=True)


def calc_Theta_Phi(TK):
    """
    Construct Theta and Phi matrices from Table A10 and A11 of Millero and Pierrot (1998).

    Parameters
    ----------
    TK : array-like
        Temperature in Kelvin
    
    Returns
    -------
    tuple of array-like
        Containing (Theta_negative, Theta_positive, Phi_NNP, Phi_PPN)
    """

    # create empty arrays
    Theta_positive = np.zeros((len(Pind), len(Pind), *TK.shape))
    Theta_negative = np.zeros((len(Nind), len(Nind), *TK.shape))
    Phi_PPN = np.zeros((len(Pind), len(Pind), len(Nind), *TK.shape))
    Phi_NNP = np.zeros((len(Nind), len(Nind), len(Pind), *TK.shape))

    # Assign static values from Tabe A11
    for _, row in TABA11.iterrows():
        ions = row.Parameter.split('-')
        index = get_ion_index(row.Parameter)

        if ions[0] in Pind:
            if len(ions) == 2:
                Theta_positive[index] = row.Value
                Theta_positive[index[::-1]] = row.Value
            elif len(ions) == 3:
                Phi_PPN[index] = row.Value
                Phi_PPN[index[1], index[0], index[2]] = row.Value
        
        if ions[0] in Nind:
            if len(ions) == 2:
                Theta_negative[index] = row.Value
                Theta_negative[index[::-1]] = row.Value
            elif len(ions) == 3:
                Phi_NNP[index] = row.Value
                Phi_NNP[index[1], index[0], index[2]] = row.Value

    # Assign T-sensitive values from Table A10
    pnames = ['a1', 'a2', 'a3_e4', 'a4_e4', 'a5_e6']  # parameter names in TABle
    for _, row in TABA10.iterrows():
        ions = row.Parameter.split('-')
        index = get_ion_index(row.Parameter)
        
        a = row[pnames]  # identify parameters
        val = EqA10(a, TK)
        
        if ions[0] in Pind:
            if len(ions) == 2:
                Theta_positive[index] = val
                Theta_positive[index[::-1]] = val
            elif len(ions) == 3:
                Phi_PPN[index] = val
                Phi_PPN[index[1], index[0], index[2]] = val
        if ions[0] in Nind:
            if len(ions) == 2:
                Theta_negative[index] = val
                Theta_negative[index[::-1]] = val
            elif len(ions) == 3:
                Phi_NNP[index] = val
                Phi_NNP[index[1], index[0], index[2]] = val

    # Special cases that deviate from values in Millero and Pierrot (1998)
    special = {
        'Na-Ca-Cl': -7.6398 + -1.2990e-2 * TK + 1.1060e-5 * TK**2 + 1.8475 * np.log(TK),  # Spencer et al 1990
        'Mg-Ca-Cl': 4.15790220e1 + 1.30377312e-2 * TK - 9.81658526e2 / TK - 7.4061986 * np.log(TK),  # Spencer et al 1990
        'Cl-CO3': -0.092,  #Spencer et al 1990
        'CO3-OH': 0.1,  # http://www.aim.env.uea.ac.uk/aim/accent4/parameters.html
    }

    for ionstr, v in special.items():
        ions = ionstr.split('-')
        index = get_ion_index(ionstr)
        if ions[0] in Pind:
            if len(ions) == 2:
                Theta_positive[index] = v
                Theta_positive[index[::-1]] = v
            elif len(ions) == 3:
                Phi_PPN[index] = v
                Phi_PPN[index[1], index[0], index[2]] = v
        if ions[0] in Nind:
            if len(ions) == 2:
                Theta_negative[index] = v
                Theta_negative[index[::-1]] = v
            elif len(ions) == 3:
                Phi_NNP[index] = v
                Phi_NNP[index[1], index[0], index[2]] = v

    return Theta_negative, Theta_positive, Phi_NNP, Phi_PPN


# Load Pitzer Parameters
def load_pitzer_params(keep_sources=False):
    pitzer_params = load_params('pitzer_params.json', asarrays=False)
    
    out = {}
    for k, v in pitzer_params.items():
        if k == 'sources':
            if keep_sources:
                out[k] = v
        else:
            out[k] = np.array(v)
    return out

# Load here, so file only needs to be read once on import
PITZER_PARAMS = load_pitzer_params()
THETA_BASES = load_params('theta_bases.json')
PHI_BASES = load_params('phi_bases.json')


# functions from inside SupplyParams
def Equation_TabA1(T, Tinv, lnT, a):
    return (
        a[:, 0] +
        a[:, 1] * T +
        a[:, 2] * Tinv +
        a[:, 3] * lnT +
        a[:, 4] / (T - 263) +
        a[:, 5] * T**2 +
        a[:, 6] / (680 - T) +
        a[:, 7] / (T - 227)
    )

def EquationSpencer(T, lnT, q):
    return (
        q[:, 0] +
        q[:, 1] * T +
        q[:, 2] * T * T +
        q[:, 3] * T**3 +
        q[:, 4] / T +
        q[:, 5] * lnT
    )

def Equation1_TabA2(T, q):
    return q[:, 0] + q[:, 1] * T + q[:, 2] * T**2

def Equation2_TabA2(T, Tpower2, Tpower3, Tpower4, q):
    return (
        q[:, 0] * ((T / 2) + (88804) / (2 * T) - 298) +
        q[:, 1] * ((Tpower2 / 6) + (26463592) / (3 * T) - (88804 / 2)) +
        q[:, 2] * (Tpower3 / 12 + 88804 * 88804 / (4 * T) - 26463592 / 3) +
        q[:, 3] * ((Tpower4 / 20) + 88804 * 26463592 / (5 * T) - 88804 * 88804 / 4) +
        q[:, 4] * (298 - (88804 / T)) +
        q[:, 5]
    )

def Equation_TabA3andTabA4andTabA5(TC, a):
    return a[:, 0] + a[:, 1] * TC + a[:, 2] * TC**2

def Equation_TabA3andTabA4andTabA5_Simonson(T, a):
    return a[:, 0] + a[:, 1] * (T - 298.15) + a[:, 2] * (T - 303.15) * (T - 303.15)

def Equation_TabA7(T, P):
    return (
        P[:, 0]
        + P[:, 1] * (8834524.639 - 88893.4225 * P[:, 2]) * (1 / T - (1 / 298.15))
        + P[:, 1] / 6 * (T**2 - 88893.4225)
    )

def Equation_HCl(T, a):
    return a[:, 0] + a[:, 1] * T + a[:, 2] / T

def Equation_HSO4(T, a):
    return a[:, 0] + (T - 328.15) * 1e-3 * (
        a[:, 1] + (T - 328.15) * ((a[:, 2] / 2) + (T - 328.15) * (a[:, 3] / 6))
    )

def Equation_Na2SO4_Moller(T, lnT, a):
    return (
        a[:,0] +
        a[:,1] * T +
        a[:,2] / T +
        a[:,3] * lnT +
        a[:,4] / (T - 263) +
        a[:,5] * T**2 +
        a[:,6] / (680. - T)
    )

def Equation_HSO4_Clegg94(T, a):
    return a[:, 0] + (T - 328.15) * (
        1e-3 * a[:, 1]
        + (T - 328.15) * ((1e-3 * a[:, 2] / 2) + (T - 328.15) * 1e-3 * a[:, 3] / 6)
    )

def Eq_b2_MgSO4(T, Tpower2, Tpower3, Tpower4, q):
    return (
        q[0] * ((T / 2) + (88804) / (2 * T) - 298) +
        q[1] * ((Tpower2 / 6) + (26463592) / (3 * T) - (88804 / 2)) +
        q[2] * (Tpower3 / 12 + 88804 * 88804 / (4 * T) - 26463592 / 3) +
        q[3] * ((Tpower4 / 20) + 88804 * 26463592 / (5 * T) - 88804 * 88804 / 4) +
        q[4] * (298 - (88804 / T)) +
        q[5]
    )

def Eq_b2_MgANDCaBOH42(T, a):
    return a[0] + a[1] * (T - 298.15) + a[2] * (T - 303.15) * (T - 303.15)

def Eq_b2_CaSO4(T, a):
    return a[0] + a[1] * T


def PitzerParams(T):
    """
    Return Pitzer params for given T (Kelvin).
    
    Parameters
    ----------
    T : array-like
        Temperature in Kelvin
        
    Returns
    -------
    dict of arrays
        with keys: beta_0, beta_1, beta_2, C_phi, Theta_negative, Theta_positive, Phi_NNP, Phi_PPN, C1_HSO4
    """
    if isinstance(T, (float, int)):
        T = np.asanyarray(T)

    Tinv = 1. / T
    lnT = np.log(T)
    # ln_of_Tdiv29815 = np.log(T / 298.15)
    Tpower2 = T**2.
    Tpower3 = T**3.
    Tpower4 = T**4.
    TC = T - 298.15  # Temperature in Celcius

    ################################################################################
    # Pitzer equations, based on Millero and Pierrot (1998)

    # load pitzer params and expand dimensions to match T
    pitzer_params = {k: expand_dims(v, T) for k, v in PITZER_PARAMS.items()}
    

    # Table A8 - - - Pitzer parameters unknown; beta's known for 25degC
    Equation_KHSO4 = np.array([-0.0003, 0.1735, 0.0])

    # Equation_MgHSO42 = np.array([0.4746, 1.729, 0.0])  #  XX no Cphi #from Harvie et al 1984 as referenced in MP98
    Equation_MgHSO42 = np.array(
        [
            -0.61656 - 0.00075174 * TC,
            7.716066 - 0.0164302 * TC,
            0.43026 + 0.00199601 * TC,
        ]
    )  # from Pierrot and Millero 1997 as used in the Excel file

    # Equation_MgHCO32 = np.array([0.329, 0.6072, 0.0])  # Harvie et al 1984
    Equation_MgHCO32 = np.array([0.03, 0.8, 0.0])  # Millero and Pierrot redetermined after Thurmond and Millero 1982
    Equation_CaHSO42 = np.array([0.2145, 2.53, 0.0])
    # Equation_CaHCO32 = np.array([0.4, 2.977, 0.0])  # Harvie et al. 1984  - ERRONEOUS, see comments by Zeebe and response by Hain
    Equation_CaHCO32 = np.array([0.2, 0.3, 0])  # He and Morse 1993 after Pitzeretal85
    Equation_CaOH2 = np.array([-0.1747, -0.2303, -5.72])  # according to Harvie84, the -5.72 should be for beta2, not Cphi (which is zero) -- but likely typo in original ref since 2:1 electrolytes don't usually have a beta2
    Equation_SrHSO42 = Equation_CaHSO42
    Equation_SrHCO32 = Equation_CaHCO32
    Equation_SrOH2 = Equation_CaOH2
    # Equation_MgOHCl = np.array([-0.1, 1.658, 0.0])
    Equation_NaOH = np.array([0.0864, 0.253, 0.0044])  # Rai et al 2002 ref to Pitzer91(CRC Press)
    Equation_CaSO4_PnM74 = np.array([0.2, 2.65, 0])  # Pitzer and Mayorga74

    # param_HSO4 = np.array([[0.065, 0.134945, 0.022374, 7.2E-5],
    #                           [-15.009, -2.405945, 0.335839, -0.004379],
    #                           [0.008073, -0.113106, -0.003553, 3.57E-5]])  # XXXXX two equations for C
    # param_HSO4_Clegg94 = np.array([[0.0348925351, 4.97207803, 0.317555182, 0.00822580341],
    #                                  [-1.06641231, -74.6840429, -2.26268944, -0.0352968547],
    #                                  [0.00764778951, -0.314698817, -0.0211926525, 0.000586708222],
    #                                  [0.0, -0.176776695, -0.731035345, 0.0]])



    ############################################################
    # beta_0, beta_1 and C_phi values arranged into arrays
    N_cations = 6  # H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    N_anions = 7  # OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

    beta_0 = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array
    beta_1 = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array
    C_phi = np.zeros((N_cations, N_anions, *T.shape))  # creates empty array

    # H = cation
    # beta_0[0, 0], beta_1[0, 0], C_phi[0, 0] = n / a
    beta_0[0, 1], beta_1[0, 1], C_phi[0, 1] = Equation_HCl(T, pitzer_params['HCl'])  # H-Cl
    # beta_0[0, 2], beta_1[0, 2], C_phi[0, 2] = n / a
    # beta_0[0, 3], beta_1[0, 3], C_phi[0, 3] = n / a
    # beta_0[0, 4], beta_1[0, 4], C_phi[0, 4] = n / a
    # beta_0[0, 5], beta_1[0, 5], C_phi[0, 5] = n / a
    # beta_0[0, 6], beta_1[0, 6], C_phi[0, 6] = Equation_HSO4(T, param_HSO4)
    # beta_0[0, 6], beta_1[0, 6], C_phi[0, 6] C1_HSO4] = Equation_HSO4_Clegg94(T, param_HSO4_Clegg94)
    C1_HSO4 = 0  # What does this do?
    # print beta_0[0, :], beta_1[0, :]#, beta_2[0, :]

    # Na = cation
    beta_0[1, 0], beta_1[1, 0], C_phi[1, 0] = Equation_NaOH  # Na-OH
    beta_0[1, 1], beta_1[1, 1], C_phi[1, 1] = Equation_TabA1(T, Tinv, lnT, pitzer_params['NaCl'])  # Na-Cl
    beta_0[1, 2], beta_1[1, 2], C_phi[1, 2] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['NaBOH4'])  # Na-B(OH)4
    beta_0[1, 3], beta_1[1, 3], C_phi[1, 3] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['NaHCO3'])  # Na-HCO3
    beta_0[1, 4], beta_1[1, 4], C_phi[1, 4] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['NaHSO4'])  # Na-HSO4
    beta_0[1, 5], beta_1[1, 5], C_phi[1, 5] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['Na2CO3'])  # Na-CO3 
    # beta_0[1, 6], beta_1[1, 6], C_phi[1, 6] = Equation_Na2SO4_TabA3(T, ln_of_Tdiv29815, pitzer_params['Na2SO4'])  # Na-SO4
    beta_0[1, 6], beta_1[1, 6], C_phi[1, 6] = Equation_Na2SO4_Moller(T, lnT, pitzer_params['Na2SO4_Moller'])  # Na-SO4

    # K = cation
    beta_0[2, 0], beta_1[2, 0], C_phi[2, 0] = Equation_TabA7(T, pitzer_params['KOH'])  # K-OH
    beta_0[2, 1], beta_1[2, 1], C_phi[2, 1] = Equation_TabA1(T, Tinv, lnT, pitzer_params['KCl'])  # K-Cl
    beta_0[2, 2], beta_1[2, 2], C_phi[2, 2] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['KBOH4'])  # K-B(OH)4
    beta_0[2, 3], beta_1[2, 3], C_phi[2, 3] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['KHCO3'])  # K-HCO3
    beta_0[2, 4], beta_1[2, 4], C_phi[2, 4] = Equation_KHSO4  # K-HSO4
    beta_0[2, 5], beta_1[2, 5], C_phi[2, 5] = Equation_TabA3andTabA4andTabA5(TC, pitzer_params['K2CO3'])  # K-CO3
    beta_0[2, 6], beta_1[2, 6], C_phi[2, 6] = Equation_TabA1(T, Tinv, lnT, pitzer_params['K2SO4'])  # K-SO4

    # Mg = cation
    # beta_0[3, 0], beta_1[3, 0], C_phi[3, 0] = n / a
    beta_0[3, 1], beta_1[3, 1], C_phi[3, 1] = Equation1_TabA2(T, pitzer_params['MgCl2'])  # Mg-Cl
    beta_0[3, 2], beta_1[3, 2], C_phi[3, 2] = Equation_TabA3andTabA4andTabA5_Simonson(T, pitzer_params['MgBOH42'])    # Mg-B(OH)4
    beta_0[3, 3], beta_1[3, 3], C_phi[3, 3] = Equation_MgHCO32  # Mg-HCO3
    beta_0[3, 4], beta_1[3, 4], C_phi[3, 4] = Equation_MgHSO42  # Mg-HSO4
    # beta_0[3, 5], beta_1[3, 5], C_phi[3, 5] = n / a
    beta_0[3, 6], beta_1[3, 6], C_phi[3, 6] = Equation2_TabA2(T, Tpower2, Tpower3, Tpower4, pitzer_params['MgSO4'])   # Mg-SO4
    # print beta_0[3, 6], beta_1[3, 6], C_phi[3, 6]

    # Ca = cation
    beta_0[4, 0], beta_1[4, 0], C_phi[4, 0] = Equation_CaOH2  # Ca-OH
    beta_0[4, 1], beta_1[4, 1], C_phi[4, 1] = Equation_TabA1(T, Tinv, lnT, pitzer_params['CaCl2'])  # Ca-Cl
    beta_0[4, 2], beta_1[4, 2], C_phi[4, 2] = Equation_TabA3andTabA4andTabA5_Simonson(T, pitzer_params['CaBOH42'])  # Ca-B(OH)4
    beta_0[4, 3], beta_1[4, 3], C_phi[4, 3] = Equation_CaHCO32  # Ca-CO3
    beta_0[4, 4], beta_1[4, 4], C_phi[4, 4] = Equation_CaHSO42  # Ca-HSO4
    # beta_0[4, 5], beta_1[4, 5], C_phi[4, 5] = n / a
    # beta_0[4, 6], beta_1[4, 6], C_phi[4, 6] = Equation_TabA1(T, Tinv, lnT, pitzer_params['CaSO4'])  # Ca-SO4
    beta_0[4, 6], beta_1[4, 6], C_phi[4, 6] = Equation_CaSO4_PnM74  # Ca-SO4

    # Sr = cation
    beta_0[5, 0], beta_1[5, 0], C_phi[5, 0] = Equation_SrOH2  # Sr-OH
    beta_0[5, 1], beta_1[5, 1], C_phi[5, 1] = Equation_TabA7(T, pitzer_params['SrCl2'])  # Sr-Cl
    beta_0[5, 2], beta_1[5, 2], C_phi[5, 2] = Equation_TabA3andTabA4andTabA5_Simonson(T, pitzer_params['SrBOH42'])  # Sr-B(OH)4
    beta_0[5, 3], beta_1[5, 3], C_phi[5, 3] = Equation_SrHCO32  # Sr-HCO3
    beta_0[5, 4], beta_1[5, 4], C_phi[5, 4] = Equation_SrHSO42  # Sr-HSO4
    # beta_0[5, 5], beta_1[5, 5], C_phi[5, 5] = n / a
    # beta_0[5, 6], beta_1[5, 6], C_phi[5, 6] = Equation_TabA1(T, Tinv, lnT, param_SrSO4)  # Sr-SO4
    beta_0[5, 6], beta_1[5, 6], C_phi[5, 6] = Equation_CaSO4_PnM74  # Sr-SO4
    
    # for 2:2 ion pairs beta_2 is needed
    beta_2 = np.zeros((N_cations, N_anions, *T.shape))
    b2_param_MgSO4 = np.array([-13.764, 0.12121, -2.7642e-4, 0, -0.21515, -32.743])

    b2_param_MgBOH42 = np.array([-11.47, 0.0, -3.24e-3])
    b2_param_CaBOH42 = np.array([-15.88, 0.0, -2.858e-3])

    b2_param_CaSO4 = np.array([-55.7, 0])  # Pitzer and Mayorga74 
    # b2_param_CaSO4 = [-1.29399287e2, 4.00431027e-1])  # Moller88

    N_cations = 6  # H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    N_anions = 7  # OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

    beta_2[3, 6] = Eq_b2_MgSO4(T, Tpower2, Tpower3, Tpower4, b2_param_MgSO4)  # Mg-SO4
    beta_2[3, 2] = Eq_b2_MgANDCaBOH42(T, b2_param_MgBOH42)  # Mg-B(OH)4
    beta_2[4, 2] = Eq_b2_MgANDCaBOH42(T, b2_param_CaBOH42)  # Ca-B(OH)4
    beta_2[5, 2] = beta_2[4, 2]  # Sr-B(OH)4
    beta_2[4, 6] = Eq_b2_CaSO4(T, b2_param_CaSO4)  # Ca-SO4

    #############################################################################
    # Data and T - based calculations to create arrays holding Theta and Phi values
    # based on Table A10 and A11

    # Theta of positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    # Array to hold Theta values between ion two ions (for numbering see list above)
    
    Theta_negative, Theta_positive, Phi_NNP, Phi_PPN = calc_Theta_Phi(T)

    # # These are only the temperature-sensitive modifications, 
    # # which are added to the theta_base table imported at the top of the file
    # Theta_positive_mod = np.zeros((6, 6, *TC.shape))
    # Theta_positive_mod[[0,5], [5,0]] = 4.5e-4 * TC  # H - Sr
    # Theta_positive_mod[[0,1], [1,0]] = -2.09e-4 * TC  # H - Na
    # Theta_positive_mod[[0,2], [2,0]] = -2.275e-4 * TC  # H - K
    # Theta_positive_mod[[0,3], [3,0]] = 3.275e-4 * TC  # H - Mg
    # Theta_positive_mod[[0,4], [4,0]] = 3.275e-4 * TC  # H - Ca
    # Theta_positive_mod[[1,2], [2,1]] = 14.0213141 / T  # Na - K

    # Theta_positive = match_dims(THETA_BASES['Theta_positive'], Theta_positive_mod) + Theta_positive_mod

    # # Theta of negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
    # # Array to hold Theta values between ion two ions (for numbering see list above)

    # # These are only the temperature-sensitive modifications, 
    # # which are added to the theta_base table imported at the top of the file
    # Theta_negative_mod = np.zeros((7, 7, *T.shape))
    # Theta_negative_mod[[1,2], [2,1]] = -0.42333e-4 * TC - 21.926 * 1e-6 * TC**2  # Cl - BOH4
    # Theta_negative_mod[[0,1], [1,0]] = 3.125e-4 * TC - 8.362 * 1e-6 * TC**2  # OH - Cl

    # Theta_negative = match_dims(THETA_BASES['Theta_negative'], Theta_negative_mod) + Theta_negative_mod

    # # Phi
    # # positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
    # # negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

    # # Phi_PPN holds the values for cation - cation - anion - Table A11
    
    # # These are only the temperature-sensitive modifications, 
    # # which are added to the phi_base table imported at the top of the file
    # Phi_PPN_mod = np.zeros((6, 6, 7, *T.shape))  # Array to hold Theta values between ion two ions (for numbering see list above)
    # Phi_PPN_mod[[1,2], [2,1], 1] = -5.10212917 / T  # Na-K-Cl
    # Phi_PPN_mod[[1,2], [2,1], 6] = -8.21656777 / T  # Na-K-SO4
    # Phi_PPN_mod[[1,3], [3,1], 1] = -9.51 / T  # Na-Mg-Cl
    # Phi_PPN_mod[[1,4], [4,1], 1] = -1.2990e-2 * T + 1.1060e-5 * T**2 + 1.8475 * lnT  # Na-Ca-Cl. Spencer et al 1990 DIFFERENT FROM Table A11 # -0.003  
    # Phi_PPN_mod[[2,3], [3,2], 1] = -14.27 / T  # K-Mg-Cl
    # Phi_PPN_mod[[2,4], [4,2], 1] = -27.0770507 / T  # K-Ca-Cl
    # Phi_PPN_mod[[0,5], [5,0], 1] = -2.1e-4 * TC  # H-Sr-Cl
    # Phi_PPN_mod[[0,3], [3,0], 1] = -7.325e-4 * TC  # H-Mg-Cl
    # Phi_PPN_mod[[0,4], [4,0], 1] = -7.25e-4 * TC  # H-Ca-Cl
    # Phi_PPN_mod[[3,4], [4,3], 1] = 1.30377312e-2 * T - 9.81658526e2 / T - 7.4061986 * lnT  # Spencer et al 1990 DIFFERENT FROM Table A11 #-0.012  # Mg-Ca-Cl

    # Phi_PPN = match_dims(PHI_BASES['Phi_PPN'], Phi_PPN_mod) + Phi_PPN_mod

    # # Phi_NNP holds the values for anion - anion - cation
    # # Array to hold Theta values between ion two ions (for numbering see list above)
    
    # Phi_NNP_mod = np.zeros((7, 7, 6, *T.shape))
    # Phi_NNP_mod[[1,6], [6,1], 2] = 37.5619614 / T + 2.8469833 * 1e-4 * T  # Cl-SO4-K

    # Phi_NNP = match_dims(PHI_BASES['Phi_NNP'], Phi_NNP_mod) + Phi_NNP_mod
    
    # restore typo in original to see if it matters?
    # H-K-SO4
    # Phi_PPN[[0,2], [2,0], 6] = 0.197

    # Phi_PPN[[0,2], [2,0], 1] = 0.197  # this overwrites the H-K-Cl parameter, and is present in original MyAMI
    # print('TYPO', Phi_PPN[0,2,1])

    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'C_phi': C_phi,
        'Theta_negative': Theta_negative,
        'Theta_positive': Theta_positive,
        'Phi_NNP': Phi_NNP,
        'Phi_PPN': Phi_PPN,
        'C1_HSO4': C1_HSO4,
    }