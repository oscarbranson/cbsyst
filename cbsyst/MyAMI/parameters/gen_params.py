# Generate Pitzer, Theta and Phi parameter tables for use by MyAMI
from argon2 import Parameters
import numpy as np
import pkg_resources as pkgrs
import json

##############################################################################
# Pitzer Parameters
# 
# Isolated from MyAMI model for clarity. 
##############################################################################

param_NaCl = np.array(
    [
        (
            1.43783204e01,
            5.6076740e-3,
            -4.22185236e2,
            -2.51226677e0,
            0.0,
            -2.61718135e-6,
            4.43854508,
            -1.70502337,
        ),
        (
            -4.83060685e-1,
            1.40677470e-3,
            1.19311989e2,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.23433299,
        ),
        (
            -1.00588714e-1,
            -1.80529413e-5,
            8.61185543e0,
            1.2488095e-2,
            0.0,
            3.41172108e-8,
            6.83040995e-2,
            2.93922611e-1,
        ),
    ]
)
# note that second value is changed to original ref (e-3 instead e01)

param_KCl = np.array(
    [
        [
            2.67375563e1,
            1.00721050e-2,
            -7.58485453e2,
            -4.70624175,
            0.0,
            -3.75994338e-6,
            0.0,
            0.0,
        ],
        [-7.41559626, 0.0, 3.22892989e2, 1.16438557, 0.0, 0.0, 0.0, -5.94578140],
        [
            -3.30531334,
            -1.29807848e-3,
            9.12712100e1,
            5.864450181e-1,
            0.0,
            4.95713573e-7,
            0.0,
            0.0,
        ],
    ]
)

param_K2SO4 = np.array(
    [
        [
            4.07908797e1,
            8.26906675e-3,
            -1.418242998e3,
            -6.74728848,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [-1.31669651e1, 2.35793239e-2, 2.06712592e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.88e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

param_CaCl2 = np.array(
    [
        [
            -9.41895832e1,
            -4.04750026e-2,
            2.34550368e3,
            1.70912300e1,
            -9.22885841e-1,
            1.51488122e-5,
            -1.39082000e0,
            0.0,
        ],
        [3.4787, -1.5417e-2, 0.0, 0.0, 0.0, 3.1791e-5, 0.0, 0.0],
        [
            1.93056024e1,
            9.77090932e-3,
            -4.28383748e2,
            -3.57996343,
            8.82068538e-2,
            -4.62270238e-6,
            9.91113465,
            0.0,
        ],
    ]
)
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

# Table A2 (Millero and Pierrot, 1998; after Pabalan and Pitzer, 1987) valid 25 to 200degC
param_MgCl2 = np.array(
    [
        [0.576066, -9.31654e-04, 5.93915e-07],
        [2.60135, -0.0109438, 2.60169e-05],
        [0.059532, -2.49949e-04, 2.41831e-07],
    ]
)

param_MgSO4 = np.array(
    [
        [-1.0282, 8.4790e-03, -2.33667e-05, 2.1575e-08, 6.8402e-04, 0.21499],
        [-2.9596e-01, 9.4564e-04, 0.0, 0.0, 1.1028e-02, 3.3646],
        [4.2164e-01, -3.5726e-03, 1.0040e-05, -9.3744e-09, -3.5160e-04, 2.7972e-02],
    ]
)
# param_MgSO4 = np.array([[-1.0282, 8.4790E-03, -2.33667E-05, 2.1575E-08, 6.8402E-04, 0.21499],[-2.9596E-01, 9.4564E-04, 0.0, 0.0, 1.1028E-02, 3.3646], [1.0541E-01, -8.9316E-04, 2.51E-06, -2.3436E-09, -8.7899E-05, 0.006993]])  # Cparams corrected after Pabalan and Pitzer ... but note that column lists Cmx not Cphi(=4xCmx) ... MP98 is correct

# Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC)
# param_NaHSO4 = np.array([[0.030101, -0.362E-3, 0.0], [0.818686, -0.019671, 0.0], [0.0, 0.0, 0.0]])  # corrected after Pierrot et al., 1997
param_NaHSO4 = np.array(
    [
        [0.0544, -1.8478e-3, 5.3937e-5],
        [0.3826401, -1.8431e-2, 0.0],
        [0.003905, 0.0, 0.0],
    ]
)  # corrected after Pierrot and Millero, 1997

param_NaHCO3 = np.array([
    [0.028, 1.0e-3, -2.6e-5 / 2],
    [0.044, 1.1e-3, -4.3e-5 / 2],
    [0.0, 0.0, 0.0]
    ])  # corrected after Peiper and Pitzer 1982

# param_Na2SO4 = np.array([[6.536438E-3, -30.197349, -0.20084955],
#                             [0.8742642, -70.014123, 0.2962095],
#                             [7.693706E-3, 4.5879201, 0.019471746]])  # corrected according to Hovey et al 1993; note also that alpha = 1.7, not 2
param_Na2SO4_Moller = np.array([
        [
            81.6920027,
            0.0301104957,
            -2321.93726,
            -14.3780207,
            -0.666496111,
            -1.03923656e-05,
            0,
        ],
        [
            1004.63018,
            0.577453682,
            -21843.4467,
            -189.110656,
            -0.2035505488,
            -0.000323949532,
            1467.72243,
        ],
        [
            -80.7816886,
            -0.0354521126,
            2024.3883,
            14.619773,
            -0.091697474,
            1.43946005e-05,
            -2.42272049,
        ]
    ]
)
# Moller 1988 parameters as used in Excel MIAMI code !!!!!! careful this formula assumes alpha1=2 as opposed to alpha1=1.7 for the Hovey parameters
# XXXXX - - > need to go to the calculation of beta's (to switch Hovey / Moller) and of B et al (to switch alpha1

# param_Na2CO3 = np.array([[0.0362, 1.79E-3, 1.694E-21], [1.51, 2.05E-3, 1.626E-19], [0.0052, 0.0, 0.0]])  # Millero and Pierrot referenced to Peiper and Pitzer
param_Na2CO3 = np.array([
        [0.0362, 1.79e-3, -4.22e-5 / 2],
        [1.51, 2.05e-3, -16.8e-5 / 2],
        [0.0052, 0.0, 0.0],
    ])  # Peiper and Pitzer 1982
# XXXX check below if Haynes 2003 is being used.

param_NaBOH4 = np.array([
    [-0.051, 5.264e-3, 0.0],
    [0.0961, -1.068e-2, 0.0],
    [0.01498, -15.7e-4, 0.0]]
)  # corrected after Simonson et al 1987 5th param should be e-2

# def Equation_Na2SO4_TabA3(T, ln_of_Tdiv29815, a):
#     return (a[:, 0] + a[:, 1] * ((1 / T) - (1 / 298.15)) + a[:, 2] * ln_of_Tdiv29815)

# Table A4 (Millero and Pierrot, 1998; after mutiple studies, at least valid 5 to 45degC)
param_KHCO3 = np.array([
    [-0.0107, 0.001, 0.0],
    [0.0478, 0.0011, 6.776e-21],
    [0.0, 0.0, 0.0]
    ])

param_K2CO3 = np.array([
    [0.1288, 1.1e-3, -5.1e-6],
    [1.433, 4.36e-3, 2.07e-5],
    [0.0005, 0.0, 0.0]
    ])

param_KBOH4 = np.array([
        [0.1469, 2.881e-3, 0.0],
        [-0.0989, -6.876e-3, 0.0],
        [-56.43 / 1000, -9.56e-3, 0.0],
    ])  # corrected after Simonson et al 1988
# same function as TabA3 "Equation_TabA3andTabA4andTabA5(TC,a)"

# Table A5 (Millero and Pierrot, 1998; after Simonson et al, 1987b; valid 5 - 55degC
param_MgBOH42 = np.array([
    [-0.623, 6.496e-3, 0.0],
    [0.2515, -0.01713, 0.0],
    [0.0, 0.0, 0.0]
    ])  # corrected after Simonson et al 1988 first param is negative

param_CaBOH42 = np.array([
    [-0.4462, 5.393e-3, 0.0], 
    [-0.868, -0.0182, 0.0], 
    [0.0, 0.0, 0.0]
    ])
param_SrBOH42 = param_CaBOH42  # see Table A6

# Table A7 (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC
param_KOH = np.array([
        [0.1298, -0.946e-5, 9.914e-4],
        [0.32, -2.59e-5, 11.86e-4],
        [0.0041, 0.0638e-5, -0.944e-4],
    ])

param_SrCl2 = np.array([
        [0.28575, -0.18367e-5, 7.1e-4],
        [1.66725, 0.0e-5, 28.425e-4],
        [-0.0013, 0.0e-5, 0.0e-4],
    ])

param_HCl = np.array([
            [1.2859, -2.1197e-3, -142.58770],
            [-4.4474, 8.425698e-3, 665.7882],
            [-0.305156, 5.16e-4, 45.521540],
        ])

pitzer_params = {k.replace('param_', ''): v.tolist() for k, v in locals().items() if 'param_' in k}
sources = {
    'NaCl': 'Table A1 (Millero and Pierrot, 1998; after Moller, 1988 & Greenberg and Moller, 1989) valid 0 to 250degC. Note that second value is changed to original ref (e-3 instead e01)',
    'KCl': 'Table A1 (Millero and Pierrot, 1998; after Moller, 1988 & Greenberg and Moller, 1989) valid 0 to 250degC',
    'K2SO4': 'Table A1 (Millero and Pierrot, 1998; after Moller, 1988 & Greenberg and Moller, 1989) valid 0 to 250degC',
    'CaCl2': 'Table A1 (Millero and Pierrot, 1998; after Moller, 1988 & Greenberg and Moller, 1989) valid 0 to 250degC',
    'MgCl2': 'Table A2 (Millero and Pierrot, 1998; after Pabalan and Pitzer, 1987) valid 25 to 200degC',
    'MgSO4': 'Table A2 (Millero and Pierrot, 1998; after Pabalan and Pitzer, 1987) valid 25 to 200degC',
    'NaHSO4': 'Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC). Corrected after Pierrot and Millero, 1997',
    'NaHCO3': 'Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC). Corrected after Peiper and Pitzer 1982',
    'Na2SO4_Moller': 'Moller 1988 parameters as used in Excel MIAMI code !!!!!! careful this formula assumes alpha1=2 as opposed to alpha1=1.7 for the Hovey parameters',
    'Na2CO3': 'Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC). Also noted as Peiper and Pitzer 1982.',
    'NaBOH4': 'Table A3 (Millero and Pierrot, 1998; after mutiple studies, at least valid 0 to 50degC). Corrected after Simonson et al 1987 5th param should be e-2',
    'KHCO3': 'Table A4 (Millero and Pierrot, 1998; after mutiple studies, at least valid 5 to 45degC)',
    'K2CO3': 'Table A4 (Millero and Pierrot, 1998; after mutiple studies, at least valid 5 to 45degC)',
    'KBOH4': 'Table A4 (Millero and Pierrot, 1998; after mutiple studies, at least valid 5 to 45degC). Corrected after Simonson et al 1988',
    'MgBOH42': 'Table A5 (Millero and Pierrot, 1998; after Simonson et al, 1987b; valid 5 - 55degC. Corrected after Simonson et al 1988 first param is negative',
    'CaBOH42': 'Table A5 (Millero and Pierrot, 1998; after Simonson et al, 1987b; valid 5 - 55degC',
    'SrBOH42': 'Table A5 (Millero and Pierrot, 1998; after Simonson et al, 1987b; valid 5 - 55degC',
    'KOH': 'Table A7 (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC',
    'SrCl2': 'Table A7 (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC',
    'HCl': 'Table A9 (Millero and Pierrot, 1998; after multiple studies; valid 0 - 50degC. beta1 first param corrected to negative according to original reference (Campbell et al)'
}

pitzer_params['sources'] = sources

with open(pkgrs.resource_filename('cbsyst', 'MyAMI/parameters/pitzer_params.json'), 'w') as f:
    json.dump(pitzer_params, f, indent=2)

##############################################################################
# Theta_positive and Theta_negative
##############################################################################

# Theta of positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
# Array to hold Theta values between ion two ions (for numbering see list above)
Theta_positive_base = np.zeros((6, 6))
Theta_positive_base[[0,5], [5,0]] = 0.0591 # H - Sr; modified by Temp
Theta_positive_base[[0,1], [1,0]] = 0.03416  # H - Na; modified by Temp
Theta_positive_base[[0,2], [2,0]] = 0.005 # H - K; modified by Temp
Theta_positive_base[[0,3], [3,0]] = 0.062 # H - Mg; modified by Temp
Theta_positive_base[[0,4], [4,0]] = 0.0612 # H - Ca; modified by Temp
Theta_positive_base[[1,2], [2,1]] = -5.02312111e-2 # Na - K; modified by Temp
Theta_positive_base[[1,3], [3,1]] = 0.07  # Na - Mg
Theta_positive_base[[1,4], [4,1]] = 0.05  # Na - Ca
# Theta_positive_base[[2,3], [3,2]] = 0.0  # K - Mg
Theta_positive_base[[2,4], [4,2]] = 0.1156  # K - Ca
Theta_positive_base[[5,1], [1,5]] = 0.07  # Sr - Na
Theta_positive_base[[5,2], [2,5]] = 0.01  # Sr - K
Theta_positive_base[[3,4], [4,3]] = 0.007  # Mg - Ca

# To be modified by temperature-sensitive adjustments for:
# Theta_positive_mod = np.zeros((6, 6, *TC.shape))
# Theta_positive_mod[[0,5], [5,0]] = 4.5e-4 * TC  # H - Sr
# Theta_positive_mod[[0,1], [1,0]] = -2.09e-4 * TC  # H - Na
# Theta_positive_mod[[0,2], [2,0]] = -2.275e-4 * TC  # H - K
# Theta_positive_mod[[0,3], [3,0]] = 3.275e-4 * TC  # H - Mg
# Theta_positive_mod[[0,4], [4,0]] = 3.275e-4 * TC  # H - Ca
# Theta_positive_mod[[1,2], [2,1]] = 14.0213141 / T  # Na - K

# Theta_positive = match_dims(Theta_positive_base, Theta_positive_mod) + Theta_positive_mod


# Theta of negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
# Array to hold Theta values between ion two ions (for numbering see list above)
Theta_negative_base = np.zeros((7, 7))
Theta_negative_base[[1,6], [6,1]] = 0.07  # Cl - SO4
Theta_negative_base[[1,5], [5,1]] = -0.092  # Cl - CO3, corrected after Pitzer and Peiper 1982
Theta_negative_base[[1,3], [3,1]] = 0.0359  # Cl - HCO3
Theta_negative_base[[1,2], [2,1]] = -0.0323 # Cl - BOH4
# Theta_negative_base[[3,5], [5,3]] = 0.0  # CO3 - HCO3
# Theta_negative_base[[4,6], [6,4]] = 0.0  # SO4 - HSO4
Theta_negative_base[[0,1], [1,0]] = -0.05 # OH - Cl
Theta_negative_base[[5,6], [6,5]] = 0.02  # SO4 - CO3
Theta_negative_base[[3,6], [6,3]] = 0.01  # SO4 - HCO3
Theta_negative_base[[2,6], [6,2]] = -0.012  # SO4 - BOH4
Theta_negative_base[[1,4], [4,1]] = -0.006  # HSO4 - Cl
Theta_negative_base[[0,6], [6,0]] = -0.013  # OH - SO4
Theta_negative_base[[3,0], [0,3]] = 0.1  # CO3 - OH #http: / /www.aim.env.uea.ac.uk / aim / accent4 / parameters.html

# To be modified by temperature-sensitive adjustments for:
# Theta_negative_mod = np.zeros((7, 7, *T.shape))
# Theta_negative_mod[[1,2], [2,1]] = -0.42333e-4 * TC - 21.926 * 1e-6 * TC ** 2  # Cl - BOH4
# Theta_negative_mod[[0,1], [1,0]] = 3.125e-4 * TC - 8.362 * 1e-6 * TC ** 2  # OH - Cl

# Theta_negative = match_dims(Theta_negative_base, Theta_negative_mod) + Theta_negative_mod

Theta_base = {
    'Theta_positive': Theta_positive_base.tolist(),
    'Theta_negative': Theta_negative_base.tolist()
}

with open(pkgrs.resource_filename('cbsyst', 'MyAMI/parameters/theta_bases.json'), 'w') as f:
    json.dump(Theta_base, f, indent=2)

##############################################################################
# Phi_NNP and Phi_PPN
##############################################################################

# Phi
# positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
# negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
# Phi_PPN holds the values for cation - cation - anion

Phi_PPN_base = np.zeros((6, 6, 7))  # Array to hold Theta values between ion two ions (for numbering see list above)
Phi_PPN_base[[1,2], [2,1], 1] = 1.34211308e-2  # Na-K-Cl
Phi_PPN_base[[1,2], [2,1], 6] = 3.48115174e-2  # Na-K-SO4
Phi_PPN_base[[1,3], [3,1], 1] = 0.0199  # Na-Mg-Cl
Phi_PPN_base[[1,4], [4,1], 1] = -7.6398  # Na-Ca-Cl. Spencer et al 1990 # -0.003  
Phi_PPN_base[[1,4], [4,1], 6] = -0.012  # Na-Ca-SO4
Phi_PPN_base[[2,3], [3,2], 1] = 0.02586  # K-Mg-Cl
Phi_PPN_base[[2,4], [4,2], 1] = 0.047627877  # K-Ca-Cl
# Phi_PPN_base[[2,4], [4,2], 6] = 0.0  # K-Ca-SO4
Phi_PPN_base[[0,5], [5,0], 1] = 0.0054  # H-Sr-Cl
Phi_PPN_base[[0,3], [3,0], 1] = 0.001  # H-Mg-Cl
Phi_PPN_base[[0,4], [4,0], 1] = 0.0008  # H-Ca-Cl
Phi_PPN_base[[5,1], [1,5], 1] = -0.015  # Sr-Na-Cl
Phi_PPN_base[[5,2], [2,5], 1] = -0.015  # Sr-K-Cl
Phi_PPN_base[[1,3], [3,1], 6] = -0.015  # Na-Mg-SO4
Phi_PPN_base[[2,3], [3,2], 6] = -0.048  # K-Mg-SO4
Phi_PPN_base[[3,4], [4,3], 1] = 4.15790220e1  # Spencer et al 1990 #-0.012  # Mg-Ca-Cl
Phi_PPN_base[[3,4], [4,3], 6] = 0.024  # Mg-Ca-SO4
Phi_PPN_base[[0,1], [1,0], 1] = 0.0002  # H-Na-Cl
# Phi_PPN_base[[0,1], [1,0], 6] = 0.0  # H-Na-SO4
Phi_PPN_base[[0,2], [2,0], 1] = -0.011  # H-K-Cl
Phi_PPN_base[[0,2], [2,0], 6] = 0.197  # H-K-SO4
# Phi_PPN[[0,2], [2,0], 1] = 0.197  # typo in original MyAMI
# this overwrites H-K-Cl with H-K-SO4 parameter, and leaves out H-K-SO4 parameter
# Doesn't seem to make much difference for [Ca] and [Mg], but may be important if
# changing [SO4]?


# Phi_NNP holds the values for anion - anion - cation
# Array to hold Theta values between ion two ions (for numbering see list above)
Phi_NNP_base = np.zeros((7, 7, 6))
Phi_NNP_base[[1,6], [6,1], 1] = -0.009  # Cl-SO4-Na
Phi_NNP_base[[1,6], [6,1], 2] = -0.21248147  # Cl-SO4-K
Phi_NNP_base[[1,6], [6,1], 4] = -0.018  # Cl-SO4-Ca
Phi_NNP_base[[1,5], [5,1], 4] = 0.016  # Cl-CO3-Ca
Phi_NNP_base[[1,3], [3,1], 1] = -0.0143  # Cl-HCO3-Na
Phi_NNP_base[[1,2], [2,1], 1] = -0.0132  # Cl-BOH4-Na
Phi_NNP_base[[1,2], [2,1], 3] = -0.235  # Cl-BOH4-Mg
Phi_NNP_base[[1,2], [2,1], 4] = -0.8  # Cl-BOH4-Ca
Phi_NNP_base[[4,6], [6,4], 1] = 0.0  # HSO4-SO4-Na
Phi_NNP_base[[3,5], [5,3], 1] = 0.0  # CO3-HCO3-Na
Phi_NNP_base[[3,5], [5,3], 2] = 0.0  # CO3-HCO3-K
Phi_NNP_base[[1,6], [6,1], 3] = -0.004  # Cl-SO4-Mg
Phi_NNP_base[[1,3], [3,1], 3] = -0.0196  # Cl-HCO3-Mg
Phi_NNP_base[[6,5], [5,6], 1] = -0.005  # SO4-CO3-Na
Phi_NNP_base[[6,5], [5,6], 2] = -0.009  # SO4-CO3-K
Phi_NNP_base[[6,3], [3,6], 1] = -0.005  # SO4-HCO3-Na
Phi_NNP_base[[6,3], [3,6], 3] = -0.161  # SO4-HCO3-Mg
Phi_NNP_base[[4,1], [1,4], 1] = -0.006  # HSO4-Cl-Na
Phi_NNP_base[[4,6], [6,4], 2] = -0.0677  # HSO4-SO4-K
Phi_NNP_base[[0,1], [1,0], 1] = -0.006  # OH-Cl-Na
Phi_NNP_base[[0,1], [1,0], 2] = -0.006  # OH-Cl-K
Phi_NNP_base[[0,1], [1,0], 4] = -0.025  # OH-Cl-Ca
Phi_NNP_base[[0,6], [6,0], 1] = -0.009  # OH-SO4-Na
Phi_NNP_base[[0,6], [6,0], 2] = -0.05  # OH-SO4-K

Phi_base = {
    'Phi_PPN': Phi_PPN_base.tolist(),
    'Phi_NNP': Phi_NNP_base.tolist()
}

with open(pkgrs.resource_filename('cbsyst', 'MyAMI/parameters/phi_bases.json'), 'w') as f:
    json.dump(Phi_base, f, indent=2)