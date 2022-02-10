import numpy as np

# TODO: Use Kgen instead of local definitions.

# Functions from K_HSO4_thermo.py
# --------------------------------------
def supplyKHSO4(T, Istr):
    """
    Calculate KHSO4 for given temperature and salinity
    """
    Istr = pow(Istr, 1)
    # param_HSO4 = np.array([562.69486, -13273.75, -102.5154, 0.2477538, -1.117033e-4]) #Clegg et al. 1994
    # K_HSO4 = np.power(10,param_HSO4[0] + param_HSO4[1]/T + param_HSO4[2]*np.log(T) + param_HSO4[3]*T + param_HSO4[4]*T*T)

    param_HSO4 = np.array(
        [141.411, -4340.704, -23.4825, 0.016637]
    )  # Campbell et al. 1993
    # param_HSO4 = np.array([141.328, -4276.1, -23.093, 0]) #Dickson 1990
    # param_HSO4 = np.array([141.411, -4340.704, -23.4825, 0.016637])
    K_HSO4 = np.power(
        10,
        (
            param_HSO4[0]
            + param_HSO4[1] / T
            + param_HSO4[2] * np.log(T)
            + param_HSO4[3] * T
        ),
    )

    param_HSO4_cond = np.array(
        [
            141.328,
            -4276.1,
            -23.093,
            324.57,
            -13856,
            -47.986,
            -771.54,
            35474,
            114.723,
            -2698,
            1776,
        ]
    )  # Dickson 1990
    K_HSO4_cond = np.exp(
        param_HSO4_cond[0]
        + param_HSO4_cond[1] / T
        + param_HSO4_cond[2] * np.log(T)
        + np.sqrt(Istr)
        * (param_HSO4_cond[3] + param_HSO4_cond[4] / T + param_HSO4_cond[5] * np.log(T))
        + Istr
        * (param_HSO4_cond[6] + param_HSO4_cond[7] / T + param_HSO4_cond[8] * np.log(T))
        + param_HSO4_cond[9] / T * Istr * np.sqrt(Istr)
        + param_HSO4_cond[10] / T * Istr * Istr
    )

    return [K_HSO4_cond, K_HSO4]


# Functions from K_HF_cond.py
# --------------------------------------
def supplyKHF(T, sqrtI):
    return np.exp(1590.2 / T - 12.641 + 1.525 * sqrtI)
