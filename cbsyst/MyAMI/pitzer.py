import numpy as np
from .helpers import expand_dims, match_dims, standard_seawater, calc_Istr
from kgen import calc_K

# TODO: new file for user-facing functions.

def calc_seawater_ions(Sal=35., Na=None, K=None, Ca=None, Mg=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None):
    """
    Returns modern seawater composition with given ions modified at specified salinity. 

    All units are mol/kg.

    NOTE: Assumes that the provided ionic concentrations are at Sal=35.

    Returns
    -------
    tuple of arrays
        Containing (cations, anions) in the order:
        cations = [H, Na, K, Mg, Ca, Sr]
        anions = [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4] 

    """
    modified_cations = [None, Na, K, Mg, Ca, Sr]
    modified_anions = [None, Cl, BOH4, HCO3, None, CO3, SO4]

    m_cations, m_anions = standard_seawater()

    m_cations = np.full(
        (m_cations.size, *Sal.shape),
        expand_dims(m_cations, Sal)
        )

    m_anions = np.full(
        (m_anions.size, *Sal.shape),
        expand_dims(m_anions, Sal)
        )

    for i, m in enumerate(modified_cations):
        if m is not None:
            m_cations[i] = m
    
    for i, m in enumerate(modified_anions):
        if m is not None:
            m_anions[i] = m

    sal_factor = Sal / 35.

    return m_cations * sal_factor, m_anions * sal_factor


def calculate_gKs(Tc, Sal, Na=None, K=None, Ca=None, Mg=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None,
                  beta_0=None, beta_1=None, beta_2=None, C_phi=None, Theta_negative=None, Theta_positive=None, Phi_NNP=None, Phi_PPN=None, C1_HSO4=None):
    """
    Calculate Ks at given conditions using MyAMI model.
    """

    m_cation, m_anion = calc_seawater_ions(Sal, Na=Na, K=K, Mg=Mg, Ca=Ca, Sr=Sr, Cl=Cl, BOH4=BOH4, HCO3=HCO3, CO3=CO3, SO4=SO4)

    Istr = calc_Istr(Sal)

    [
        gamma_cation,
        gamma_anion,
        alpha_Hsws,
        alpha_Ht,
        alpha_OH,
        alpha_CO3,
    ] = CalculateGammaAndAlphas(
        Tc, Sal, Istr, m_cation, m_anion, 
        beta_0=beta_0, beta_1=beta_1, beta_2=beta_2, C_phi=C_phi,
        Theta_negative=Theta_negative, Theta_positive=Theta_positive,
        Phi_NNP=Phi_NNP, Phi_PPN=Phi_PPN, C1_HSO4=C1_HSO4)

    gammaT_OH = gamma_anion[0] * alpha_OH
    gammaT_BOH4 = gamma_anion[2]
    gammaT_HCO3 = gamma_anion[3]
    gammaT_CO3 = gamma_anion[5] * alpha_CO3

    # gammaT_Hsws = gamma_cation[0] * alpha_Hsws
    gammaT_Ht = gamma_cation[0] * alpha_Ht
    gammaT_Ca = gamma_cation[4]

    [gammaCO2, gammaCO2gas, gammaB] = gammaCO2_gammaB_fn(Tc, m_anion, m_cation)

    gKspC = 1 / gammaT_Ca / gammaT_CO3
    gKspA = 1 / gammaT_Ca / gammaT_CO3
    gK1 = 1 / gammaT_Ht / gammaT_HCO3 * gammaCO2
    gK2 = 1 / gammaT_Ht / gammaT_CO3 * gammaT_HCO3
    gKW = 1 / gammaT_Ht / gammaT_OH
    gKB = 1 / gammaT_BOH4 / gammaT_Ht * gammaB
    gK0 = 1 / gammaCO2 * gammaCO2gas
    gKHSO4 = 1 / gamma_anion[6] / gammaT_Ht * gamma_anion[4]

    return gKspC, gK1, gK2, gKW, gKB, gKspA, gK0, gKHSO4


def CalculateGammaAndAlphas(Tc, Sal, Istr, m_cation, m_anion,
                  beta_0=None, beta_1=None, beta_2=None, C_phi=None, Theta_negative=None, Theta_positive=None, Phi_NNP=None, Phi_PPN=None, C1_HSO4=None):
    """Calculate Gammas and Alphas for K calculations.

    Parameters
    ----------
    Tc : array-like
        Temperature in Celcius
    S : array-like
        Salinity in PSU
    Istr : array-like
        Ionic strength of solution
    m_cation : array-like
        Matrix of major cations in seawater in mol/kg in order:
        [H, Na, K, Mg, Ca, Sr]
    m_anion : array-like
        Matrix of major anions in seawater in mol/kg in order:
        [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4]

    Returns
    -------
    list of arrays
        [gamma_cation, gamma_anion, alpha_Hsws, alpha_Ht, alpha_OH, alpha_CO3]
    """
    # Testbed case T=25C, I=0.7, seawatercomposition
    T = Tc + 273.15
    sqrtI = np.sqrt(Istr)
    
    # make tables of ion charges used in later calculations

    # cation order: [H, Na, K, Mg, Ca, Sr]
    cation_charges = np.array([1, 1, 1, 2, 2, 2])
    Z_cation = np.full(
        (cation_charges.size, *Tc.shape),
        expand_dims(cation_charges, Tc)
        )

    # anion order: [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4]
    anion_charges = np.array([-1, -1, -1, -1, -1, -2, -2])
    Z_anion = np.full(
        (anion_charges.size, *Tc.shape),
        expand_dims(anion_charges, Tc)
        )   

    ##########################################################################

    A_phi = (
        3.36901532e-01
        - 6.32100430e-04 * T
        + 9.14252359 / T
        - 1.35143986e-02 * np.log(T)
        + 2.26089488e-03 / (T - 263)
        + 1.92118597e-6 * T * T
        + 4.52586464e01 / (680 - T)
    )  # note correction of last parameter, E + 1 instead of E-1
    # A_phi = 8.66836498e1 + 8.48795942e-2 * T - 8.88785150e-5 * T * T +
    # 4.88096393e-8 * T * T * T -1.32731477e3 / T - 1.76460172e1 * np.log(T)
    # # Spencer et al 1990

    f_gamma = -A_phi * (sqrtI / (1 + 1.2 * sqrtI) + (2 / 1.2) * np.log(1 + 1.2 * sqrtI))

    # E_cat = sum(m_cation * Z_cation)
    E_an = -sum(m_anion * Z_anion)
    E_cat = -E_an

    BMX_phi = beta_0 + beta_1 * np.exp(-2 * sqrtI)
    BMX = beta_0 + (beta_1 / (2 * Istr)) * (1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI))
    BMX_apostroph = (beta_1 / (2 * Istr * Istr)) * (-1 + (1 + (2 * sqrtI) + (2 * sqrtI)) * np.exp(-2 * sqrtI))
    CMX = C_phi / (2 * np.sqrt(-np.expand_dims(Z_anion, 0) * np.expand_dims(Z_cation, 1)))
    
    # BMX* and CMX are calculated differently for 2:2 ion pairs, corrections
    # below  # § alpha2= 6 for borates ... see Simonson et al 1988
    
    # MgBOH42
    cat, an = 3, 2
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )
    
    # MgSO4
    cat, an = 3, 6 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-12 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (72 * Istr)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (72 * Istr * Istr)) * (
        -1 - (1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)
    )
    # BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr)) * (-1 + (1 + 1.4
    # * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)) + (beta_2[cat, an] / (72 *
    # Istr)) * (-1-(1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)) # not 1 /
    # (0.98 * Istr * Istr) ... compare M&P98 equation A17 with Pabalan and Pitzer
    # 1987 equation 15c / 16b
    
    # CaBOH42
    cat, an = 4, 2 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )
    
    # CaSO4
    cat, an = 4, 6
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-12 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (72 * Istr)) * (1 - (1 + 12 * sqrtI) * np.exp(-12 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (72 * Istr)) * (
        -1 - (1 + 12 * sqrtI + 72 * Istr) * np.exp(-12 * sqrtI)
    )

    # SrBOH42
    cat, an = 5, 2 
    BMX_phi[cat, an] = (
        beta_0[cat, an]
        + beta_1[cat, an] * np.exp(-1.4 * sqrtI)
        + beta_2[cat, an] * np.exp(-6 * sqrtI)
    )
    BMX[cat, an] = (
        beta_0[cat, an]
        + (beta_1[cat, an] / (0.98 * Istr))
        * (1 - (1 + 1.4 * sqrtI) * np.exp(-1.4 * sqrtI))
        + (beta_2[cat, an] / (18 * Istr)) * (1 - (1 + 6 * sqrtI) * np.exp(-6 * sqrtI))
    )
    BMX_apostroph[cat, an] = (beta_1[cat, an] / (0.98 * Istr * Istr)) * (
        -1 + (1 + 1.4 * sqrtI + 0.98 * Istr) * np.exp(-1.4 * sqrtI)
    ) + (beta_2[cat, an] / (18 * Istr)) * (
        -1 - (1 + 6 * sqrtI + 18 * Istr) * np.exp(-6 * sqrtI)
    )

    # H-SO4
    cat, an = 0, 6
    # BMX* is calculated with T-dependent alpha for H-SO4; see Clegg et al.,
    # 1994 --- Millero and Pierrot are completly off for this ion pair
    xClegg = (2 - 1842.843 * (1 / T - 1 / 298.15)) * sqrtI
    # xClegg = (2) * sqrtI
    gClegg = 2 * (1 - (1 + xClegg) * np.exp(-xClegg)) / (xClegg * xClegg)
    # alpha = (2 - 1842.843 * (1 / T - 1 / 298.15)) see Table 6 in Clegg et al
    # 1994
    BMX[cat, an] = beta_0[cat, an] + beta_1[cat, an] * gClegg
    BMX_apostroph[cat, an] = beta_1[cat, an] / Istr * (np.exp(-xClegg) - gClegg)

    CMX[cat, an] = C_phi[cat, an] + 4 * C1_HSO4 * (
        6
        - (6 + 2.5 * sqrtI * (6 + 3 * 2.5 * sqrtI + 2.5 * sqrtI * 2.5 * sqrtI))
        * np.exp(-2.5 * sqrtI)
    ) / (
        2.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI * 2.5 * sqrtI
    )  # w = 2.5 ... see Clegg et al., 1994

    # unusual alpha=1.7 for Na2SO4
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (2.89 * Istr)) * 2 * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))
    # BMX[1, 6] = beta_0[1, 6] + (beta_1[1, 6] / (1.7 * Istr)) * (1 - (1 + 1.7 * sqrtI) * np.exp(-1.7 * sqrtI))

    # BMX[4, 6] =BMX[4, 6] * 0  # knock out Ca-SO4
    
    R = (m_anion * np.expand_dims(m_cation, 1) * BMX_apostroph).sum((0,1))
    S = (m_anion * np.expand_dims(m_cation, 1) * CMX).sum((0,1))

    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S

    # Original ln_gamma_anion calculation loop:
    ln_gamma_anion = Z_anion * Z_anion * (f_gamma + R) + Z_anion * S
    for an in range(0, 7):
        for cat in range(0, 6):
            ln_gamma_anion[an] += 2 * m_cation[cat] * (
                BMX[cat, an] + E_cat * CMX[cat, an]
            )
        for an2 in range(0, 7):
            ln_gamma_anion[an] += m_anion[an2] * (
                2 * Theta_negative[an, an2]
            )
        for an2 in range(0, 7):
            for cat in range(0, 6):
                ln_gamma_anion[an] += (
                    m_anion[an2] * m_cation[cat] * Phi_NNP[an, an2, cat]
                )
        for cat in range(0, 6):
            for cat2 in range(cat + 1, 6):
                ln_gamma_anion[an] += (
                    m_cation[cat] * m_cation[cat2] * Phi_PPN[cat, cat2, an]
                )
    
    # vectorised ln_gamma_anion calculation:
    # cat, cat2 = np.triu_indices(6, 1)
    # ln_gamma_anion = (
    #     Z_anion * Z_anion * (f_gamma + R) + Z_anion * S + 
    #     (2 * np.expand_dims(m_cation, 1) * (BMX + E_cat * CMX)).sum(0) + 
    #     (np.expand_dims(m_anion, 1) * 2 * Theta_negative).sum(0) + 
    #     (np.expand_dims(m_anion, (0,2)) * np.expand_dims(m_cation, (0,1)) * Phi_NNP).sum(axis=(1,2)) +
    #     (np.expand_dims(m_cation[cat], 1) * np.expand_dims(m_cation[cat2], 1) * Phi_PPN[cat, cat2]).sum(axis=0)
    # )  # TODO - could be simplified further?
    gamma_anion = np.exp(ln_gamma_anion)


    # ln_gammaCl = Z_anion[1] * Z_anion[1] * f_gamma + R - S

    # Original ln_gamma_cation calculation loop:
    ln_gamma_cation = Z_cation * Z_cation * (f_gamma + R) + Z_cation * S
    for cat in range(0, 6):
        for an in range(0, 7):
            ln_gamma_cation[cat] += 2 * m_anion[an] * (
                BMX[cat, an] + E_cat * CMX[cat, an]
            )
        for cat2 in range(0, 6):
            ln_gamma_cation[cat] += m_cation[cat2] * (2 * Theta_positive[cat, cat2])
        for cat2 in range(0, 6):
            for an in range(0, 7):
                ln_gamma_cation[cat] += (
                    m_cation[cat2] * m_anion[an] * Phi_PPN[cat, cat2, an]
                )
        for an in range(0, 7):
            for an2 in range(an + 1, 7):
                ln_gamma_cation[cat] += (
                    + m_anion[an] * m_anion[an2] * Phi_NNP[an, an2, cat]
                )

    # vectorised ln_gamma_cation calculation:
    # an, an2 = np.triu_indices(7, 1)
    # ln_gamma_cation = (
    #     Z_cation * Z_cation * (f_gamma + R) + Z_cation * S +
    #     (2 * np.expand_dims(m_anion, 0) * (BMX + E_cat * CMX)).sum(axis=1) +
    #     (np.expand_dims(m_cation, 1) * (2 * Theta_positive)).sum(axis=0) +
    #     (np.expand_dims(m_cation, (0,2)) * np.expand_dims(m_anion, (0,1)) * Phi_PPN).sum(axis=(1,2))+
    #     (np.expand_dims(m_anion[an], 1) * np.expand_dims(m_anion[an2], 1) * Phi_NNP[an, an2]).sum(axis=0)
    # )  # TODO - could be simplified further?
    gamma_cation = np.exp(ln_gamma_cation)

    # choice of pH-scale = total pH-scale [H]T = [H]F + [HSO4]
    # so far gamma_H is the [H]F activity coefficient (= free-H pH-scale)
    # thus, conversion is required
    # * (gamma_anion[4] / gamma_anion[6] / gamma_cation[0])
    
    K_HSO4_conditional = calc_K(k='KS', TempC=Tc, Sal=Sal)
    K_HF_conditional = calc_K(k='KF', TempC=Tc, Sal=Sal)
    
    # print (gamma_anion[4], gamma_anion[6], gamma_cation[0])
    # alpha_H = 1 / (1+ m_anion[6] / K_HSO4_conditional + 0.0000683 / (7.7896E-4 * 1.1 / 0.3 / gamma_cation[0]))
    alpha_Hsws = 1 / (
        # 1 + m_anion[6] / K_HSO4_conditional + 0.0000683 / (supplyKHF(T, sqrtI))
        1 + m_anion[6] / K_HSO4_conditional + 0.0000683 / K_HF_conditional
    )
    alpha_Ht = 1 / (1 + m_anion[6] / K_HSO4_conditional)
    # alpha_H = 1 / (1+ m_anion[6] / K_HSO4_conditional)

    # A number of ion pairs are calculated explicitly: MgOH, CaCO3, MgCO3, SrCO3
    # since OH and CO3 are rare compared to the anions the anion alpha (free /
    # total) are assumed to be unity
    gamma_MgCO3 = 1
    gamma_CaCO3 = gamma_MgCO3
    gamma_SrCO3 = gamma_MgCO3

    b0b1CPhi_MgOH = np.array([-0.1, 1.658, 0, 0.028])
    BMX_MgOH = b0b1CPhi_MgOH[0] + (b0b1CPhi_MgOH[1] / (2 * Istr)) * (
        1 - (1 + 2 * sqrtI) * np.exp(-2 * sqrtI)
    )
    ln_gamma_MgOH = 1 * (f_gamma + R) + (1) * S
    # interaction between MgOH-Cl affects MgOH gamma
    ln_gamma_MgOH = ln_gamma_MgOH + 2 * m_anion[1] * (
        BMX_MgOH + E_cat * b0b1CPhi_MgOH[2]
    )
    # interaction between MgOH-Mg-OH affects MgOH gamma
    ln_gamma_MgOH = ln_gamma_MgOH + m_cation[3] * m_anion[1] * b0b1CPhi_MgOH[3]
    gamma_MgOH = np.exp(ln_gamma_MgOH)

    K_MgOH = np.power(10, -(3.87 - 501.6 / T)) / (
        gamma_cation[3] * gamma_anion[0] / gamma_MgOH
    )
    K_MgCO3 = np.power(10, -(1.028 + 0.0066154 * T)) / (
        gamma_cation[3] * gamma_anion[5] / gamma_MgCO3
    )
    K_CaCO3 = np.power(10, -(1.178 + 0.0066154 * T)) / (
        gamma_cation[4] * gamma_anion[5] / gamma_CaCO3
    )
    # K_CaCO3 = np.power(10, (-1228.732 - 0.299444 * T + 35512.75 / T +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] / gamma_CaCO3) # Plummer and Busenberg82
    # K_MgCO3 = np.power(10, (-1228.732 +(0.15) - 0.299444 * T + 35512.75 / T
    # +485.818 * np.log10(T))) / (gamma_cation[4] * gamma_anion[5] /
    # gamma_CaCO3)# Plummer and Busenberg82
    K_SrCO3 = np.power(10, -(1.028 + 0.0066154 * T)) / (
        gamma_cation[5] * gamma_anion[5] / gamma_SrCO3
    )

    alpha_OH = 1 / (1 + (m_cation[3] / K_MgOH))
    alpha_CO3 = 1 / (
        1 + (m_cation[3] / K_MgCO3) + (m_cation[4] / K_CaCO3) + (m_cation[5] / K_SrCO3)
    )

    return gamma_cation, gamma_anion, alpha_Hsws, alpha_Ht, alpha_OH, alpha_CO3

def gammaCO2_gammaB_fn(Tc, m_an, m_cat):
    T = Tc + 273.15
    lnT = np.log(T)

    m_ion = np.array(
        [m_cat[0], m_cat[1], m_cat[2], m_cat[3], m_cat[4], m_an[1], m_an[6]]
    )

    param_lamdaCO2 = np.zeros([7, 5])
    param_lamdaCO2[0, :] = [0, 0, 0, 0, 0]  # H
    param_lamdaCO2[1, :] = [
        -5496.38465,
        -3.326566,
        0.0017532,
        109399.341,
        1047.021567,
    ]  # Na
    param_lamdaCO2[2, :] = [
        2856.528099,
        1.7670079,
        -0.0009487,
        -55954.1929,
        -546.074467,
    ]  # K
    param_lamdaCO2[3, :] = [
        -479.362533,
        -0.541843,
        0.00038812,
        3589.474052,
        104.3452732,
    ]  # Mg
    # param_lamdaCO2[3, :] = [9.03662673e+03, 5.08294701e+00, -2.51623005e-03, -1.88589243e+05, -1.70171838e+03]  # Mg refitted
    param_lamdaCO2[4, :] = [
        -12774.6472,
        -8.101555,
        0.00442472,
        245541.5435,
        2452.50972,
    ]  # Ca
    # param_lamdaCO2[4, :] = [-8.78153999e+03, -5.67606538e+00, 3.14744317e-03, 1.66634223e+05, 1.69112982e+03]  # Ca refitted
    param_lamdaCO2[5, :] = [
        1659.944942,
        0.9964326,
        -0.00052122,
        -33159.6177,
        -315.827883,
    ]  # Cl
    param_lamdaCO2[6, :] = [
        2274.656591,
        1.8270948,
        -0.00114272,
        -33927.7625,
        -457.015738,
    ]  # SO4

    param_zetaCO2 = np.zeros([2, 6, 5])
    param_zetaCO2[0, 0, :] = [
        -804.121738,
        -0.470474,
        0.000240526,
        16334.38917,
        152.3838752,
    ]  # Cl & H
    param_zetaCO2[0, 1, :] = [
        -379.459185,
        -0.258005,
        0.000147823,
        6879.030871,
        73.74511574,
    ]  # Cl & Na
    param_zetaCO2[0, 2, :] = [
        -379.686097,
        -0.257891,
        0.000147333,
        6853.264129,
        73.79977116,
    ]  # Cl & K
    param_zetaCO2[0, 3, :] = [
        -1342.60256,
        -0.772286,
        0.000391603,
        27726.80974,
        253.62319406,
    ]  # Cl & Mg
    param_zetaCO2[0, 4, :] = [
        -166.06529,
        -0.018002,
        -0.0000247349,
        5256.844332,
        27.377452415,
    ]  # Cl & Ca
    param_zetaCO2[1, 1, :] = [
        67030.02482,
        37.930519,
        -0.0189473,
        -1399082.37,
        -12630.27457,
    ]  # SO4 & Na
    param_zetaCO2[1, 2, :] = [
        -2907.03326,
        -2.860763,
        0.001951086,
        30756.86749,
        611.37560512,
    ]  # SO4 & K
    param_zetaCO2[1, 3, :] = [
        -7374.24392,
        -4.608331,
        0.002489207,
        143162.6076,
        1412.302898,
    ]  # SO4 & Mg

    lamdaCO2 = np.zeros((7, *Tc.shape))
    for ion in range(0, 7):
        lamdaCO2[ion] = (
            param_lamdaCO2[ion, 0]
            + param_lamdaCO2[ion, 1] * T
            + param_lamdaCO2[ion, 2] * T ** 2
            + param_lamdaCO2[ion, 3] / T
            + param_lamdaCO2[ion, 4] * lnT
        )

    zetaCO2 = np.zeros([2, 5, *Tc.shape])
    for ion in range(0, 5):
        zetaCO2[0, ion] = (
            param_zetaCO2[0, ion, 0]
            + param_zetaCO2[0, ion, 1] * T
            + param_zetaCO2[0, ion, 2] * T ** 2
            + param_zetaCO2[0, ion, 3] / T
            + param_zetaCO2[0, ion, 4] * lnT
        )
    for ion in range(1, 4):
        zetaCO2[1, ion] = (
            param_zetaCO2[1, ion, 0]
            + param_zetaCO2[1, ion, 1] * T
            + param_zetaCO2[1, ion, 2] * T ** 2
            + param_zetaCO2[1, ion, 3] / T
            + param_zetaCO2[1, ion, 4] * lnT
        )

    
    # original calculation:
    # ln_gammaCO2 = 0
    # for ion in range(0, 7):
    #     ln_gammaCO2 = ln_gammaCO2 + m_ion[ion] * 2 * lamdaCO2[ion]
    
    # vectorised calculation:
    ln_gammaCO2 = (m_ion * 2 * lamdaCO2).sum(0)

    # for cat in range(0, 5):
    # ln_gammaCO2 = ln_gammaCO2 + m_ion[5] * m_ion[cat] * zetaCO2[0, cat] + m_ion[6] * m_ion[cat] * zetaCO2[1, cat]

    gammaCO2 = np.exp(ln_gammaCO2)  # as according to He and Morse 1993
    # gammaCO2 = np.power(10, ln_gammaCO2) # pK1 is "correct if log-base 10 is assumed

    gammaCO2gas = np.exp(
        1
        / (
            8.314462175
            * T
            * (0.10476 - 61.0102 / T - 660000 / T / T / T - 2.47e27 / np.power(T, 12))
        )
    )

    ##########################
    # CALCULATION OF gammaB
    lamdaB = np.array([0, -0.097, -0.14, 0, 0, 0.091, 0.018])  # Felmy and Wear 1986
    # lamdaB = np.array([0.109, 0.028, -0.026, 0.191, 0.165, 0, -0.205]) #Chanson and Millero 2006
    
    # original calculation:

    # ln_gammaB = m_ion[1] * m_ion[6] * 0.046  # tripple ion interaction Na-SO4
    # for ion in range(0, 7):
    #     ln_gammaB = ln_gammaB + m_ion[ion] * 2 * lamdaB[ion]
    
    # vectorised calculation:
    ln_gammaB = m_ion[1] * m_ion[6] * 0.046 + (m_ion * 2 * match_dims(lamdaB, m_ion)).sum(0)
    
    gammaB = np.exp(ln_gammaB)  # as according to Felmy and Wear 1986
    # print gammaB

    return gammaCO2, gammaCO2gas, gammaB