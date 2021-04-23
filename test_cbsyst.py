import unittest
import os
import pandas as pd
import numpy as np
import cbsyst.carbon_fns as cf
import cbsyst.boron_fns as bf
from cbsyst.cbsyst import Csys, CBsys
from cbsyst.helpers import Bunch
from cbsyst.test_data.GLODAP_data.get_GLODAP_data import get_GLODAP


class BoronFnTestCase(unittest.TestCase):
    """Test all B functions"""

    def test_Boron_Fns(self):
        ref = Bunch(
            {
                "ABO3": 0.80882931,
                "ABO4": 0.80463763,
                "ABT": 0.80781778,
                "BO3": 328.50895695,
                "BO4": 104.49104305,
                "BT": 433.0,
                "Ca": 0.0102821,
                "H": 7.94328235e-09,
                "Ks": {
                    "K0": 0.02839188,
                    "K1": 1.42182814e-06,
                    "K2": 1.08155475e-09,
                    "KB": 2.52657299e-09,
                    "KSO4": 0.10030207,
                    "KW": 6.06386369e-14,
                    "KspA": 6.48175907e-07,
                    "KspC": 4.27235093e-07,
                },
                "Mg": 0.0528171,
                "S": 35.0,
                "T": 25.0,
                "alphaB": 1.02725,
                "dBO3": 46.30877684,
                "dBO4": 18.55320208,
                "dBT": 39.5,
                "pHtot": 8.1,
            }
        )

        Ks = Bunch(ref.Ks)

        # Speciation
        self.assertAlmostEqual(
            bf.BT_BO3(ref.BT, ref.BO3, Ks), ref.H, msg="BT_BO3", places=6
        )

        self.assertAlmostEqual(
            bf.BT_BO4(ref.BT, ref.BO4, Ks), ref.H, msg="BT_BO4", places=6
        )

        self.assertAlmostEqual(
            bf.pH_BO3(ref.pHtot, ref.BO3, Ks), ref.BT, msg="pH_BO3", places=6
        )

        self.assertAlmostEqual(
            bf.pH_BO4(ref.pHtot, ref.BO4, Ks), ref.BT, msg="pH_BO4", places=6
        )

        self.assertAlmostEqual(
            bf.cBO3(ref.BT, ref.H, Ks), ref.BO3, msg="cBO3", places=6
        )

        self.assertAlmostEqual(
            bf.cBO4(ref.BT, ref.H, Ks), ref.BO4, msg="cBO4", places=6
        )

        self.assertEqual(
            bf.chiB_calc(ref.H, Ks), 1 / (1 + Ks.KB / ref.H), msg="chiB_calc"
        )

        # Isotopes
        self.assertEqual(
            bf.alphaB_calc(ref.T), 1.0293 - 0.000082 * ref.T, msg="alphaB_calc"
        )

        self.assertAlmostEqual(
            bf.pH_ABO3(ref.pHtot, ref.ABO3, Ks, ref.alphaB),
            ref.ABT,
            msg="pH_ABO3",
            places=6,
        )

        self.assertAlmostEqual(
            bf.pH_ABO4(ref.pHtot, ref.ABO4, Ks, ref.alphaB),
            ref.ABT,
            msg="pH_ABO4",
            places=6,
        )

        self.assertAlmostEqual(
            bf.cABO3(ref.H, ref.ABT, Ks, ref.alphaB), ref.ABO3, msg="cABO3", places=6
        )

        self.assertAlmostEqual(
            bf.cABO4(ref.H, ref.ABT, Ks, ref.alphaB), ref.ABO4, msg="cABO4", places=6
        )

        # Isotope unit conversions
        self.assertAlmostEqual(
            bf.A11_2_d11(0.807817779214075), 39.5, msg="A11_2_d11", places=6
        )

        self.assertAlmostEqual(
            bf.d11_2_A11(39.5), 0.807817779214075, msg="d11_2_A11", places=6
        )

        return


class CarbonFnTestCase(unittest.TestCase):
    """Test all C functions"""

    def test_Carbon_Fns(self):
        ref = Bunch(
            {
                "BAlk": 104.39451552567037,
                "BT": 432.6,
                "CAlk": 2340.16132518,
                "CO2": 10.27549047,
                "CO3": 250.43681565,
                "Ca": 0.0102821,
                "DIC": 2100.0,
                "H": 7.94328235e-09,
                "HCO3": 1839.28769389,
                "HF": 0.00017903616862286758,
                "HSO4": 0.0017448760289520083,
                "Hfree": 0.0061984062104620289,
                "Ks": {
                    "K0": 0.028391881804015699,
                    "K1": 1.4218281371391736e-06,
                    "K2": 1.0815547472209423e-09,
                    "KB": 2.5265729902477677e-09,
                    "KF": 0.0023655007956108367,
                    "KP1": 0.024265183950721327,
                    "KP2": 1.0841036169428488e-06,
                    "KP3": 1.612502080867568e-09,
                    "KSO4": 0.10030207107256615,
                    "KSi": 4.1025099579058308e-10,
                    "KW": 6.019824161802715e-14,
                    "KspA": 6.4817590680119676e-07,
                    "KspC": 4.2723509278625912e-07,
                },
                "Mg": 0.0528171,
                "OH": 7.57850961,
                "P": None,
                "PAlk": 0.0,
                "S": 35.0,
                "SiAlk": 0.0,
                "T": 25.0,
                "TA": 2452.126228,
                "TF": 6.832583968836728e-05,
                "TP": 0.0,
                "TS": 0.028235434132860126,
                "TSi": 0.0,
                "fCO2": 361.91649919340324,
                "pCO2": 363.074540437976,
                "pHtot": 8.1,
                "unit": 1000000.0,
            }
        )

        Ks = Bunch(ref.Ks)

        self.assertAlmostEqual(
            cf.CO2_pH(ref.CO2, ref.pHtot, Ks), ref.DIC, msg="CO2_pH", places=6
        )

        self.assertAlmostEqual(
            cf.CO2_HCO3(ref.CO2, ref.HCO3, Ks)[0], ref.H, msg="CO2_HCO3 (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.CO2_CO3(ref.CO2, ref.CO3, Ks)[0], ref.H, msg="CO2_CO3 (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.CO2_TA(
                CO2=ref.CO2 / ref.unit,
                TA=ref.TA / ref.unit,
                BT=ref.BT / ref.unit,
                TP=ref.TP / ref.unit,
                TSi=ref.TSi / ref.unit,
                TS=ref.TS,
                TF=ref.TF,
                Ks=Ks,
            )[0],
            ref.pHtot,
            msg="CO2_TA",
            places=6,
        )
        self.assertAlmostEqual(
            cf.CO2_DIC(ref.CO2, ref.DIC, Ks)[0], ref.H, msg="CO2_DIC (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.pH_HCO3(ref.pHtot, ref.HCO3, Ks), ref.DIC, msg="pH_HCO3", places=6
        )

        self.assertAlmostEqual(
            cf.pH_CO3(ref.pHtot, ref.CO3, Ks), ref.DIC, msg="pH_CO3", places=6
        )

        self.assertAlmostEqual(
            cf.pH_TA(
                pH=ref.pHtot,
                TA=ref.TA / ref.unit,
                BT=ref.BT / ref.unit,
                TP=ref.TP / ref.unit,
                TSi=ref.TSi / ref.unit,
                TS=ref.TS,
                TF=ref.TF,
                Ks=Ks,
            )
            * ref.unit,
            ref.DIC,
            msg="pH_TA",
            places=6,
        )

        self.assertAlmostEqual(
            cf.pH_DIC(ref.pHtot, ref.DIC, Ks), ref.CO2, msg="pH_DIC", places=6
        )

        self.assertAlmostEqual(
            cf.HCO3_CO3(ref.HCO3, ref.CO3, Ks)[0], ref.H, msg="HCO3_CO3 (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.HCO3_TA(ref.HCO3 / ref.unit, ref.TA / ref.unit, ref.BT / ref.unit, Ks)[
                0
            ],
            ref.H,
            msg="HCO3_TA (zf)",
            places=6,
        )

        self.assertAlmostEqual(
            cf.HCO3_DIC(ref.HCO3, ref.DIC, Ks)[0], ref.H, msg="HCO3_DIC (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.CO3_TA(ref.CO3 / ref.unit, ref.TA / ref.unit, ref.BT / ref.unit, Ks)[0],
            ref.H,
            msg="CO3_TA (zf)",
            places=6,
        )

        self.assertAlmostEqual(
            cf.CO3_DIC(ref.CO3, ref.DIC, Ks)[0], ref.H, msg="CO3_DIC (zf)", places=6
        )

        self.assertAlmostEqual(
            cf.TA_DIC(
                TA=ref.TA / ref.unit,
                DIC=ref.DIC / ref.unit,
                BT=ref.BT / ref.unit,
                TP=ref.TP / ref.unit,
                TSi=ref.TSi / ref.unit,
                TS=ref.TS,
                TF=ref.TF,
                Ks=Ks,
            )[0],
            ref.pHtot,
            msg="TA_DIC",
            places=6,
        )

        self.assertAlmostEqual(
            cf.cCO2(ref.H, ref.DIC, Ks), ref.CO2, msg="cCO2", places=6
        )

        self.assertAlmostEqual(
            cf.cCO3(ref.H, ref.DIC, Ks), ref.CO3, msg="cCO3", places=6
        )

        self.assertAlmostEqual(
            cf.cHCO3(ref.H, ref.DIC, Ks), ref.HCO3, msg="cHCO3", places=6
        )

        (TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF) = cf.cTA(
            H=ref.H,
            DIC=ref.DIC / ref.unit,
            BT=ref.BT / ref.unit,
            TP=ref.TP / ref.unit,
            TSi=ref.TSi / ref.unit,
            TS=ref.TS,
            TF=ref.TF,
            Ks=Ks,
            mode="multi",
        )

        self.assertAlmostEqual(TA * ref.unit, ref.TA, msg="cTA - TA", places=6)

        self.assertAlmostEqual(CAlk * ref.unit, ref.CAlk, msg="cTA - CAlk", places=6)

        self.assertAlmostEqual(BAlk * ref.unit, ref.BAlk, msg="cTA - BAlk", places=6)

        self.assertAlmostEqual(PAlk * ref.unit, ref.PAlk, msg="cTA - PAlk", places=6)

        self.assertAlmostEqual(SiAlk * ref.unit, ref.SiAlk, msg="cTA - SiAlk", places=6)

        self.assertAlmostEqual(OH * ref.unit, ref.OH, msg="cTA - OH", places=6)

        self.assertAlmostEqual(Hfree * ref.unit, ref.Hfree, msg="cTA - Hfree", places=6)

        self.assertAlmostEqual(HSO4 * ref.unit, ref.HSO4, msg="cTA - HSO4", places=6)

        self.assertAlmostEqual(HF * ref.unit, ref.HF, msg="cTA - HF", places=6)

        self.assertAlmostEqual(
            cf.fCO2_to_CO2(ref.fCO2, Ks), ref.CO2, msg="fCO2_to_CO2", places=6
        )

        self.assertAlmostEqual(
            cf.CO2_to_fCO2(ref.CO2, Ks), ref.fCO2, msg="CO2_to_fCO2", places=6
        )

        self.assertAlmostEqual(
            cf.fCO2_to_pCO2(ref.fCO2, ref.T), ref.pCO2, msg="fCO2_to_pCO2", places=6
        )

        self.assertAlmostEqual(
            cf.pCO2_to_fCO2(ref.pCO2, ref.T), ref.fCO2, msg="pCO2_to_fCO2", places=6
        )

        return


class ReferenceDataTestCase(unittest.TestCase):
    """Test `yt` against reference data."""

    def test_Bockmon_Data_Csys(self):
        # Measured data from paper
        batch_A = {"S": 33.190, "TA": 2215.08, "DIC": 2015.72, "pH": 7.8796}

        batch_B = {"S": 33.186, "TA": 2216.26, "DIC": 2141.94, "pH": 7.5541}

        pH = np.array([batch_A["pH"], batch_B["pH"]])
        TA = np.array([batch_A["TA"], batch_B["TA"]])
        DIC = np.array([batch_A["DIC"], batch_B["DIC"]])
        S = np.array([batch_A["S"], batch_B["S"]])
        BT = 433.0

        # Csys calculations
        # TA from pH and DIC
        cTA = Csys(pHtot=pH, DIC=DIC, BT=BT, S_in=S)
        # Calculate % differences from measured
        dTA = 100 * (TA - cTA.TA) / TA

        self.assertLess(max(abs(dTA)), 0.2, msg="TA from DIC and pH")

        # pH from TA and DIC
        cpH = Csys(DIC=DIC, TA=TA, BT=BT, S_in=S)
        # Calculate % differences from measured
        dpH = 100 * (pH - cpH.pHtot) / pH

        self.assertLess(max(abs(dpH)), 0.2, msg="pH from TA and DIC")

        # DIC from pH and TA
        cDIC = Csys(pHtot=pH, TA=TA, BT=BT, S_in=S)
        # Calculate % differences from measured
        dDIC = 100 * (DIC - cDIC.DIC) / DIC

        self.assertLess(max(abs(dDIC)), 0.2, msg="DIC from TA and pH")

        return

    def test_Bockmon_Data_CBsys(self):
        # Measured data from paper
        batch_A = {"S": 33.190, "TA": 2215.08, "DIC": 2015.72, "pH": 7.8796}

        batch_B = {"S": 33.186, "TA": 2216.26, "DIC": 2141.94, "pH": 7.5541}

        pH = np.array([batch_A["pH"], batch_B["pH"]])
        TA = np.array([batch_A["TA"], batch_B["TA"]])
        DIC = np.array([batch_A["DIC"], batch_B["DIC"]])
        S = np.array([batch_A["S"], batch_B["S"]])
        BT = 433.0

        # Csys calculations
        # TA from pH and DIC
        cTA = CBsys(pHtot=pH, DIC=DIC, BT=BT, S_in=S)
        # Calculate % differences from measured
        dTA = 100 * (TA - cTA.TA) / TA

        self.assertLess(max(abs(dTA)), 0.2, msg="TA from DIC and pH")

        # pH from TA and DIC
        cpH = CBsys(DIC=DIC, TA=TA, BT=BT, S_in=S)
        # Calculate % differences from measured
        dpH = 100 * (pH - cpH.pHtot) / pH

        self.assertLess(max(abs(dpH)), 0.2, msg="pH from TA and DIC")

        # DIC from pH and TA
        cDIC = CBsys(pHtot=pH, TA=TA, BT=BT, S_in=S)
        # Calculate % differences from measured
        dDIC = 100 * (DIC - cDIC.DIC) / DIC

        self.assertLess(max(abs(dDIC)), 0.2, msg="DIC from TA and pH")

        return

    def test_Lueker_Data_Csys(self):
        """
        Need to incorporate nutrients!
        """
        ld = pd.read_csv(
            "cbsyst/test_data/Lueker2000/Lueker2000_Table3.csv", comment="#"
        )

        # Calculate using cbsys
        # TA from DIC and fCO2
        cTA = Csys(
            DIC=ld.DIC.values,
            fCO2=ld.fCO2.values,
            T_in=ld.Temp.values,
            S_in=ld.Sal.values,
        )
        dTA = ld.TA - cTA.TA
        dTA_median = np.median(dTA)
        dTA_pc95 = np.percentile(dTA, [2.5, 97.5])
        self.assertLessEqual(abs(dTA_median), 2.5, msg="TA Offset <= 2.5")
        self.assertTrue(all(abs(dTA_pc95 - dTA_median) <= 16), msg="TA 95% Conf <= 16")

        # fCO2 from TA and DIC
        cfCO2 = Csys(
            TA=ld.TA.values, DIC=ld.DIC.values, T_in=ld.Temp.values, S_in=ld.Sal.values
        )
        dfCO2 = ld.fCO2 - cfCO2.fCO2
        dfCO2_median = np.median(dfCO2)
        # dfCO2_pc95 = np.percentile(dfCO2, [2.5, 97.5])
        dfCO2_percent_offset = 100 * dfCO2 / ld.fCO2
        self.assertLessEqual(dfCO2_median, 2.5, msg="fCO2 Offset <= 2.5")
        self.assertLessEqual(np.std(dfCO2_percent_offset), 3, msg="fCO2 STD within 3%")
        # print(dfCO2_pc95)
        # self.assertTrue(all(abs(dfCO2_pc95) <= 70), msg='fCO2 95% Conc <= 70')

        # DIC from TA and fCO2
        cDIC = Csys(
            TA=ld.TA.values,
            fCO2=ld.fCO2.values,
            T_in=ld.Temp.values,
            S_in=ld.Sal.values,
        )
        dDIC = ld.DIC - cDIC.DIC
        dDIC_median = np.median(dDIC)
        dDIC_pc95 = np.percentile(dDIC, [2.5, 97.5])
        self.assertLessEqual(abs(dDIC_median), 2, msg="DIC Offset <= 2")
        self.assertTrue(all(abs(dDIC_pc95) <= 15), msg="DIC 95% Conc <= 15")

        return

    def test_Lueker_Data_CBsys(self):
        """
        Need to incorporate nutrients!
        """
        ld = pd.read_csv(
            "cbsyst/test_data/Lueker2000/Lueker2000_Table3.csv", comment="#"
        )

        # Calculate using cbsys
        # TA from DIC and fCO2
        cTA = CBsys(
            DIC=ld.DIC.values,
            fCO2=ld.fCO2.values,
            T_in=ld.Temp.values,
            S_in=ld.Sal.values,
        )
        dTA = ld.TA - cTA.TA
        dTA_median = np.median(dTA)
        dTA_pc95 = np.percentile(dTA, [2.5, 97.5])
        self.assertLessEqual(abs(dTA_median), 2.5, msg="TA Offset <= 2.5")
        self.assertTrue(all(abs(dTA_pc95 - dTA_median) <= 16), msg="TA 95% Conf <= 16")

        # fCO2 from TA and DIC
        cfCO2 = CBsys(
            TA=ld.TA.values, DIC=ld.DIC.values, T_in=ld.Temp.values, S_in=ld.Sal.values
        )
        dfCO2 = ld.fCO2 - cfCO2.fCO2
        dfCO2_median = np.median(dfCO2)
        # dfCO2_pc95 = np.percentile(dfCO2, [2.5, 97.5])
        dfCO2_percent_offset = 100 * dfCO2 / ld.fCO2
        self.assertLessEqual(dfCO2_median, 2.5, msg="fCO2 Offset <= 2.5")
        self.assertLessEqual(np.std(dfCO2_percent_offset), 3, msg="fCO2 STD within 3%")
        # print(dfCO2_pc95)
        # self.assertTrue(all(abs(dfCO2_pc95) <= 70), msg='fCO2 95% Conc <= 70')

        # DIC from TA and fCO2
        cDIC = CBsys(
            TA=ld.TA.values,
            fCO2=ld.fCO2.values,
            T_in=ld.Temp.values,
            S_in=ld.Sal.values,
        )
        dDIC = ld.DIC - cDIC.DIC
        dDIC_median = np.median(dDIC)
        dDIC_pc95 = np.percentile(dDIC, [2.5, 97.5])
        self.assertLessEqual(abs(dDIC_median), 2, msg="DIC Offset <= 2")
        self.assertTrue(all(abs(dDIC_pc95) <= 15), msg="DIC 95% Conc <= 15")

        return

    def test_GLODAPv2_Csys(self):
        """
        Test Csys against GLODAP data (n = 83,030).

        Check median offsets are within acceptable limits.
        Check 95% confidence of residuals are within acceptable limits.
        """
        # load GLODAP data
        gd = pd.read_csv("cbsyst/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv")
        gd.dropna(
            subset=[
                "phtsinsitutp",
                "temperature",
                "salinity",
                "tco2",
                "talk",
                "pressure",
                "phosphate",
                "silicate",
            ],
            inplace=True,
        )
        gd.pressure /= 10  # convert pressure to bar

        # exclude weird cruise 270 data
        gd = gd.loc[gd.cruise != 270]

        # calculate pH from TA and DIC
        cpH = Csys(
            TA=gd.talk,
            DIC=gd.tco2,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        pH_resid = gd.phtsinsitutp - cpH.pHtot
        pH_median = np.median(pH_resid)
        pH_pc95 = np.percentile(pH_resid, [2.5, 97.5])

        self.assertLessEqual(abs(pH_median), 0.005, msg="pH Offset <= 0.01")
        self.assertTrue(all(abs(pH_pc95) <= 0.05), msg="pH 95% Conf <= 0.05")

        # calculate TA from pH and DIC
        cTA = Csys(
            pHtot=gd.phtsinsitutp,
            DIC=gd.tco2,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        TA_resid = gd.talk - cTA.TA
        TA_median = np.median(TA_resid)
        TA_pc95 = np.percentile(TA_resid, [2.5, 97.5])

        self.assertLessEqual(abs(TA_median), 0.5, msg="TA Offset <= 2.5")
        self.assertTrue(all(abs(TA_pc95) < 13), msg="TA 95% Conf <= 15")

        # calculate DIC from TA and pH
        cDIC = Csys(
            pHtot=gd.phtsinsitutp,
            TA=gd.talk,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        DIC_resid = gd.tco2 - cDIC.DIC
        DIC_median = np.median(DIC_resid)
        DIC_pc95 = np.percentile(DIC_resid, [2.5, 97.5])

        self.assertLessEqual(abs(DIC_median), 0.5, msg="DIC Offset <= 2")
        self.assertTrue(all(abs(DIC_pc95) < 13), msg="DIC 95% Conf <= 15")

        return

    def test_GLODAPv2_CBsys(self):
        """
        Test Csys against GLODAP data (n = 83,030).

        Check median offsets are within acceptable limits.
        Check 95% confidence of residuals are within acceptable limits.
        """
        if not os.path.exists(
            "cbsyst/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv"
        ):
            get_GLODAP(path="cbsyst/test_data/GLODAP_data/", leave_zip=True)

        # load GLODAP data
        gd = pd.read_csv("cbsyst/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv")
        gd.dropna(
            subset=[
                "phtsinsitutp",
                "temperature",
                "salinity",
                "tco2",
                "talk",
                "pressure",
                "phosphate",
                "silicate",
            ],
            inplace=True,
        )
        gd.pressure /= 10  # convert pressure to bar

        # exclude weird cruise 270 data
        gd = gd.loc[gd.cruise != 270]

        # calculate pH from TA and DIC
        cpH = CBsys(
            TA=gd.talk,
            DIC=gd.tco2,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        pH_resid = gd.phtsinsitutp - cpH.pHtot
        pH_median = np.median(pH_resid)
        pH_pc95 = np.percentile(pH_resid, [2.5, 97.5])

        self.assertLessEqual(abs(pH_median), 0.005, msg="pH Offset <= 0.01")
        self.assertTrue(all(abs(pH_pc95) <= 0.05), msg="pH 95% Conf <= 0.05")

        # calculate TA from pH and DIC
        cTA = CBsys(
            pHtot=gd.phtsinsitutp,
            DIC=gd.tco2,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        TA_resid = gd.talk - cTA.TA
        TA_median = np.median(TA_resid)
        TA_pc95 = np.percentile(TA_resid, [2.5, 97.5])

        self.assertLessEqual(abs(TA_median), 0.5, msg="TA Offset <= 2.5")
        self.assertTrue(all(abs(TA_pc95) < 13), msg="TA 95% Conf <= 15")

        # calculate DIC from TA and pH
        cDIC = CBsys(
            pHtot=gd.phtsinsitutp,
            TA=gd.talk,
            T_in=gd.temperature,
            S_in=gd.salinity,
            P_in=gd.pressure,
            TP=gd.phosphate,
            TSi=gd.silicate,
            BT=415.7,
        )
        DIC_resid = gd.tco2 - cDIC.DIC
        DIC_median = np.median(DIC_resid)
        DIC_pc95 = np.percentile(DIC_resid, [2.5, 97.5])

        self.assertLessEqual(abs(DIC_median), 0.5, msg="DIC Offset <= 2")
        self.assertTrue(all(abs(DIC_pc95) < 13), msg="DIC 95% Conf <= 15")

        return


if __name__ == "__main__":
    unittest.main()
