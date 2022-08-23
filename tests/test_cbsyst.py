import unittest
import os
import pandas as pd
import numpy as np
import cbsyst.carbon as cf
import cbsyst.boron as bf
import cbsyst.boron_isotopes as bif
from cbsyst.cbsyst import Csys, CBsys
from cbsyst.helpers import Bunch
from tests.test_data.GLODAP_data.get_GLODAP_data import get_GLODAP


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
            "tests/test_data/Lueker2000/Lueker2000_Table3.csv", comment="#"
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
            "tests/test_data/Lueker2000/Lueker2000_Table3.csv", comment="#"
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
        gd = pd.read_csv("tests/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv")
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

        # set negative nutrient values to zero
        gd.phosphate[gd.phosphate < 0] = 0
        gd.silicate[gd.silicate < 0] = 0
        
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

        self.assertLessEqual(abs(pH_median), 0.005, msg="pH Offset <= 0.005")
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

        self.assertLessEqual(abs(TA_median), 0.5, msg="TA Offset <= 0.5")
        self.assertTrue(all(abs(TA_pc95) < 13), msg="TA 95% Conf <= 13")

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

        self.assertLessEqual(abs(DIC_median), 0.5, msg="DIC Offset <= 0.5")
        self.assertTrue(all(abs(DIC_pc95) < 13), msg="DIC 95% Conf <= 13")

        return

    def test_GLODAPv2_CBsys(self):
        """
        Test Csys against GLODAP data (n = 83,030).

        Check median offsets are within acceptable limits.
        Check 95% confidence of residuals are within acceptable limits.
        """
        if not os.path.exists(
            "tests/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv"
        ):
            get_GLODAP(path="tests/test_data/GLODAP_data/", leave_zip=True)

        # load GLODAP data
        gd = pd.read_csv("tests/test_data/GLODAP_data/GLODAPv2_pH_DIC_ALK_subset.csv")
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
        
        # set negative nutrient values to zero
        gd.phosphate[gd.phosphate < 0] = 0
        gd.silicate[gd.silicate < 0] = 0
        
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

        self.assertLessEqual(abs(pH_median), 0.005, msg="pH Offset <= 0.005")
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

        self.assertLessEqual(abs(TA_median), 1, msg="TA Offset <= 1")
        self.assertTrue(all(abs(TA_pc95) < 15), msg="TA 95% Conf <= 15")

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

        self.assertLessEqual(abs(DIC_median), 1, msg="DIC Offset <= 1")
        self.assertTrue(all(abs(DIC_pc95) < 15), msg="DIC 95% Conf <= 15")
