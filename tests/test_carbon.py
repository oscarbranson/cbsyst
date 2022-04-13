import unittest
from .check_vals import carbon_ref as ref
from cbsyst import carbon as cf

class CarbonFnTestCase(unittest.TestCase):
    """Test all C functions"""

    def test_Carbon(self):
        
        with self.subTest(msg='CO2_pH'):
            self.assertAlmostEqual(
                cf.CO2_pH(CO2=ref.CO2, pH=ref.pHtot, Ks=ref.Ks), ref.DIC, places=6
            )

        with self.subTest(msg='CO2_HCO3 (zf)'):
            self.assertAlmostEqual(
                cf.CO2_HCO3(CO2=ref.CO2, HCO3=ref.HCO3, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='CO2_CO3 (zf)'):
            self.assertAlmostEqual(
                cf.CO2_CO3(CO2=ref.CO2, CO3=ref.CO3, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='CO2_TA'):
            self.assertAlmostEqual(
                cf.CO2_TA(
                    CO2=ref.CO2 / ref.unit,
                    TA=ref.TA / ref.unit,
                    BT=ref.BT / ref.unit,
                    TP=ref.TP / ref.unit,
                    TSi=ref.TSi / ref.unit,
                    TS=ref.TS,
                    TF=ref.TF,
                    Ks=ref.Ks,
                )[0],
                ref.pHtot,
                places=6,
            )
        
        with self.subTest(msg='CO2_DIC (zf)'):        
            self.assertAlmostEqual(
                cf.CO2_DIC(CO2=ref.CO2, DIC=ref.DIC, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='pH_HCO3'):        
            self.assertAlmostEqual(
                cf.pH_HCO3(pH=ref.pHtot, HCO3=ref.HCO3, Ks=ref.Ks), ref.DIC, places=6
            )

        with self.subTest(msg='pH_CO3'):        
            self.assertAlmostEqual(
                cf.pH_CO3(pH=ref.pHtot, CO3=ref.CO3, Ks=ref.Ks), ref.DIC, places=6
            )

        with self.subTest(msg='pH_TA'):        
            self.assertAlmostEqual(
                cf.pH_TA(
                    pH=ref.pHtot,
                    TA=ref.TA / ref.unit,
                    BT=ref.BT / ref.unit,
                    TP=ref.TP / ref.unit,
                    TSi=ref.TSi / ref.unit,
                    TS=ref.TS,
                    TF=ref.TF,
                    Ks=ref.Ks,
                )
                * ref.unit,
                ref.DIC,
                places=6,
            )

        with self.subTest(msg='pH_DIC'):        
            self.assertAlmostEqual(
                cf.pH_DIC(pH=ref.pHtot, DIC=ref.DIC, Ks=ref.Ks), ref.CO2, places=6
            )

        with self.subTest(msg='HCO3_CO3 (zf)'):        
            self.assertAlmostEqual(
                cf.HCO3_CO3(HCO3=ref.HCO3, CO3=ref.CO3, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='HCO3_TA (zf)'):        
            self.assertAlmostEqual(
                cf.HCO3_TA(HCO3=ref.HCO3 / ref.unit, TA=ref.TA / ref.unit, BT=ref.BT / ref.unit, Ks=ref.Ks)[
                    0
                ],
                ref.H,
                places=6,
            )

        with self.subTest(msg='HCO3_DIC (zf)'):        
            self.assertAlmostEqual(
                cf.HCO3_DIC(HCO3=ref.HCO3, DIC=ref.DIC, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='CO3_TA (zf)'):        
            self.assertAlmostEqual(
                cf.CO3_TA(CO3=ref.CO3 / ref.unit, TA=ref.TA / ref.unit, BT=ref.BT / ref.unit, Ks=ref.Ks)[0],
                ref.H,
                places=6,
            )

        with self.subTest(msg='CO3_DIC (zf)'):        
            self.assertAlmostEqual(
                cf.CO3_DIC(CO3=ref.CO3, DIC=ref.DIC, Ks=ref.Ks)[0], ref.H, places=6
            )

        with self.subTest(msg='TA_DIC'):        
            self.assertAlmostEqual(
                cf.TA_DIC(
                    TA=ref.TA / ref.unit,
                    DIC=ref.DIC / ref.unit,
                    BT=ref.BT / ref.unit,
                    TP=ref.TP / ref.unit,
                    TSi=ref.TSi / ref.unit,
                    TS=ref.TS,
                    TF=ref.TF,
                    Ks=ref.Ks,
                )[0],
                ref.pHtot,
                places=6,
            )

        with self.subTest(msg='cCO2'):        
            self.assertAlmostEqual(
                cf.cCO2(H=ref.H, DIC=ref.DIC, Ks=ref.Ks), ref.CO2, places=6
            )

        with self.subTest(msg='cCO3'):        
            self.assertAlmostEqual(
                cf.cCO3(H=ref.H, DIC=ref.DIC, Ks=ref.Ks), ref.CO3, places=6
            )

        with self.subTest(msg='cHCO3'):        
            self.assertAlmostEqual(
                cf.cHCO3(H=ref.H, DIC=ref.DIC, Ks=ref.Ks), ref.HCO3, places=6
            )

        with self.subTest(msg='cTA'):        
            (TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF) = cf.cTA(
                H=ref.H,
                DIC=ref.DIC / ref.unit,
                BT=ref.BT / ref.unit,
                TP=ref.TP / ref.unit,
                TSi=ref.TSi / ref.unit,
                TS=ref.TS,
                TF=ref.TF,
                Ks=ref.Ks,
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

        with self.subTest(msg='fCO2_to_CO2'):        
            self.assertAlmostEqual(
                cf.fCO2_to_CO2(fCO2=ref.fCO2, Ks=ref.Ks), ref.CO2, places=6
            )

        with self.subTest(msg='CO2_to_fCO2'):        
            self.assertAlmostEqual(
                cf.CO2_to_fCO2(CO2=ref.CO2, Ks=ref.Ks), ref.fCO2, places=6
            )

        with self.subTest(msg='fCO2_to_pCO2'):        
            self.assertAlmostEqual(
                cf.fCO2_to_pCO2(fCO2=ref.fCO2, Tc=ref.T), ref.pCO2, places=6
            )

        with self.subTest(msg='pCO2_to_fCO2'):        
            self.assertAlmostEqual(
                cf.pCO2_to_fCO2(pCO2=ref.pCO2, Tc=ref.T), ref.fCO2, places=6
            )

