import unittest
from .check_vals import boron_ref as ref
from cbsyst import boron as bf

class BoronFunctions(unittest.TestCase):
    """Test B concentration and speciation functions"""

    def test_boron(self):

        # Speciation
        with self.subTest(msg='BT_BO3'):
            self.assertAlmostEqual(
                bf.BT_BO3(BT=ref.BT, BO3=ref.BO3, Ks=ref.Ks), ref.H, places=6
            )

        with self.subTest(msg='BT_BO4'):
            self.assertAlmostEqual(
                bf.BT_BO4(BT=ref.BT, BO4=ref.BO4, Ks=ref.Ks), ref.H, places=6
            )

        with self.subTest(msg='pH_BO3'):
            self.assertAlmostEqual(
                bf.pH_BO3(ref.pHtot, BO3=ref.BO3, Ks=ref.Ks), ref.BT, places=6
            )

        with self.subTest(msg='pH_BO3'):
            self.assertAlmostEqual(
                bf.pH_BO4(ref.pHtot, BO4=ref.BO4, Ks=ref.Ks), ref.BT, places=6
            )

        with self.subTest(msg='cBO3'):
            self.assertAlmostEqual(
                bf.cBO3(BT=ref.BT, H=ref.H, Ks=ref.Ks), ref.BO3, places=6
            )

        with self.subTest(msg='cBO4'):
            self.assertAlmostEqual(
                bf.cBO4(BT=ref.BT, H=ref.H, Ks=ref.Ks), ref.BO4, places=6
            )

        with self.subTest(msg='chiB_calc'):
            self.assertEqual(
                bf.chiB_calc(ref.H, Ks=ref.Ks), 1 / (1 + ref.Ks.KB / ref.H)
                )
