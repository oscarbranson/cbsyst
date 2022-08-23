import unittest
from .check_vals import boron_ref as ref
from cbsyst import boron_isotopes as bif

class BoronIsotopeFunctions(unittest.TestCase):
    """Test B isotope functions"""

    def test_boron_isotopes(self):
        
        with self.subTest(msg='get_alphaB'):
            self.assertEqual(bif.get_alphaB(), ref.alphaB)

        with self.subTest(msg='calculate_ABT (H, BO3)'):
            self.assertAlmostEqual(
                bif.calculate_ABT(H=ref.H, ABO3=ref.ABO3, Ks=ref.Ks, alphaB=ref.alphaB),
                ref.ABT,
                places=6,
            )

        with self.subTest(msg='calculate_ABT (H, BO4)'):
            self.assertAlmostEqual(
                bif.calculate_ABT(H=ref.H, ABO4=ref.ABO4, Ks=ref.Ks, alphaB=ref.alphaB),
                ref.ABT,
                places=6,
            )

        with self.subTest(msg='calculate_ABO3'):
            self.assertAlmostEqual(
                bif.calculate_ABO3(H=ref.H, ABT=ref.ABT, Ks=ref.Ks, alphaB=ref.alphaB), ref.ABO3, places=6
            )

        with self.subTest(msg='calculate_ABO4'):
            self.assertAlmostEqual(
                bif.calculate_ABO4(H=ref.H, ABT=ref.ABT, Ks=ref.Ks, alphaB=ref.alphaB), ref.ABO4, places=6
            )

        # Isotope unit conversions
        with self.subTest(msg='A11_to_d11'):
            self.assertAlmostEqual(
                bif.A11_to_d11(0.807817779214075), 39.5, places=6
            )

        with self.subTest(msg='d11_to_A11'):
            self.assertAlmostEqual(
                bif.d11_to_A11(39.5), 0.807817779214075, places=6
            )
    