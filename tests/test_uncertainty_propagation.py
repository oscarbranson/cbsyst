"""
Test uncertainty propagation through cbsyst calculations.
"""

import unittest
import numpy as np
import uncertainties
from uncertainties import ufloat
import cbsyst as cb
from cbsyst.helpers import Bunch


class TestUncertaintyPropagation(unittest.TestCase):
    """Test uncertainty propagation for carbon system calculations."""
    
    def setUp(self):
        """Set up test parameters with uncertainties."""
        # Standard seawater conditions
        self.T = 25.0  # Temperature in °C
        self.S = 35.0  # Salinity
        self.P = 0.0   # Pressure in dbar
        
        # Parameters with uncertainties
        self.DIC_uncertain = ufloat(2100, 10)  # μmol/kg, ±10
        self.TA_uncertain = ufloat(2300, 15)   # μmol/kg, ±15
        self.pH_uncertain = ufloat(8.1, 0.02)  # total scale, ±0.02
        self.CO2_uncertain = ufloat(15, 1)     # μmol/kg, ±1
        self.HCO3_uncertain = ufloat(1800, 20) # μmol/kg, ±20
        self.CO3_uncertain = ufloat(250, 10)   # μmol/kg, ±10
        
        # Ancillary parameters (no uncertainties for simplicity)
        self.BT = 416.0
        self.PT = 2.3
        self.SiT = 50.0
        self.ST = 28900.0
        self.FT = 68.0
    
    def test_algebraic_functions_with_uncertainties(self):
        """Test algebraic functions (cases 1, 6-9) with uncertainties."""
        # Get Ks object by creating a simple system
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Case 1: CO2 and pH -> DIC
        result = cb.carbon.CO2_pH(self.CO2_uncertain, self.pH_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_pH result: {result}")
        
        # Case 6: pH and HCO3 -> DIC  
        result = cb.carbon.pH_HCO3(self.pH_uncertain, self.HCO3_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"pH_HCO3 result: {result}")
    
    def test_zero_finder_functions_with_uncertainties(self):
        """Test zero-finder functions (cases 2, 3, 5, 10, 12, 14) with uncertainties."""
        # Get Ks object by creating a simple system
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Case 2: CO2 and HCO3 -> H
        result = cb.carbon.CO2_HCO3(self.CO2_uncertain, self.HCO3_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_HCO3 result: {result}")
        
        # Case 3: CO2 and CO3 -> H
        result = cb.carbon.CO2_CO3(self.CO2_uncertain, self.CO3_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_CO3 result: {result}")
        
        # Case 5: CO2 and DIC -> H
        result = cb.carbon.CO2_DIC(self.CO2_uncertain, self.DIC_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_DIC result: {result}")
        
        # Case 10: HCO3 and CO3 -> H
        result = cb.carbon.HCO3_CO3(self.HCO3_uncertain, self.CO3_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"HCO3_CO3 result: {result}")
        
        # Case 12: HCO3 and DIC -> H
        result = cb.carbon.HCO3_DIC(self.HCO3_uncertain, self.DIC_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"HCO3_DIC result: {result}")
        
        # Case 14: CO3 and DIC -> H
        result = cb.carbon.CO3_DIC(self.CO3_uncertain, self.DIC_uncertain, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO3_DIC result: {result}")
    
    def test_iterative_functions_with_uncertainties(self):
        """Test iterative functions (cases 4, 11, 13, 15) with uncertainties using decorator."""
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Case 4: CO2 and TA -> pH
        result = cb.carbon.CO2_TA(self.CO2_uncertain, self.TA_uncertain, 
                                self.BT, self.PT, self.SiT, self.ST, self.FT, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_TA result: {result}")
        
        # Case 11: HCO3 and TA -> H
        result = cb.carbon.HCO3_TA(self.HCO3_uncertain, self.TA_uncertain, self.BT, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"HCO3_TA result: {result}")
        
        # Case 13: CO3 and TA -> H
        result = cb.carbon.CO3_TA(self.CO3_uncertain, self.TA_uncertain, self.BT, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO3_TA result: {result}")
        
        # Case 15: TA and DIC -> pH
        result = cb.carbon.TA_DIC(self.TA_uncertain, self.DIC_uncertain, 
                                self.BT, self.PT, self.SiT, self.ST, self.FT, Ks)
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"TA_DIC result: {result}")
    
    def test_no_uncertainties_preserved(self):
        """Test that functions work normally when no uncertainties are present."""
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Test that functions return normal floats when no uncertainties
        result = cb.carbon.CO2_TA(15.0, 2300.0, self.BT, self.PT, self.SiT, self.ST, self.FT, Ks)
        self.assertIsInstance(result, (float, np.floating))
        print(f"CO2_TA without uncertainties: {result}")
        
        result = cb.carbon.CO2_HCO3(15.0, 1800.0, Ks)
        self.assertIsInstance(result, (float, np.floating))
        print(f"CO2_HCO3 without uncertainties: {result}")
    
    def test_mixed_uncertainties(self):
        """Test functions with mix of uncertain and certain parameters."""
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Mix of uncertain and certain parameters
        result = cb.carbon.CO2_HCO3(self.CO2_uncertain, 1800.0, Ks)  # Only CO2 has uncertainty
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_HCO3 with mixed uncertainties: {result}")
        
        result = cb.carbon.CO2_TA(15.0, self.TA_uncertain, 
                                self.BT, self.PT, self.SiT, self.ST, self.FT, Ks)  # Only TA has uncertainty
        self.assertIsInstance(result, uncertainties.UFloat)
        self.assertGreater(result.std_dev, 0)
        print(f"CO2_TA with mixed uncertainties: {result}")
    
    def test_array_inputs_with_uncertainties(self):
        """Test functions with array inputs containing uncertainties."""
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Create arrays with uncertainties
        CO2_array = np.array([ufloat(10, 0.5), ufloat(15, 1.0), ufloat(20, 1.5)])
        HCO3_array = np.array([ufloat(1700, 15), ufloat(1800, 20), ufloat(1900, 25)])
        
        # Test with array inputs
        result = cb.carbon.CO2_HCO3(CO2_array, HCO3_array, Ks)
        self.assertTrue(hasattr(result, '__len__'))  # Should be array-like
        for r in result:
            if hasattr(r, 'std_dev'):
                self.assertGreater(r.std_dev, 0)
        print(f"CO2_HCO3 with array uncertainties: {result}")
    
    def test_uncertainty_magnitude_reasonable(self):
        """Test that propagated uncertainties have reasonable magnitudes."""
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Test with small input uncertainties
        small_CO2 = ufloat(15, 0.1)  # Small uncertainty
        small_HCO3 = ufloat(1800, 1)  # Small uncertainty
        
        result = cb.carbon.CO2_HCO3(small_CO2, small_HCO3, Ks)
        # Output uncertainty should be small but non-zero
        self.assertGreater(result.std_dev, 0)
        self.assertLess(result.std_dev / abs(result.nominal_value), 0.1)  # Less than 10% relative error
        print(f"Small uncertainty test: {result}")
        
        # Test with large input uncertainties
        large_CO2 = ufloat(15, 5)  # Large uncertainty
        large_HCO3 = ufloat(1800, 200)  # Large uncertainty
        
        result = cb.carbon.CO2_HCO3(large_CO2, large_HCO3, Ks)
        # Output uncertainty should be larger
        self.assertGreater(result.std_dev, 0)
        print(f"Large uncertainty test: {result}")
    
    def test_uarray_support(self):
        """Test functions with uarray inputs (correlated uncertainties)."""
        import uncertainties.unumpy as unp
        
        Ks = Bunch(cb.calc_Ks(temp_c=self.T, sal=self.S, p_bar=self.P))
        
        # Create uarray objects
        CO2_uarray = unp.uarray([10.0, 15.0, 20.0], [0.5, 1.0, 1.5])
        HCO3_uarray = unp.uarray([1700.0, 1800.0, 1900.0], [15.0, 20.0, 25.0])
        pH_uarray = unp.uarray([8.0, 8.1, 8.2], [0.01, 0.02, 0.03])
        TA_uarray = unp.uarray([2250.0, 2300.0, 2350.0], [10.0, 15.0, 20.0])
        
        # Test algebraic function with uarray
        result = cb.carbon.CO2_pH(CO2_uarray, pH_uarray, Ks)
        self.assertTrue(hasattr(result, '__len__'))  # Should be array-like
        for r in result:
            self.assertTrue(hasattr(r, 'std_dev'))
            self.assertGreater(r.std_dev, 0)
        print(f"CO2_pH with uarray: {result}")
        
        # Test zero-finder function with uarray
        result = cb.carbon.CO2_HCO3(CO2_uarray, HCO3_uarray, Ks)
        self.assertTrue(hasattr(result, '__len__'))  # Should be array-like
        for r in result:
            self.assertTrue(hasattr(r, 'std_dev'))
            self.assertGreater(r.std_dev, 0)
        print(f"CO2_HCO3 with uarray: {result}")
        
        # Test iterative function with uarray
        result = cb.carbon.CO2_TA(CO2_uarray, TA_uarray, 
                                self.BT, self.PT, self.SiT, self.ST, self.FT, Ks)
        self.assertTrue(hasattr(result, '__len__'))  # Should be array-like
        for r in result:
            self.assertTrue(hasattr(r, 'std_dev'))
            self.assertGreater(r.std_dev, 0)
        print(f"CO2_TA with uarray: {result}")


if __name__ == '__main__':
    unittest.main()
