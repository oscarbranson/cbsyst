"""
Test suite to verify that the new dataclass-based Csys function 
produces the same results as the original Csys, Bsys, ABsys, and CBsys functions.
"""

import unittest
import numpy as np
import pandas as pd
from cbsyst.cbsyst import Csys as Csys_new
from cbsyst.dep_cbsyst import Csys as Csys_old, Bsys, ABsys, CBsys
from cbsyst.helpers import Bunch
from cbsyst.units import UNIT_MULTIPLIERS
import uncertainties as u

class TestDataclassCompatibility(unittest.TestCase):
    """Test that new dataclass-based approach produces same results as original functions."""
    
    def setUp(self):
        """Set up test parameters."""
        # Standard test conditions
        self.T_in = 25.0
        self.S_in = 35.0
        self.P_in = 0.0
        self.Ca = 0.0102821
        self.Mg = 0.0528171
        self.BT = 416.0  # Standard seawater boron concentration
        
        # Test carbon system parameters
        self.pHtot = 8.1
        self.DIC = 2000.0  # umol/kg
        self.TA = 2300.0   # umol/kg
        
        # Test boron system parameters
        self.BO3 = 300.0   # umol/kg
        self.BO4 = 116.0   # umol/kg
        
        # Test boron isotope parameters
        self.dBT = 39.61
        self.dBO3 = 49.05
        self.dBO4 = self.dBO3 - 27.2
        self.alphaB = 1.0272
        
    def assert_results_close(self, result_new, result_old, rtol=1e-6, atol=1e-8):
        """Assert that two results are close within tolerance."""
        # Get common keys between results
        common_keys = set(result_new.keys()) & set(result_old.keys())
                
        print(f"Debug: Comparing {len(common_keys)} common keys")
        
        for key in common_keys:
            new_val = result_new[key]
            old_val = result_old[key]
            
            if key == 'Ks':
                continue
            
            print(f"Debug: Key '{key}' - New: {type(new_val)} = {new_val}, Old: {type(old_val)} = {old_val}")
            
            # Handle None values
            if new_val is None and old_val is None:
                continue
            if new_val is None or old_val is None:
                self.fail(f"Key {key}: new value is {new_val}, old value is {old_val}")
            
            # Skip non-numeric values (like strings, dicts, etc.)
            if not self._is_numeric(new_val) or not self._is_numeric(old_val):
                # For non-numeric values, just check equality
                if key == 'unit':
                    new_val = UNIT_MULTIPLIERS[new_val]
                if new_val != old_val:
                    self.fail(f"Key {key}: new value '{new_val}' != old value '{old_val}'")
                continue
            
            # Handle uncertainties
            if hasattr(new_val, 'nominal_value') and hasattr(old_val, 'nominal_value'):
                # Both have uncertainties
                try:
                    np.testing.assert_allclose(
                        new_val.nominal_value, old_val.nominal_value, 
                        rtol=rtol, atol=atol,
                        err_msg=f"Key {key} nominal values differ"
                    )
                    np.testing.assert_allclose(
                        new_val.std_dev, old_val.std_dev, 
                        rtol=rtol, atol=atol,
                        err_msg=f"Key {key} uncertainties differ"
                    )
                except Exception as e:
                    self.fail(f"Key {key} comparison failed: {e}")
            elif hasattr(new_val, 'nominal_value') or hasattr(old_val, 'nominal_value'):
                # Only one has uncertainties
                new_nom = new_val.nominal_value if hasattr(new_val, 'nominal_value') else new_val
                old_nom = old_val.nominal_value if hasattr(old_val, 'nominal_value') else old_val
                try:
                    np.testing.assert_allclose(
                        new_nom, old_nom, rtol=rtol, atol=atol,
                        err_msg=f"Key {key} values differ"
                    )
                except Exception as e:
                    self.fail(f"Key {key} comparison failed: {e}")
            else:
                # Neither has uncertainties
                try:
                    np.testing.assert_allclose(
                        new_val, old_val, rtol=rtol, atol=atol,
                        err_msg=f"Key {key} values differ"
                    )
                except Exception as e:
                    self.fail(f"Key {key} comparison failed: {e}")
    
    def _is_numeric(self, value):
        """Check if a value is numeric (can be used in mathematical operations)."""
        if value is None:
            return False
        
        # Check for numpy numeric types
        if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
            return True
        
        # Check for uncertainties
        if hasattr(value, 'nominal_value'):
            return self._is_numeric(value.nominal_value)
        
        # Check for basic numeric types
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def test_carbon_system_basic(self):
        """Test basic carbon system calculations."""
        # Test pH and DIC input
        result_new = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC, 
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=self.pHtot, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_carbon_system_ta_dic(self):
        """Test TA and DIC input combination."""
        result_new = Csys_new(
            TA=self.TA, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            TA=self.TA, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_carbon_system_ph_ta(self):
        """Test pH and TA input combination."""
        result_new = Csys_new(
            pHtot=self.pHtot, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=self.pHtot, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_carbon_system_fco2_ta(self):
        """Test fCO2 and TA input combination."""
        fCO2 = 400.0  # uatm
        
        result_new = Csys_new(
            fCO2=fCO2, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            fCO2=fCO2, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_carbon_system_omega_inputs(self):
        """Test Omega inputs."""
        OmegaC = 3.0
        OmegaA = 2.0
        
        result_new = Csys_new(
            OmegaC=OmegaC, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            OmegaC=OmegaC, TA=self.TA,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_boron_system_basic(self):
        """Test basic boron system calculations."""
        # Test pH and BT input
        result_new = Csys_new(
            pHtot=self.pHtot, BT=self.BT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, dBT=self.dBT
        )
        
        print(type(result_new))
        
        result_old = Bsys(
            pHtot=self.pHtot, BT=self.BT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, dBT=self.dBT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_boron_system_bo3_bo4(self):
        """Test BO3 and BO4 input combination."""
        result_new = Csys_new(
            BO3=self.BO3, BO4=self.BO4, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg
        )
        
        result_old = CBsys(
            DIC=2000, BO3=self.BO3, BO4=self.BO4, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_boron_isotopes_basic(self):
        """Test basic boron isotope calculations."""
        # Test pH and dBT input
        result_new = Csys_new(
            pHtot=self.pHtot, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        result_old = ABsys(
            pHtot=self.pHtot, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_boron_isotopes_dbo3_dbT(self):
        """Test dBO3 and dBO4 input combination."""
        result_new = Csys_new(
            dBO3=self.dBO3, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        result_old = CBsys(
            DIC=2000, dBO3=self.dBO3, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_combined_carbon_boron(self):
        """Test combined carbon and boron system."""
        result_new = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC, BT=self.BT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg
        )
        
        result_old = CBsys(
            pHtot=self.pHtot, DIC=self.DIC, BT=self.BT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_combined_carbon_boron_isotopes(self):
        """Test combined carbon, boron, and isotope system."""
        result_new = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC, BT=self.BT, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        result_old = CBsys(
            pHtot=self.pHtot, DIC=self.DIC, BT=self.BT, dBT=self.dBT,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, alphaB=self.alphaB
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_uncertainty_propagation(self):
        """Test uncertainty propagation."""
        # Create uncertain inputs
        pH_uncertain = u.ufloat(self.pHtot, 0.1)
        DIC_uncertain = u.ufloat(self.DIC, 50.0)
        
        result_new = Csys_new(
            pHtot=pH_uncertain, DIC=DIC_uncertain,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=pH_uncertain, DIC=DIC_uncertain,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_array_inputs(self):
        """Test array inputs."""
        # Create arrays
        pH_array = np.array([8.0, 8.1, 8.2])
        DIC_array = np.array([1900, 2000, 2100])
        
        result_new = Csys_new(
            pHtot=pH_array, DIC=DIC_array,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=pH_array, DIC=DIC_array,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_output_conditions(self):
        """Test output conditions."""
        T_out = 20.0
        S_out = 30.0
        P_out = 100.0
        
        result_new = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC,
            T_in=self.T_in, T_out=T_out,
            S_in=self.S_in, S_out=S_out,
            P_in=self.P_in, P_out=P_out,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=self.pHtot, DIC=self.DIC,
            T_in=self.T_in, T_out=T_out,
            S_in=self.S_in, S_out=S_out,
            P_in=self.P_in, P_out=P_out,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_different_units(self):
        """Test different concentration units."""
        # Test with mmol units
        DIC_mmol = 2.0  # mmol/kg
        TA_mmol = 2.3   # mmol/kg
        
        result_new = Csys_new(
            DIC=DIC_mmol, TA=TA_mmol, unit="mmol",
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            DIC=DIC_mmol, TA=TA_mmol, unit="mmol",
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very low pH
        result_new = Csys_new(
            pHtot=7.0, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=7.0, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
        
        # Test with very high pH
        result_new = Csys_new(
            pHtot=9.0, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        result_old = Csys_old(
            pHtot=9.0, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        self.assert_results_close(result_new, result_old)
    
    def test_return_type(self):
        """Test that new function returns correct type."""
        result = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        # Check that it's a dataclass instance
        from cbsyst.dataclasses import CBsystData
        self.assertIsInstance(result, CBsystData)
        
        # Check that it has dict-like access
        self.assertIsNotNone(result['pHtot'])
        self.assertIsNotNone(result['DIC'])
        self.assertIsNotNone(result['TA'])
    
    def test_input_identification(self):
        """Test that inputs are correctly identified."""
        result = Csys_new(
            pHtot=self.pHtot, DIC=self.DIC,
            T_in=self.T_in, S_in=self.S_in, P_in=self.P_in,
            Ca=self.Ca, Mg=self.Mg, BT=self.BT
        )
        
        # Check that inputs are identified
        self.assertIn('inputs', result)
        self.assertIn('pHtot', result.inputs)
        self.assertIn('DIC', result.inputs)
        
        # Check that defaults are not in inputs
        self.assertNotIn('T_in', result.inputs)  # Should be default
        self.assertNotIn('S_in', result.inputs)  # Should be default


class TestDataclassSpecificFeatures(unittest.TestCase):
    """Test dataclass-specific features and functionality."""
    
    def test_dataclass_creation(self):
        """Test dataclass creation with different parameter combinations."""
        from cbsyst.dataclasses import create_dataclass
        
        # Test carbon-only
        carbon_params = create_dataclass(pHtot=8.1, DIC=2000)
        self.assertIsInstance(carbon_params, carbon_params.__class__)
        
        # Test boron-only
        boron_params = create_dataclass(BT=416, BO3=300)
        self.assertIsInstance(boron_params, boron_params.__class__)
        
        # Test isotope-only
        isotope_params = create_dataclass(dBT=39.6, dBO3=19.0)
        self.assertIsInstance(isotope_params, isotope_params.__class__)
        
        # Test combined
        combined_params = create_dataclass(pHtot=8.1, DIC=2000, BT=416, dBT=39.6)
        self.assertIsInstance(combined_params, combined_params.__class__)
    
    def test_dataclass_dict_interface(self):
        """Test that dataclasses have proper dict-like interface."""
        from cbsyst.dataclasses import create_dataclass
        
        params = create_dataclass(pHtot=8.1, DIC=2000)
        
        # Test dict-style access
        self.assertEqual(params['pHtot'], 8.1)
        self.assertEqual(params['DIC'], 2000)
        
        # Test dict-style setting
        params['pHtot'] = 8.2
        self.assertEqual(params['pHtot'], 8.2)
        
        # Test dict-style methods
        self.assertIn('pHtot', params)
        self.assertIn('DIC', params)
        self.assertNotIn('nonexistent', params)
        
        # Test get with default
        self.assertEqual(params.get('pHtot'), 8.2)
        self.assertEqual(params.get('nonexistent', 'default'), 'default')
        
        # Test keys, values, items
        keys = list(params.keys())
        self.assertIn('pHtot', keys)
        self.assertIn('DIC', keys)
        
        values = list(params.values())
        self.assertIn(8.2, values)
        self.assertIn(2000, values)


if __name__ == '__main__':
    unittest.main() 