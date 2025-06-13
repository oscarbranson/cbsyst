import unittest
import cbsyst as cb
import numpy as np

from tests import check_vals
ref = check_vals.carbon_ref

def extract_scalar(value):
    """Extract scalar value from array or return scalar directly"""
    if hasattr(value, 'item'):
        return value.item()
    else:
        return value

class TestInputOutput(unittest.TestCase):
    """Test internal consistency of input/output condition calculations"""

    def test_Csys(self):
        
        with self.subTest(msg='Temperature Effect'):      
            c1 = cb.Csys(pHtot=8.1, TA=2300, T_in=20, T_out=30)
            c2 = cb.Csys(pHtot=c1.pHtot, TA=2300, T_in=30, T_out=20)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            
        with self.subTest(msg='Salinity Effect'):      
            c1 = cb.Csys(pHtot=8.1, TA=2300, S_in=28.2, S_out=38.1)
            c2 = cb.Csys(pHtot=c1.pHtot, TA=2300, S_in=38.1, S_out=28.2)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            
        with self.subTest(msg='Pressure Effect'):      
            c1 = cb.Csys(pHtot=8.1, TA=2300, P_in=0, P_out=400)
            c2 = cb.Csys(pHtot=c1.pHtot, TA=2300, P_in=400, P_out=0)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            

    def test_CBsys(self):
        
        with self.subTest(msg='Temperature Effect'):      
            c1 = cb.CBsys(pHtot=8.1, TA=2300, T_in=20, T_out=30, dBT=39.4)
            c2 = cb.CBsys(pHtot=c1.pHtot, TA=2300, T_in=30, T_out=20, dBT=39.4)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            
        with self.subTest(msg='Salinity Effect'):      
            c1 = cb.CBsys(pHtot=8.1, TA=2300, S_in=28.2, S_out=38.1, dBT=39.4)
            c2 = cb.CBsys(pHtot=c1.pHtot, TA=2300, S_in=38.1, S_out=28.2, dBT=39.4)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            
        with self.subTest(msg='Pressure Effect'):      
            c1 = cb.CBsys(pHtot=8.1, TA=2300, P_in=0, P_out=400, dBT=39.4)
            c2 = cb.CBsys(pHtot=c1.pHtot, TA=2300, P_in=400, P_out=0, dBT=39.4)
            self.assertAlmostEqual(extract_scalar(c1.pHtot_in), extract_scalar(c2.pHtot), places=6)
            