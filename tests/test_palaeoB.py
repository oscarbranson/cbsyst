import unittest
import os
import pandas as pd
import numpy as np
import cbsyst.carbon as cf
import cbsyst.boron as bf
import cbsyst.boron_isotopes as bif
from cbsyst.cbsyst import Bsys, CBsys, ABsys

np.random.seed(42)

n = 101
pHtot = np.random.uniform(7.6, 8.8, n)
DIC = np.random.uniform(1900, 2200, n)
BT = np.random.uniform(350, 450, n)
dBT = np.random.uniform(35, 45, n)
T = np.random.uniform(15, 35, n)
S = np.random.uniform(30, 40, n)

test = CBsys(pHtot=pHtot, dBT=dBT, BT=BT, DIC=DIC, T_in=T, S_in=S)


class ReferenceDataTestCase(unittest.TestCase):
    """Test boron idotopes"""

    def test_Bisotopes(self):
        check_A = ABsys(dBO4=test.dBO4, dBT=test.dBT, T_in=T, S_in=S)
        
        self.assertIsNone(np.testing.assert_allclose(test.pHtot, check_A.pHtot, rtol=1e-10))
        
        check_B = Bsys(dBO4=test.dBO4, dBT=test.dBT, BT=test.BT, T_in=T, S_in=S)
    
        self.assertIsNone(np.testing.assert_allclose(test.pHtot, check_B.pHtot, rtol=1e-10))

        check_CB = CBsys(dBO4=test.dBO4, dBT=test.dBT, DIC=test.DIC, BT=test.BT, T_in=T, S_in=S)

        self.assertIsNone(np.testing.assert_allclose(test.pHtot, check_CB.pHtot, rtol=1e-10))
