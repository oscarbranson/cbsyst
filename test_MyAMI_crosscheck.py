import unittest
import numpy as np

from cbsyst.MyAMI_V2 import MyAMI_Fcorr
from cbsyst import MyAMI

class MyAMI_crosscheck(unittest.TestCase):
    """Compare Fcorr factors"""

    def test_Fcorr(self):
        # generate check grid
        n = 5
        T, S, Mg, Ca = np.mgrid[0:40:n, 30:40:n, 0:0.06:n, 0:0.06:n]

        old_Fcorr = MyAMI_Fcorr(XmCa=Ca, XmMg=Mg, TempC=T, Sal=S)
        new_Fcorr = MyAMI.calc_Fcorr(Sal=S, TempC=T, Mg=Mg, Ca=Ca)

        for k in old_Fcorr:
            old = old_Fcorr[k]
            new = new_Fcorr[k]

            maxdiff = np.max(np.abs(old - new))

            self.assertAlmostEqual(0, maxdiff, places=14, msg=f'Maximum difference in {k} correction factor too large: {maxdiff}')

    def test_approx_Fcorr(self):
        n = 5
        TempC, Sal, Mg, Ca = np.mgrid[0:40:n, 30:40:n, 0:0.06:n, 0:0.06:n]
        
        Fcorr_calc = MyAMI.calc_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)
        Fcorr_approx = MyAMI.approximate_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)
        
        for k in Fcorr_calc:
            maxpcdiff = np.max(np.abs(100 * (Fcorr_approx[k] - Fcorr_calc[k]) / Fcorr_calc[k]))
            self.assertLess(maxpcdiff, 0.205, msg=f'Approximate {k} is greater than 0.2% from calculated {k}.')
    
if __name__ == "__main__":
    unittest.main()
