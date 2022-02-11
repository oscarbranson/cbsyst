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

        for k in old_Fcorr.keys():
            old = old_Fcorr[k]
            new = new_Fcorr[k]

            maxdiff = np.max(np.abs(old - new))

            self.assertAlmostEqual(0, maxdiff, places=15, msg=f'Maximum difference in {k} correction factor too large: {maxdiff}')

    
if __name__ == "__main__":
    unittest.main()
