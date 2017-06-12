# cbsyst

**A Python module for calculating seawater carbon and B chemistry**

*Work in progress! Results not guaranteed. Use at your own risk.*

Constants calculated by an adaptation of [Mathis Hain's MyAMI model](http://www.mathis-hain.net/resources/Hain_et_al_2015_GBC.pdf).

## Installation

**Developed and tested in Python 3.5**. 
Should work in 2.7+, but can't guarantee.

```bash
pip install git+https://github.com/oscarbranson/cbsyst.git@master
```

## Example Usage

```python
import cbsyst as cb
import numpy as np

# Create pH master variable for demo
pH = np.linspace(7,11,100)

# Example Usage
# -------------
# The following functions can be used to calculate the
# speciation of C and B in seawater, and the isotope
# fractionation of B, given minimal input parameters.
#
# See the docstring for each function for info on
# required minimal parameters.

# Carbon system only
Csw = cb.Csys(pH=pH, DIC=2000.)

# Boron system only
Bsw = cb.Bsys(pH=pH, BT=433., dBT=39.5)

# Carbon and Boron systems
CBsw = cb.CBsys(pH=pH, DIC=2000., BT=433., dBT=39.5)

# NOTE:
# At present, each function call can only be used to
# calculate a single minimal-parameter combination -
# i.e. you can't pass it multiple arrays of parameters
# with different combinations of parameters, as in
# the Matlab CO2SYS code.

# Example Output
# --------------
# The functions return a Bunch (modified dict with '.' 
# attribute access) containing all system parameters
# and constants.
#
# Output for a single input condition shown for clarity:

out = cb.CBsys(pH=8.1, DIC=2000., BT=433., dBT=39.5)
out

>>> {'ABO3': array([ 0.80882931]),
     'ABO4': array([ 0.80463763]),
     'ABT': array([ 0.80781778]),
     'BO3': array([ 328.50895695]),
     'BO4': array([ 104.49104305]),
     'BT': array([ 433.]),
     'CO2': array([ 9.7861814]),
     'CO3': array([ 238.511253]),
     'Ca': array([ 0.0102821]),
     'DIC': array([ 2000.]),
     'H': array([  7.94328235e-09]),
     'HCO3': array([ 1751.7025656]),
     'Ks': {'K0': array([ 0.02839188]),
      'K1': array([  1.42182814e-06]),
      'K2': array([  1.08155475e-09]),
      'KB': array([  2.52657299e-09]),
      'KSO4': array([ 0.10030207]),
      'KW': array([  6.06386369e-14]),
      'KspA': array([  6.48175907e-07]),
      'KspC': array([  4.27235093e-07])},
     'Mg': array([ 0.0528171]),
     'S': array([ 35.]),
     'T': array([ 25.]),
     'TA': array([ 2333.21612227]),
     'alphaB': array([ 1.02725]),
     'dBO3': array([ 46.30877684]),
     'dBO4': array([ 18.55320208]),
     'dBT': array([ 39.5]),
     'deltas': True,
     'fCO2': array([ 344.68238018]),
     'pCO2': array([ 345.78871573]),
     'pH': array([ 8.1]),
     'pdict': None}
```