# cbsyst

**A Python module for calculating seawater carbon and B chemistry**

*Work in progress! Results not guaranteed. Use at your own risk.*

Constants calculated from an adaptation of [Mathis Hain's MyAMI model](http://www.mathis-hain.net/resources/Hain_et_al_2015_GBC.pdf).

## Installation

**Developed and tested in Python 3.5**. Should work in 2.7+, but can't guarantee.

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
#
# The functions return a Bunch (modified dict with '.' 
# attribute access) containing all system parameters
# and constants.

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
```