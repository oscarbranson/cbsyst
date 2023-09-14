.. :changelog:

Release History
---------------

0.4.9 (2023-09-04)
------------
Updates to work with Kgen 0.3.0

Main code changes:
* Explicitly set default Ca (0.0102821) and Mg (0.0528171) concentrations so pymyami is not run at ambient conditions.
* Explicitly set P_in to 0.0 so pressure calculation is not run at surface conditions.
* Set minumum required Kgen version to 0.3.0.

0.4.8 (2023-03-15)
------------
Set minumum required Kgen version to 0.2.0.

0.4.7 (2023-03-14)
------------
Updates to work with Kgen 0.2.0.

**Minor breaking change**: All TX quantities have been renamed to XT (i.e. TF and TS are not FT and ST to be consistent with BT nomenclature)

0.4.6 (2023-03)
------------
Fix Omega Units

0.4.4 (2023-03)
------------
Fix OmegaC

0.4.3 (2023-02-07)
------------
Omega calculation.

Main code changes:
* Added calculation of OmegaA and OmegaC.
* Corrected calculation of conservative ions when S_out is specified.
* Modified test_in_out.py to accommodate S_out handling.

0.4.1 (2022-08-23)
------------
Technical: Stopped including all GLODAP data in bdist_wheel to reduce file size.

0.4.0 (2022-08-23)
------------
Delegate all K calculation to external packages:
* [kgen](https://github.com/PalaeoCarb/Kgen) for K calculation.
* [pymyami](https://github.com/PalaeoCarb/MyAMI) for adjusting Ks for seawater major ion composition.

Main code changes:
* Strip out all old MyAMI code
* re-organise carbon, boron, and boron isotope functions
* Added in additional B isotope functions for palaeo-calculations
* Functionality for providing B isotopes instead of pH for the main functions.

0.3.7 (2021-04-23)
------------------
(including changes from 0.3.6... shoddy record keeping)

* Fixed MyAMI_V2 to match original Matlab version (typo in temperature parameter)
* Revelle factor calculation.
* pH scale conversion calculator
* Allow specification of TS and TF
* Test updates to reflect changes
* general bug fixes
* Makefile for testing and distribution.
* Logo!

Thanks to @douglascoenen for typo correction.


0.3.5 (2016-06-23)
------------------

* Bring CBsys in line with new changes.


0.3.4 (2016-06-23)
------------------

* CO2SYS comparison with GLODAPv2 Bottle data.
* Minor parameter fixes and equation updates to match CO2SYS.
* pH scale correction of KP1, KP2, KP3, KSi and KW


0.3.3 (2016-06-21)
------------------

* Implemented nutrient alkalinity for all except CO3_TA and HCO3_TA cases using parameterisation of Matlab CO2SYS.


0.3.2 (2017-06-14)
------------------

* Moved pressure correction back into MyAMI_V2 functions, but still after parameter calculation so speed increase is maintained.
* Added data_out function for exporting data.
* Minor idiot-proofing of minor functions.
* Improved tests.
* General cleanup.


0.3.1 (2017-06-13)
------------------

**Moved pressure correction from MyAMI to cbsyst.**
As the pressure correction factor is multiplicative, it makes no difference to the resulting constants, and is MUCH faster on the cbsyst side.
In MyAMI, pressure correction was involved lower down in generating the K meshes parameter fitting, so a new parameter set had to be calculated for each P.


0.3.0 (2017-06-13)
------------------

* Comparison to GLODAPv2 dataset.
* Implemented pressure corrections.


0.2.1 (2017-06-13)
------------------

* Fixed missing dependency that made Pypi install fail.


0.2.0 (2017-06-13)
------------------

* Initial Pypi Release