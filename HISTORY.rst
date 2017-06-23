.. :changelog:

Release History
---------------

0.2.0 (2017-06-13)
++++++++++++++++++

* Initial Pypi Release


0.2.1 (2017-06-13)
++++++++++++++++++

* Fixed missing dependency that made Pypi install fail.


0.3.0 (2017-06-13)
++++++++++++++++++

* Comparison to GLODAPv2 dataset.
* Implemented pressure corrections.

0.3.1 (2017-06-13)
++++++++++++++++++

**Moved pressure correction from MyAMI to cbsyst.**
As the pressure correction factor is multiplicative, it makes no difference to the resulting constants, and is MUCH faster on the cbsyst side.
In MyAMI, pressure correction was involved lower down in generating the K meshes parameter fitting, so a new parameter set had to be calculated for each P.

0.3.2 (2017-06-14)
++++++++++++++++++

* Moved pressure correction back into MyAMI_V2 functions, but still after parameter calculation so speed increase is maintained.
* Added data_out function for exporting data.
* Minor idiot-proofing of minor functions.
* Improved tests.
* General cleanup.

0.3.3 (2016-06-21)
++++++++++++++++++

* Implemented nutrient alkalinity for all except CO3_TA and HCO3_TA cases using parameterisation of Matlab CO2SYS.

0.3.4 (2016-06-23)
++++++++++++++++++

* CO2SYS comparison with GLODAPv2 Bottle data.
* Minor parameter fixes and equation updates to match CO2SYS.
* pH scale correction of KP1, KP2, KP3, KSi and KW