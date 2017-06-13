.. :changelog:

Release History
---------------

0.2.0 (2017-06-13)
++++++++++++++++++

* Initial Pypi Release


0.2.1 (2017-06-13)
++++++++++++++++++

* Fixed missing dependency that made Pypi install fail.


0.3.0 (2017-06-14)
++++++++++++++++++

* Comparison to GLODAPv2 dataset.
* Implemented pressure corrections.

0.3.1 (2017-06-14)
++++++++++++++++++

**Moved pressure correction from MyAMI to cbsyst.**
As the pressure correction factor is multiplicative, it makes no difference to the resulting constants, and is MUCH faster on the cbsyst side.
In MyAMI, pressure correction was involved lower down in generating the K meshes parameter fitting, so a new parameter set had to be calculated for each P.
