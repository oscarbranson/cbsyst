.. :changelog:

Release History
---------------

0.5.0 (2024-12-13)
------------------

**COMPLETE UNCERTAINTY PROPAGATION IMPLEMENTATION**

This release implements comprehensive uncertainty propagation for all carbon system calculations using the `uncertainties` package. All 15 carbon system calculation cases now support automatic error propagation when `ufloat` objects are provided as inputs.

**Major New Features:**

* **Full Uncertainty Propagation**: All carbon system functions (cases 1-15) now support uncertainty propagation using the `uncertainties` package
* **Automatic Detection**: Functions automatically detect uncertainty objects and propagate errors appropriately
* **Three Implementation Approaches**:
  
  - **Algebraic functions** (cases 1, 6-9): Native support via `uncertainties` package
  - **Zero-finder functions** (cases 2-3, 5, 10, 12, 14): Enhanced with finite difference uncertainty propagation  
  - **Iterative functions** (cases 4, 11, 13, 15): New decorator-based uncertainty propagation for Newton-Raphson solvers

* **Robust Error Propagation**: Uses mathematically sound finite difference methods for functions incompatible with automatic differentiation
* **Complete Uncertainty Support**: Works with both individual `ufloat` objects and `uarray` objects (correlated uncertainties)
* **Performance Optimized**: Zero performance impact when uncertainties are not used

**Breaking Changes:**

* **Scalar Input/Output Consistency**: Functions now return scalar values when provided with scalar inputs, rather than single-element arrays

  **Migration**: If your code previously accessed results with `[0]` indexing (e.g., `CO2_TA(...)[0]`), remove the indexing for scalar inputs.

**Technical Implementation:**

* **New uncertainty propagation decorator** for iterative functions that use Newton-Raphson methods
* **Enhanced zero-finder wrapper** with finite difference uncertainty propagation
* **Comprehensive test suite** validating all uncertainty propagation scenarios

**Functions Enhanced:**

* **Cases 1, 6-9**: `CO2_pH`, `pH_HCO3`, `pH_CO3`, `pH_TA`, `pH_DIC` (native uncertainty support)
* **Cases 2-3, 5, 10, 12, 14**: `CO2_HCO3`, `CO2_CO3`, `CO2_DIC`, `HCO3_CO3`, `HCO3_DIC`, `CO3_DIC` (zero-finder enhancement)
* **Cases 4, 11, 13, 15**: `CO2_TA`, `HCO3_TA`, `CO3_TA`, `TA_DIC` (decorator-based propagation)

**Usage Example:**

.. code-block:: python

    import uncertainties as unc
    import cbsyst as cb
    
    # Define parameters with uncertainties
    DIC = unc.ufloat(2000, 20)  # 2000 ± 20 μmol/kg
    TA = unc.ufloat(2300, 15)   # 2300 ± 15 μmol/kg
    
    # Calculate with automatic uncertainty propagation
    result = cb.Csys(DIC=DIC, TA=TA, T_in=25, S_in=35)
    print(result.pH)  # Returns: 8.10+/-0.05 (example)
    
    # Works with uarray objects for correlated uncertainties
    import uncertainties.unumpy as unp
    DIC_array = unp.uarray([2000, 2100, 2200], [20, 25, 30])  
    TA_array = unp.uarray([2300, 2350, 2400], [15, 18, 22])
    result = cb.Csys(DIC=DIC_array, TA=TA_array, T_in=25, S_in=35)
    print(result.pH)  # Returns array with uncertainties

**Documentation:**

* Updated all docstrings to reflect uncertainty propagation capabilities
* Added comprehensive uncertainty propagation test suite
* Updated API documentation to reflect scalar/array return behavior changes

0.4.9 (2023-09-04)
------------
Updates to work with Kgen 0.3.0

Main code changes:
* Explicitly set default Ca (0.0102821) and Mg (0.0528171) concentrations so pymyami is not run at ambient conditions (kgen does not accept None as a valid input).
* Explicitly set P_in to 0.0 so pressure calculation is not run at surface conditions.
* Clarified that Ca and Mg are specified for STANDARD seawater (i.e. 35 salinity).
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