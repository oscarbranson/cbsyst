Contributing
============

If you'd like to contribute, a checklist:

1. Claim a ToDo item by linking adding yourself after in in square brackets, e.g. [@github/oscarbranson]
2. Fork repository
3. Make changes
4. Write unittests for any additions within test_cbsyst.py or test_MyAMI_V2.py, as appropriate.
5. Run tests using setup.py test, and make sure your updated code passes all tests.
6. Submit pull request.

**Tests**
Current unittests check internal consistency of functions against stable-state reference values, and compare the output of Csys against reference carbon speciation data. *Do not change existing tests without good reason*.

TO-DO
=====
A list of bite-sized tasks that can usefully be done.

Vectorising
-----------
A lot of the solver functions would be sped up by vectorising. 
At the moment, most serially apply zero_finder functions to solve for H.
In [carbon_fns](cbsyst/carbon_fns.py):

- [ ] CO2SYS-style zero finder for HCO3_TA (case 11)
- [ ] CO2SYS-style zero finder for CO3_TA (case 13)
- [ ] Vectorise CO2_HCO3 (case 2)
- [ ] Vectorise CO2_CO3 (case 3)
- [ ] Vectorise CO2_DIC (case 5)
- [ ] Vectorise HCO3_CO3 (case 10)
- [ ] Vectorise HCO3_DIC (case 12)
- [ ] Vectorise CO3_DIC (case 14)

Condition I/O
-------------
Currently, input and output conditions are the same.
It would be good to be able to separate input/output T, S, P conditions.
This should be relatively simple and inexpensive, and involve a second call to MyAMI_get_Ks and re-calculating the final speciations.
Suggest wrapping this in a separate function which takes Ks, and outputs all conditions from a single parameter combination (e.g. DIC and H for Csys).

- [ ] Support separate input output conditions [@github/oscarbranson]

Revelle Factor
--------------
Trivial, but not done yet. Thought: when is the most efficient place to do this?

- [ ] Write a function to calculate the Revelle factor, and report it.

Python 2.7 compatability
------------------------
This is low priority, but would be nice.
Current incompatability stems from repeated use of 'Extended Tuple Unpacking' (using *) in numerous function inputs throughout.
Can be solved with a generator wrapper (as described [here](https://stackoverflow.com/questions/5333680/extended-tuple-unpacking-in-python-2)).

- [ ] Replace all instances of Extended Tuple Unpacking with generators.
- [ ] Make import structure python 2 & 3 compatible.

*Note: To run tests in python 2.x, you need to remove the python 2 incompatability warning in setup.py*

Parameter I/O
-------------
**This is shelved for the moment - revisit later if it becomes an issue**

At the moment, the Xsys functions can only take single combinations of input parameters.
It might be useful to allow for the calculation of multiple combinations of input combinations at once (a la CO2SYS.m).
However, this would require a major re-tooling of the algorithms, which currently identify parameters to be calculated by whether they are 'None' or not.
I reckon the effort involved here would be much greater than the inconvenience for the user having to run the function twice for different sets of input parameters.
How often do people actually use the ability of CO2SYS.m to take more than one combination of input parameters?

