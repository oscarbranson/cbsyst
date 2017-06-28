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

I/O
---

