# Contributing

Contributions welcome! As long as you adhere to the following:

## General

All welcome.
Be nice to each other.
Constructive criticism<sup>+</sup> encouraged, being a jerk<sup>*</sup> will get you banned.

If you're having issues with a fellow contributor, feel free to <a href="mailto:oscarbranson@gmail.com">contact Oscar</a> about it.

<sup>+</sup>Constructuve criticism: assessing work critically and fairly, and offering constructive suggestions for how to improve and address your criticism.

<sup>*</sup> Being a jerk: unhelpful, offensive, rude, intolerant comments within any part of the project, or outside but relating to the project.

## Creating Issues
Be as explicit as possible, [referencing commits, functions, files as appropriate](https://help.github.com/articles/autolinked-references-and-urls/).

If it's a bug, provide a [minimal working example](https://stackoverflow.com/help/mcve), and some information about your system (python & dependency versions, operating system).


## Contributing Code

Contribution workflow:

0. If an issue doesn't exist, create one so that people know you're working on it.
1. Note your intention to contribute on the appropriate issue, and you'll be asigned to the issue.
2. Fork repository
3. Make changes
4. Write unittests for any additions within test_cbsyst.py or test_MyAMI_V2.py, as appropriate.
5. Run tests using setup.py test, and make sure your updated code passes all tests.
6. Submit pull request, referencing issues as appropriate.

### Coding Style

Please make sure all code is well-commented, and adheres to [PEP8 Guidelines](https://www.python.org/dev/peps/pep-0008/) (although I don't mind too much about line length).
Strongly recommend using a syntax checker.

Make sure you update docstrings in line with any code changes.

### Tests

Currently using the unittest module, and testing via ``setup.py test``.

Current unittests check internal consistency of functions against stable-state reference values, and compare the output of Csys against reference carbon speciation data. 

*Do not change existing tests without good reason*.

## Project Structure

```
cbsyst/
  |--- boron.py  : Functions for calculating relating to B speciation.
  |--- boron_isotopes.py  : Functions for calculating relating to B isotopes.
  |--- carbon.py  : Functions for calculating C speciation.
  |--- cbsyst.py  : User-facing functions used for calculating seawater chemistry.
  |--- helpers.py  : General functions that are used elsewhere.
  |--- MyAMI_V2.py  : Functions for K0, K1, K2, KB, KW, KS, KspA, KspC and [Mg] and [Ca] corrections.
  |--- non_MyAMI_constants.py  : Functions to calculate any constants that are not handled by MyAMI.
```