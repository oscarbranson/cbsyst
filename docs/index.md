# cbsyst

Python tools for seawater carbonate system calculations (in modified seawater).

## Overview

`cbsyst` provides a comprehensive suite of tools for calculating seawater carbonate system parameters in seawater. This package is designed for oceanographers, marine chemists, and researchers working with seawater chemistry data.

## Features

- Complete carbonate system calculations
- pH scale conversions
- Temperature and pressure corrections
- Seawater composition corrections (Mg, Ca) via [Kgen](https://doi.org/10.1029/2023GC011417)

## Quick Start

```python
import cbsyst as cb

# Example calculation
result = cb.Csys(pHtot=8.1, DIC=2000)
print(result.pCO2)
```

## Navigation

- [Installation](installation.md) - How to install cbsyst
- [Usage](usage.md) - Examples and tutorials
- [API Reference](api.md) - Complete function documentation
- [Performance](performance.md) - Benchmarking `cbsyst` against real-world data.
