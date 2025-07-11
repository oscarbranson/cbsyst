# Usage

## Basic Examples

### Simple Carbonate System Calculation

```python
import cbsyst as cb

# Calculate from pH and DIC
result = cb.Csys(pH=8.1, DIC=2000)
print(f"pCO2: {result.pCO2}")
print(f"Alkalinity: {result.TA}")
```

### pH Scale Conversion

```python
from cbsyst.helpers import pH_scale_converter

# Convert from Total to NBS scale
pH_nbs = pH_scale_converter(pH=8.1, scale_in='Total', scale_out='NBS')
```

## Advanced Usage

More examples coming soon...