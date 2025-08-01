# Usage

## Basic Examples

### Simple Carbonate System Calculation

```python
import cbsyst as cb

# Calculate from pH and DIC
result = cb.Csys(pHtot=8.1, DIC=2000)

result

> cbsyst Calculation
> ==============================
> Inputs:
> ------------------------------
> pHtot     
> DIC       
> alphaB    
> dBT       
> T_in      
> S_in      
> ------------------------------
> Calculated:
> ------------------------------
> pHtot                     8.10
> pHfree                    8.21
> pHsws                     8.09
> pHNBS                     8.24
> DIC                    2000.00
> TA                     2336.67
> CO2                       9.79
> HCO3                   1751.70
> CO3                     238.51
> pCO2                    345.79
> fCO2                    344.68
> BT                      415.70
> dBT                      39.61
> dBO3                     46.41
> dBO4                     18.70
> OmegaC                    5.74
> OmegaA                    3.78
> ==============================

```

### pH Scale Conversion

```python
from cbsyst.helpers import pH_scale_converter

# Convert from Total to NBS scale
pH_nbs = pH_scale_converter(pH=8.1, scale_in='Total', scale_out='NBS')
```

## Advanced Usage

More examples coming soon...