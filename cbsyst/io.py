from .dataclasses import CarbonSystemParams

def validate_carbon_inputs(params: CarbonSystemParams) -> None:
    """Validate that sufficient parameters are provided"""
    carbon_params = [params.pHtot, params.DIC, params.TA, params.CO2, 
                    params.HCO3, params.CO3, params.pCO2, params.fCO2]
    omega_params = [params.OmegaA, params.OmegaC]
    
    n_carbon = sum(1 for p in carbon_params if p is not None)
    n_omega = sum(1 for p in omega_params if p is not None)
    
    if n_carbon + n_omega < 2:
        raise ValueError("Must provide at least two carbon system parameters")
    
def validate_inputs(params: CarbonSystemParams):
    carbon_params = [params.pHtot, params.DIC, params.TA, params.CO2, 
                    params.HCO3, params.CO3, params.pCO2, params.fCO2]
    omega_params = [params.OmegaA, params.OmegaC]
    
    n_carbon = sum(1 for p in carbon_params if p is not None)
    n_omega = sum(1 for p in omega_params if p is not None)

    if n_carbon == 1 and n_omega == 1:
        if params.CO3 is not None:
            raise ValueError("CO3 and Omega are an invalid input combination.")
    if n_carbon < 2:
        raise ValueError("Must provide at least two carbon system parameters")