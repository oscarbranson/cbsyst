import json
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
from datetime import datetime

from .helpers import shape_matcher, load_params, MyAMI_resource_file
from .calc import calc_Fcorr

FCORR_COEFS = load_params('Fcorr_approx.json')

def approximate_Fcorr(TempC=25, Sal=35, Mg=0.0528171, Ca=0.0102821):
    """
    Approximate Fcorr using pre-calculated polynomial.
    
    Faster, but introduces up to ~0.2% error on correction factors in extreme

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius
    Sal : array-like
        Salinity in PSU
    Mg, Ca : array-like
        Mg and Ca concentration in mol/kg
    
    Returns
    -------
    dict
        Containing Fcorr factors for the specified inputs
    """
    
    TempC, Sal, Mg, Ca = shape_matcher(TempC, Sal, Mg, Ca)

    check_limits(TempC, Sal, Mg, Ca)

    in_shape = TempC.shape
    
    # Build design matrix
    TempK = TempC + 273.15   
    X = np.vstack([TempK.ravel(), np.log(TempK.ravel()), Sal.ravel(), Mg.ravel(), Ca.ravel()]).T
    
    # This expands the input to a design matrix for 3rd-order polynomial with cross terms 
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    poly = PolynomialFeatures(degree=3)
    X_ = poly.fit_transform(X)
    
    # calculate and return Fcorr
    return {k: X_.dot(c).reshape(in_shape) for k, c in FCORR_COEFS.items()}


def check_limits(TempC, Sal, Mg, Ca):

    warntext = (
        "\n\nDO NOT USE THESE APPROXIMATE CORRECTION FACTORS" +
        "\nPlease run a full MyAMI run instead."
    )

    if np.any((TempC < 0) | (TempC > 40)):
        raise ValueError('Temperature outside valid limits (0-40 C)' + warntext)

    if np.any((Sal < 30) | (Sal > 40)):
        raise ValueError('Salinity outside valid limits (30-40)' + warntext)

    if np.any((Mg < 0) | (Mg > 0.06)):
        raise ValueError('[Mg] outside valid limits (0-0.06)' + warntext)

    if np.any((Ca < 0) | (Ca > 0.06)):
        raise ValueError('[Ca] outside valid limits (0-0.06)' + warntext)


def generate_approximate_Fcorr_params(n=21, fit_reports=True):
    """
    Generate a Look Up Table (LUT) of parameters used to approximate Fcorr.

    The Fcorr factor is used to correct an empirical K value at
    a given TempC and Sal for Mg and Ca concentration, and should be
    applied as:

    Kcorr = Kempirical * Fcorr

    Parameter ranges are:
        TempC: 0 - 40 Celcius
        Sal: 30-40 PSU
        Mg, Ca: 0, 0.06 mol/kg

    Parameters
    ----------
    n : int
        The number of grid points to calculate for each parameter.
    
    Returns
    -------
    pandas.DataFrame
    """

    TempC = np.linspace(0, 40, n)
    Sal = np.linspace(30, 40, n)
    Mg = np.linspace(0, 0.06, n)
    Ca = np.linspace(0, 0.06, n)

    # grid inputs
    gTempC, gSal, gMg, gCa = np.meshgrid(TempC, Sal, Mg, Ca)
    
    # calculate Fcorr
    Fcorr = calc_Fcorr(Sal=gSal, TempC=gTempC, Mg=gMg, Ca=gCa)
    flat_Fcorr = {k: v.ravel() for k, v in Fcorr.items()}
    
    # generate design matrix for fitting
    X = np.vstack([
        gTempC.ravel() + 273.15,  # TempK
        np.log(gTempC.ravel() + 273.15),  # ln(TempK)
        gSal.ravel(),
        gMg.ravel(),
        gCa.ravel()
    ]).T
    
    poly = PolynomialFeatures(degree=3)
    X_ = poly.fit_transform(X)
    
    # calculate best-fit parameters
    coefs = {}
    for k in flat_Fcorr:
        y = flat_Fcorr[k]
        coefs[k], _, _, _ = lstsq(X_, y)
        
    if fit_reports:
        print(f'Fit reports saved in {MyAMI_resource_file()}')
        pred = {k: X_.dot(c) for k, c in coefs.items()}
        for k in coefs:
            fname = MyAMI_resource_file(f'Fcorr_approx_{k}.png')
            fig, axs = Fcorr_fit_report(k, flat_Fcorr[k], pred[k], X)
            fig.savefig(fname, dpi=150)

    with open(MyAMI_resource_file('Fcorr_approx.json'), 'w') as f:
        json.dump({k: list(v) for k, v in coefs.items()}, f)


def Fcorr_fit_report(k, obs, pred, X):
    
    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    epc = 100 * (pred - obs) / obs  # percentage error

    kwargs = dict(s=2, alpha=0.2, color='k')

    axs = []

    gs = GridSpec(3, 2, wspace=0.1, hspace=0.3)
    fig = plt.figure(figsize=[7, 9])

    ax = fig.add_subplot(gs[0,:])
    ax.scatter(obs, epc, **kwargs)
    ax.set_xlabel('F_corr')
    ax.set_ylabel('% Error')
    ax.set_title(k, weight='bold', loc='left')
    ax.text(0.98, 0.95, f'Generated: {now}', transform=ax.transAxes, ha='right', va='top')
    axs.append(ax)

    ax = fig.add_subplot(gs[1,0])
    ax.scatter(X[:,0] - 273.15, epc, **kwargs)
    ax.set_xlabel('TempC')
    ax.set_ylabel('% Error')
    axs.append(ax)

    ax = fig.add_subplot(gs[1,1])
    ax.scatter(X[:,2], epc, **kwargs)
    ax.set_xlabel('Sal')
    ax.set_yticklabels([])
    axs.append(ax)

    ax = fig.add_subplot(gs[2,0])
    ax.scatter(X[:,3], epc, **kwargs)
    ax.set_xlabel('Mg')
    ax.set_ylabel('% Error')
    axs.append(ax)

    ax = fig.add_subplot(gs[2,1])
    ax.scatter(X[:,4], epc, **kwargs)
    ax.set_xlabel('Ca')
    ax.set_yticklabels([])
    axs.append(ax)

    return fig, axs