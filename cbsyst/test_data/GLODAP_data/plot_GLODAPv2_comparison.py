import os
import pandas as pd
import numpy as np
import cbsyst as cb
import matplotlib.pyplot as plt


def GLODAPv2_comparison(figdir="."):

    print("\n********************************************")
    print("Generating GLODAPv2 Comparison Plots")
    print("********************************************")

    # if the data file is missing, get it.
    if not os.path.exists("./GLODAPv2_pH_DIC_ALK_subset.csv"):
        print("Getting GLODAPv2 Data...")
        from get_GLODAP_data import get_GLODAP

        get_GLODAP()

    # Load data
    print("Importing GLODAPv2 Data...")
    gd = pd.read_csv("./GLODAPv2_pH_DIC_ALK_subset.csv")
    gd.dropna(
        subset=[
            "phtsinsitutp",
            "temperature",
            "salinity",
            "tco2",
            "talk",
            "pressure",
            "phosphate",
            "silicate",
        ],
        inplace=True,
    )
    gd.pressure /= 10  # convert pressure to bar

    # exclude weird cruise 270 data
    gd = gd.loc[gd.cruise != 270]

    # Do the work...
    print("Calculating pH from DIC and TA...")
    cpH = cb.Csys(
        TA=gd.talk,
        DIC=gd.tco2,
        T=gd.temperature,
        S=gd.salinity,
        P=gd.pressure,
        TP=gd.phosphate,
        TSi=gd.silicate,
        BT=415.7,
    )
    print("   Making plots...")
    fig, axs = cplot(gd.phtsinsitutp, cpH.pH, "pH", "Depth", gd.depth)
    fig.savefig(figdir + "/Figures/pH_comparison.png", dpi=200)

    print("Calculating TA from pH and DIC...")
    cTA = cb.Csys(
        pH=gd.phtsinsitutp,
        DIC=gd.tco2,
        T=gd.temperature,
        S=gd.salinity,
        P=gd.pressure,
        TP=gd.phosphate,
        TSi=gd.silicate,
        BT=415.7,
    )
    print("   Making plots...")
    fig, ax = cplot(gd.talk, cTA.TA, "Alk", "Depth", gd.depth)
    fig.savefig("Figures/TA_comparison.png", dpi=200)

    print("Calculating DIC from pH and TA...")
    cDIC = cb.Csys(
        pH=gd.phtsinsitutp,
        TA=gd.talk,
        T=gd.temperature,
        S=gd.salinity,
        P=gd.pressure,
        TP=gd.phosphate,
        TSi=gd.silicate,
        BT=415.7,
    )
    print("   Making plots...")
    fig, ax = cplot(gd.tco2, cDIC.DIC, "DIC", "Depth", gd.depth)
    fig.savefig("Figures/DIC_comparison.png", dpi=200)

    print("Done.")
    print()
    print(" > Plots are saved in : " + figdir + "/Figures/")
    print("********************************************")


# plotting function
def cplot(obs, pred, var, cvar, c, alpha=0.4, pclims=[0.05, 99.95]):
    fig = plt.figure(figsize=[11, 3.5])

    ax1 = fig.add_axes([0.07, 0.13, 0.33, 0.8])
    ax2 = fig.add_axes([0.47, 0.13, 0.33, 0.8])
    hax = fig.add_axes([0.805, 0.13, 0.08, 0.8])
    cax = fig.add_axes([0.9, 0.13, 0.015, 0.8])

    ad = np.concatenate([obs, pred])
    mn, mx = np.percentile(ad, pclims)
    xr = mx - mn
    pad = 0.1 * xr

    # Measured vs Predicted
    cm = ax1.scatter(obs, pred, s=5, alpha=alpha, c=c, lw=0)
    ax1.set_xlim(mn - pad, mx + pad)
    ax1.set_ylim(mn - pad, mx + pad)
    ax1.plot(
        ax1.get_xlim(), ax1.get_xlim(), ls="dashed", c=(0.5, 0.5, 0.5, 0.8), zorder=1
    )

    ax1.set_xlabel("GLODAPv2 Measured")
    ax1.set_ylabel("cbsyst predicted")

    ax1.text(
        0.05,
        0.95,
        "GLODAPv2 " + var,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color=(0.4, 0.4, 0.4),
    )

    # Measured vs Difference
    diff = obs - pred
    mn, mx = np.percentile(diff, pclims)
    xr = mx - mn
    pad = 0.15 * xr

    # Scatterplot
    ax2.scatter(obs, diff, s=5, alpha=alpha, c=c, lw=0)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(mn - pad, mx + pad)
    ax2.axhline(0, ls="dashed", c=(0.5, 0.5, 0.5, 0.8), zorder=1)

    ax2.set_xlabel("GLODAPv2 Measured")
    ax2.set_ylabel("measured - predicted")

    # Histogram
    bins = np.linspace(*ax2.get_ylim(), 100)
    hax.hist(diff, bins=bins, orientation="horizontal", color=(0.6, 0.6, 0.6))
    hax.set_ylim(ax2.get_ylim())
    hax.set_xlabel("n")
    hax.set_yticklabels([])

    # Stats
    median = np.median(diff)
    pc95 = np.percentile(diff, [2.5, 97.5])
    hax.axhline(median, color="r", ls="dashed", zorder=2)
    hax.axhspan(*pc95, color="r", alpha=0.2, zorder=1)
    hax.set_ylim(ax2.get_ylim())
    hax.axhline(0, ls="dashed", c=(0.5, 0.5, 0.5, 0.8), zorder=1)

    ax2.text(
        0.03,
        0.97,
        "Median Offset: {:.1e}".format(median)
        + "\n95% Limits: {:.1e} / +{:.1e}".format(*(pc95 - median)),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        backgroundcolor=(1, 1, 1, 0.5),
    )

    if not isinstance(c, str):
        fig.colorbar(cm, cax=cax, label=cvar)
    else:
        cax.set_visible(False)

    return fig, (ax1, ax2, hax, cax)


if __name__ == "__main__":
    GLODAPv2_comparison()
