import os
import tarfile
import urllib.request as ureq

import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm


# Status bar
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_GLODAP_WOA(save=True):
    # download GLODAP data if not present
    if not os.path.exists("./GLODAP_data/"):
        print("Downloading GLODAP data...")
        os.mkdir("./GLODAP_data")
        # download data
        GLODAP_url = "http://cdiac.ornl.gov/ftp/oceans/GLODAP_Gridded_Data/{:}"
        GLODAP_files = ["Alk.tar.Z", "TCO2.tar.Z"]
        for f in GLODAP_files:
            with TqdmUpTo(
                unit="B", unit_scale=True, miniters=1, desc="GLODAP {:}".format(f)
            ) as t:
                ureq.urlretrieve(
                    GLODAP_url.format(f),
                    "./GLODAP_data/{:}".format(f),
                    reporthook=t.update_to,
                )

    # download WOA data if not present
    if not os.path.exists("./WOA2009_data/"):
        print("Downloading WOA2009 data...")
        os.mkdir("./WOA2009_data")
        # directory: parameter_key
        WOA_url = "ftp://ftp.nodc.noaa.gov/pub/woa/WOA09/DATA/{:}/csv/{:}_00_1d.tar.gz"
        WOA_params = {
            "salinity": "s",
            "temperature": "t",
            "nitrate": "n",
            "phosphate": "p",
            "silicate": "i",
        }
        # download data
        for d, p in WOA_params.items():
            url = WOA_url.format(d, p)
            fname = "./WOA2009_data/" + url.split("/")[-1]

            with TqdmUpTo(
                unit="B", unit_scale=True, miniters=1, desc="WOA2009 {:}".format(d)
            ) as t:
                ureq.urlretrieve(url, fname, reporthook=t.update_to)

    # Load Data
    # ---------

    # GLODAP
    # ++++++
    print("Reading GLODAP data...")

    # Uncompress files (Python can't read them...)
    try:
        for f in [z for z in os.listdir("GLODAP_data/") if "tar.Z" in z]:
            os.system("uncompress ./GLODAP_data/" + f)
    except:
        raise RuntimeError(
            "Cannot unzip GLODAP data files. Perhaps you are\non Windows? If so, please unzip them manually (extension shoudl be .tar)\nthen try again."
        )

    # load GLODAP data
    glodap = {}
    for f in os.listdir("GLODAP_data/"):
        var = f.split(".")[0]
        tar = tarfile.open("GLODAP_data/{:}.tar".format(var))
        tmp = np.genfromtxt(tar.extractfile(var + "/" + var + ".data"), delimiter=",")
        # reshape to 3D array
        glodap[var.lower()] = np.reshape(tmp, (33, 360, 180))

    # axes
    lat = np.genfromtxt(tar.extractfile(var + "/" + "Lat.centers"), delimiter=",")
    lon = np.genfromtxt(tar.extractfile(var + "/" + "Long.centers"), delimiter=",")
    dep = np.genfromtxt(tar.extractfile(var + "/" + "Depth.centers"), delimiter=",")
    glon, gdep, glat = np.meshgrid(lon, dep, lat)

    # WOA-2009
    # ++++++++
    # set up column headins and params
    print("Reading WOA2009 data...")

    columns = [
        "lat",
        "lon",
        "depth",
        "obj_mean",
        "mean",
        "std",
        "se",
        "obj_mean_mean",
        "obj_mean_ann_mean",
        "ngrids",
        "ndata",
    ]
    params = ["s", "t", "n", "p", "i"]
    pdict = {
        k: v
        for k, v in zip(params, ["sal", "temp", "nitrate", "phosphate", "silicate"])
    }

    # read in all data
    cdat = []
    for p in params:
        tar = tarfile.open("WOA2009_data/" + p + "_00_1d.tar.gz")

        dat = []
        for f in tar.getnames():
            dat.append(pd.read_csv(tar.extractfile(f), names=columns, comment="#"))
        dat = pd.concat(dat)
        dat.loc[:, "var"] = p

        cdat.append(dat)
    cdat = pd.concat(cdat).loc[:, ["lat", "lon", "depth", "obj_mean", "var"]]

    # recaset into 3D arrays
    woa = pd.pivot_table(cdat, index=["lat", "lon", "depth"], columns="var")
    woa.columns = woa.columns.droplevel()
    woa.columns = [pdict[c] for c in woa.columns]
    params = [pdict[p] for p in params]

    shape = list(map(len, woa.index.levels))

    gwoa = {}
    for p in params:
        # create empy
        gwoa[p] = np.full(shape, np.nan)
        # fill with variables based on MultiIndex levels
        gwoa[p][woa.index.labels] = woa.loc[:, p].values.flat

    # axes
    woa_lat = woa.index.levels[0].values
    woa_lon = woa.index.levels[1].values
    woa_dep = woa.index.levels[2].values

    # create interpolators for use later.
    interp = {}
    for p in params:
        interp[p] = RegularGridInterpolator((woa_lat, woa_lon, woa_dep), gwoa[p])

    # Combine Datasets
    # ++++++++++++++++
    # dataframe of glodap data
    print("Generating GLODAP-WOA dataset...")
    glodap_woa = pd.DataFrame(
        {"lat": glat.ravel(), "lon": glon.ravel(), "dep": gdep.ravel()}
    )
    for k, v in glodap.items():
        glodap_woa.loc[:, k] = v.ravel()

    # replace -999 with nan
    glodap_woa.replace(-999, np.nan, inplace=True)
    # drop nan
    glodap_woa.dropna(inplace=True, subset=glodap.keys())

    # interpolate WOA parameters onto GLODAP grid
    print("Interpolating WOA2009 onto GLODAP grid...")
    for p in params:
        glodap_woa.loc[:, p] = interp[p](
            glodap_woa.loc[:, ["lat", "lon", "dep"]].values
        )

    # drop failed interpolation values
    glodap_woa.dropna(inplace=True, subset=params)

    # very approx pressure estimate...
    glodap_woa.loc[:, "pres"] = glodap_woa.loc[:, "dep"] / 10

    # reset index
    glodap_woa.reset_index(inplace=True, drop=True)

    # save data
    if save:
        print("Saving...")
        glodap_woa.to_pickle("glodap_woa.pkl")
        return
    else:
        return glodap_woa


if __name__ == "__main__":
    print("\n*************************************************")
    print("Making GLODAP-WOA dataset (after Orr et al, 2015)")
    print("*************************************************")
    get_GLODAP_WOA()
    print("Done.")
    print()
    print("   Saved as: ./glodap_woa.pkl")
    print("             (read with pandas.read_pickle())")
    print("*************************************************\n")
