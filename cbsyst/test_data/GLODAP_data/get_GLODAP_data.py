import os
import zipfile
import urllib.request as ureq

import pandas as pd
import numpy as np

from tqdm import tqdm


# Progress Bar
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


def get_GLODAP(path="./", leave_zip=True):
    if not os.path.exists(path + "/GLODAPv2 Merged Master File.csv.zip"):
        print("Fetching GLODAPv2 Data (Olsen et al, 2016)...")

        GLODAP_urls = [
            "https://cdiac.ess-dive.lbl.gov/ftp/oceans/GLODAPv2/Data_Products/data_product/GLODAPv2%20Merged%20Master%20File.csv.zip",
            "http://cdiac.ornl.gov/ftp/oceans/GLODAPv2/Data_Products/data_product/GLODAPv2%20Merged%20Master%20File.csv.zip"  # this one stopped working April 2021?
        ]
        downloaded = False
        # download GLODAP data
        i = 0
        while not downloaded:
            GLODAP_url = GLODAP_urls[i]
            try:
                with TqdmUpTo(
                    unit="B", unit_scale=True, miniters=1, desc="Downloading GLODAPv2"
                ) as t:
                    ureq.urlretrieve(
                        GLODAP_url,
                        path + "/GLODAPv2 Merged Master File.csv.zip",
                        reporthook=t.update_to,
                    )
                downloaded = True
            except:
                pass
            i += 1

        # # open URL
        # file = requests.get(GLODAP_url, stream=True)
        # total_size = int(file.headers.get('content-length', 0))

        # # Download data
        # with open('./GLODAPv2 Merged Master File.csv.zip', 'wb') as f:
        #     for data in tqdm(file.iter_content(1024), total=total_size / (1024), unit='KB', unit_scale=True):
        #         f.write(data)
    else:
        print("Found GLODAPv2 Data (Olsen et al, 2016)...")

    print("Reading data...")
    # open zip
    zf = zipfile.ZipFile(path + "/GLODAPv2 Merged Master File.csv.zip")

    # read data into pandas
    gd = pd.read_csv(zf.open("GLODAPv2 Merged Master File.csv"))

    # replace missing values with nan
    gd.replace(-9999, np.nan, inplace=True)

    print("Selecting 'good' (flag == 2) data...")
    # isolate good data only (flag = 2)
    # gd.loc[gd.phts25p0f != 2, 'phts25p0'] = np.nan
    gd.loc[gd.phtsinsitutpf != 2, "phtsinsitutp"] = np.nan
    gd.loc[gd.tco2f != 2, "tco2"] = np.nan
    gd.loc[gd.talkf != 2, "talk"] = np.nan
    gd.loc[gd.salinityf != 2, "salinity"] = np.nan
    gd.loc[gd.phosphatef != 2, "phosphate"] = np.nan
    gd.loc[gd.silicatef != 2, "silicate"] = np.nan

    # Identify rows where ph, dic, talk and sal are present
    # phind = ~gd.phtsinsitutp.isnull()
    # dicind = ~gd.tco2.isnull()
    # alkind = ~gd.talk.isnull()
    # salind = ~gd.salinity.isnull()

    gd.dropna(
        subset=[
            "phtsinsitutp",
            "tco2",
            "talk",
            "temperature",
            "salinity",
            "pressure",
            "silicate",
            "phosphate",
        ],
        inplace=True,
    )

    print("Saving data subset...")
    # Isolate those data
    # gds = gd.loc[phind & dicind & alkind & salind, ['phts25p0', 'phtsinsitutp', 'tco2', 'talk', 'temperature', 'salinity',
    #                                                 'cruise', 'station', 'cast', 'year', 'month', 'day', 'hour',
    #                                                 'latitude', 'longitude', 'bottomdepth', 'maxsampdepth', 'bottle',
    #                                                 'pressure', 'depth', 'theta', 'silicate', 'phosphate']]
    gds = gd.loc[
        :,
        [
            "phts25p0",
            "phtsinsitutp",
            "tco2",
            "talk",
            "temperature",
            "salinity",
            "cruise",
            "station",
            "cast",
            "year",
            "month",
            "day",
            "hour",
            "latitude",
            "longitude",
            "bottomdepth",
            "maxsampdepth",
            "bottle",
            "pressure",
            "depth",
            "theta",
            "silicate",
            "phosphate",
        ],
    ]

    gds.to_csv(path + "/GLODAPv2_pH_DIC_ALK_subset.csv", index=False)

    if not leave_zip:
        os.remove(path + "/GLODAPv2 Merged Master File.csv.zip")

    return


if __name__ == "__main__":
    print("\n********************************************")
    print("Get GLODAPv2 carbon data for cbsyst testing.")
    print("********************************************")
    get_GLODAP()
    print("Done.")
    print("********************************************\n")
