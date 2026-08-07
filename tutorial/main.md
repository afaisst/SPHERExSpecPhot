---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
#!pip install multiprocess
#!pip install photutils
```

```{code-cell} ipython3
import os
import sys
import time
import importlib
from tqdm import tqdm

import numpy as np

import concurrent
from functools import partial
import fsspec

import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

sys.path.append("../code/")
import SPHERExSpecPhot as ssp
importlib.reload(ssp)
```

```{code-cell} ipython3
## INPUT FOR TOOL ##

INPUT = dict()

## Calibration Star
INPUT["target_name"] = "2MASS 21282814-0559310"
INPUT["RA"] = 322.117265*u.degree
INPUT["DEC"] = -5.991964*u.degree
INPUT["tmpdir"] = "../tmp/"
INPUT["outdir"] = "../output/"
INPUT["cutout_size"] = 11*u.pix
INPUT["aperture_radius"] = 3*u.pix

## Deep Field Source
#INPUT["target_name"] = "DESI-39633458741381032"
#INPUT["RA"] = 270.5991*u.degree
#INPUT["DEC"] = 66.5608*u.degree
#INPUT["tmpdir"] = "../tmp/"
#INPUT["outdir"] = "../output/"
#INPUT["cutout_size"] = 11*u.pixel
#INPUT["aperture_radius"] = 3*u.pix

## Galaxy
#INPUT["target_name"] = "DESI-39633462931489543"
#INPUT["RA"] = 180.149670*u.degree
#INPUT["DEC"] = 67.093939*u.degree
#INPUT["tmpdir"] = "../tmp/"
#INPUT["outdir"] = "../output/"
#INPUT["cutout_size"] = 11*u.pixel
#INPUT["aperture_radius"] = 3*u.pix

## Galaxy
#INPUT["target_name"] = "DESI-39627409825203277"
#INPUT["RA"] = 65.912846*u.degree
#INPUT["DEC"] = -15.718582*u.degree
#INPUT["tmpdir"] = "../tmp/"
#INPUT["outdir"] = "../output/"
#INPUT["cutout_size"] = 11*u.pixel
#INPUT["aperture_radius"] = 3*u.pix
```

```{code-cell} ipython3
## SEARCH LVF IMAGES ===========
lvf_results = ssp.search_lvfs(ra = INPUT["RA"],
                         dec = INPUT["DEC"],
                         maxrec = 1000
                         )
```

```{code-cell} ipython3
## GET LINKS FOR IRSA AND S3 CLOUD STORAGE ======
lvf_results = ssp.get_links(lvf_results)
```

```{code-cell} ipython3
## GET CUTOUTS FROM S3 or IRSA (parallel) =============
# We have two options
# 1. the S3 cloud
# 2. directly download cutout from IRSA

## Prepare
lvf_results["cutout_index"] = range(1, len(lvf_results) + 1)
lvf_results["central_wavelength"] = np.full(len(lvf_results), np.nan)
lvf_results["bandwidth"] = np.full(len(lvf_results), np.nan)
lvf_results["hdus"] = np.full(len(lvf_results), None)

REPO = "IRSA" # IRSA | S3

## get Cutouts (IRSA)
if REPO == "IRSA":
    t1 = time.time()
    print("Creating cutouts (IRSA)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                ssp.process_cutout_irsa,
                row=row,
                position=SkyCoord(ra=INPUT["RA"], dec=INPUT["DEC"], frame="icrs"),
                size=11*6.15*u.arcsec,
                keys=['IMAGE','FLAGS','VARIANCE'],
                cache=False,
                NTRIES=15,
                SLEEP=1.5,
                uri_key="irsa_uri",
            )
            for row in lvf_results
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing cutouts"
        ):
            future.result()
    print(f"Done (IRSA) - time to create {len(lvf_results)} cutouts in parallel mode:  {np.round((time.time()-t1)/60,2)} minutes (= {np.round(((time.time()-t1))/len(lvf_results),2)}s/image).")
    print(f"Number of failed downloads: {len(np.where( lvf_results["hdus"] == None)[0])}")


## get Cutouts (S3)
if REPO == "CLOUD":
    t1 = time.time()
    print("Creating cutouts (S3)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                ssp.process_cutout_s3,
                row=row,
                position=SkyCoord(ra=INPUT["RA"], dec=INPUT["DEC"], frame="icrs"),
                size=INPUT["cutout_size"],
                keys=['IMAGE','FLAGS','VARIANCE'],
                cache=False,
                uri_key="s3_access",
                fs=fsspec.filesystem("s3", anon=True),
            )
            for row in lvf_results
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing cutouts"
        ):
            future.result()
    print(f"Done - time to create {len(lvf_results)} cutouts in parallel mode:  {np.round((time.time()-t1)/60,2)} minutes (= {np.round(((time.time()-t1))/len(lvf_results),2)}s/image).")
```

```{code-cell} ipython3
## CREATE MULTIEXTENSION FITS
combined_hdul = ssp.create_multiFITS(results_table=lvf_results, 
                outdir = "../output/",
                outname = "test",
                savefits = False)
```

```{code-cell} ipython3
## EXTRACT SPECTRUM FROM MULTIEXTENSION FITS
t1 = time.time()
prim_cat, sec_cat = ssp.extract_spectrum(combined_hdul = combined_hdul,
                                         aperture_radius = INPUT["aperture_radius"],
                                         lam_bins_width = 0.3 *u.micrometer,
                                         n_processes = 10, chunk_size = 5)
print("Spectra extracted in {:2.2f} seconds.".format( time.time()-t1 ) )
```

```{code-cell} ipython3
## Plot
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(1,1,1)

sel_good = np.where(prim_cat["good"])[0]
ax1.plot(prim_cat["lam_int"][sel_good], prim_cat["flux_int"][sel_good], "o", markersize=1, label="Individual Data")

ax1.errorbar(sec_cat["lam_bin"], sec_cat["flux_bin"],
             xerr = 0, yerr = sec_cat["fluxerr_bin"],
             fmt = "o",
             markersize = 1, capsize=2, markerfacecolor="black", capthick=0.5,
            markeredgecolor="black", ecolor="black", elinewidth=0.5,
            label="Binned Data")


ylims = np.nanpercentile(prim_cat["flux_int"] , q=(2,97))

ax1.legend(loc="upper right", fontsize=12)
ax1.set_title(INPUT["target_name"])
ax1.set_xlim(0.6 , 5.1)
ax1.set_ylim(ylims)


ax1.set_xlabel(r"$\lambda_{\rm obs}$ ($\mu m$)")
ax1.set_ylabel(r"Flux (mJy)")


plt.show()
```

```{code-cell} ipython3

```
