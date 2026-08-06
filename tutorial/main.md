---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: python3
  language: python
  name: python3
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

import numpy as np

import concurrent

import astropy.units as u

import matplotlib.pyplot as plt

sys.path.append("../code/")
import SPHERExSpecPhot as ssp
importlib.reload(ssp)
```

```{code-cell} ipython3
## INPUT FOR TOOL ##

INPUT = dict()

INPUT["target_name"] = "2MASS 21282814-0559310"
INPUT["RA"] = 322.117265*u.degree
INPUT["DEC"] = -5.991964*u.degree
INPUT["tmpdir"] = "../tmp/"
INPUT["outdir"] = "../output/"
INPUT["cutout_size"] = 11*u.pixel
```

```{code-cell} ipython3
## SEARCH LVF IMAGES ===========
lvf_results = ssp.search_lvfs(ra = INPUT["RA"],
                         dec = INPUT["DEC"]
                         )
```

```{code-cell} ipython3
## GET IRSA LINKS ========
irsa_base = "https://irsa.ipac.caltech.edu/"
lvf_results["irsa_uri"] = [os.path.join(irsa_base , t["uri"]) for t in lvf_results]

## GET S3 BUCKET LINKS ==========
bucket_name = "nasa-irsa-spherex"
s3_access = f"s3://{bucket_name}"
lvf_results["s3_access"] = [os.path.join(s3_access, "/".join(t["uri"].split("/")[3:])) for t in lvf_results]
```

```{code-cell} ipython3
## GET CUTOUTS FROM S3 (parallel) =============
# we choose the S3 cloud cutout tool. Alternatively
# once could choose the IRSA cutout tool (_process_cutout_irsa)
## Prepare
lvf_results["cutout_index"] = range(1, len(lvf_results) + 1)
lvf_results["central_wavelength"] = np.full(len(lvf_results), np.nan)
lvf_results["bandwidth"] = np.full(len(lvf_results), np.nan)
lvf_results["hdus"] = np.full(len(lvf_results), None)


## get Cutouts (IRSA)
'''t1 = time.time()
print("Creating cutouts (IRSA)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(
            ssp._process_cutout_irsa,
            row=row,
            ra=INPUT["RA"],
            dec=INPUT["DEC"],
            size=11*6.15*u.arcsec,
            cache=False,
            uri_key="irsa_uri",
        )
        for row in lvf_results[:5]
    ]
    concurrent.futures.wait(futures)
print("Done - time to create cutouts in parallel mode: {:2.2f} minutes.".format( (time.time()-t1)/60 ) )'''

## get Cutouts (S3)
t1 = time.time()
print("Creating cutouts (S3)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(
            ssp._process_cutout_s3,
            row=row,
            ra=INPUT["RA"],
            dec=INPUT["DEC"],
            size=INPUT["cutout_size"],
            cache=False,
            uri_key="s3_access",
        )
        for row in lvf_results
    ]
    concurrent.futures.wait(futures)
print("Done - time to create cutouts in parallel mode: {:2.2f} minutes (= {:2.2f}/image).".format( (time.time()-t1)/60, ((time.time()-t1)/60)/len(lvf_results) ) )

```

```{code-cell} ipython3
## CREATE MULTIEXTENSION FITS
image_hdul = ssp.create_multiFITS(results_table=lvf_results, 
                outdir = "../output/",
                outname = "test",
                savefits = False)
```

```{code-cell} ipython3
## EXTRACT SPECTRUM FROM MULTIEXTENSION FITS
t1 = time.time()
prim_cat, sec_cat = ssp.extract_spectrum(combined_hdul = image_hdul,
                                         lam_bins_width = 0.2 *u.micrometer,
                                         n_processes = 5, chunk_size = 5)
print("Spectra extracted in {:2.2f} seconds.".format( time.time()-t1 ) )
```

```{code-cell} ipython3
## Plot
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(1,1,1)

ax1.plot(prim_cat["lam_int"], prim_cat["flux_int"], "o", markersize=1)

ax1.errorbar(sec_cat["lam_bin"], sec_cat["flux_bin"],
             xerr = 0, yerr = sec_cat["fluxerr_bin"],
             fmt = "o",
             markersize = 1, capsize=2, markerfacecolor="black", capthick=0.5,
            markeredgecolor="black", ecolor="black", elinewidth=0.5)


ax1.set_title(INPUT["target_name"])
ax1.set_xlim(0.75 , 5.0)
ax1.set_ylim(0,200)

ax1.set_xlabel(r"$\lambda_{\rm obs}$ ($\mu m$)")
ax1.set_ylabel(r"Flux (mJy)")


plt.show()
```

```{code-cell} ipython3

```
