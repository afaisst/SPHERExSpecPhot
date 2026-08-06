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
import time
from tqdm import tqdm

import numpy as np

import concurrent
import multiprocess as mp
import contextlib

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, sigma_clip

from photutils.aperture import CircularAperture, aperture_photometry

import pyvo

import matplotlib.pyplot as plt



# Suppress logging temporarily to prevent astropy
# from repeatedly printing out warning notices related to alternate WCSs
import logging
logging.getLogger('astropy').setLevel(logging.ERROR)
```

```{code-cell} ipython3
## FUNCTIONS
# Will have to be put in separate file in the end for clean up

def search_lvfs(ra, dec, IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"):

    """
    Search the SPHEREx IRSA TAP service for Level-2 images covering a sky position.
    
    Parameters
    ----------
    ra : `~astropy.units.Quantity`
        Right ascension (ICRS) of the target position.
    
    dec : `~astropy.units.Quantity`
        Declination (ICRS) of the target position.
    
    IRSA_TAP_URL : str, optional
        URL of the IRSA TAP service. Default is the public IRSA TAP endpoint.
    
    Returns
    -------
    `~astropy.table.Table`
        Table of matching SPHEREx Level-2 images containing the columns
        ``energy_bandpassname``, ``time_bounds_lower``,
        ``time_bounds_upper``, and ``uri``.
    """

    ## Set up Search
    adql = f"""
    SELECT
        p.energy_bandpassname, p.time_bounds_lower, p.time_bounds_upper, a.uri
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE CONTAINS(POINT('ICRS', {ra.to(u.degree).value}, {dec.to(u.degree).value}), p.poly) = 1
    """
    
    ## Run Query
    print("Querying IRSA TAP for L2 images…")
    lvf_result = tap_search(adql, maxrec=5000, IRSA_TAP_URL=IRSA_TAP_URL)
    lvf_results = lvf_result.to_table()
    print(f"Found {len(lvf_results)} L2 images covering RA = {ra.to_value(u.degree)}, Dec = {dec.to_value(u.degree)}")
    if lvf_result.query_status == "OVERFLOW":
        print("WARNING: result count hit the MAXREC limit — some images may be missing. "
              "Raise the maxrec value in the tap_service.search() call if needed.")

    return(lvf_results)


def tap_search(adql, maxrec=None, retries=3, delay=5, IRSA_TAP_URL="https://irsa.ipac.caltech.edu/TAP"):
    """Run a TAP query against ``tap_service`` with retry logic for transient errors.

    Parameters
    ----------
    adql : str
        ADQL query string.
    maxrec : int, optional
        Maximum number of rows to return (passed as TAP MAXREC).
    retries : int, optional
        Maximum number of attempts before re-raising the exception.
    delay : float, optional
        Seconds to wait between attempts.
    IRSA_TAP_URL: str
        Link to IRSA TAP. Default is "https://irsa.ipac.caltech.edu/TAP".

    Returns
    -------
    `~pyvo.dal.TAPResults`
        Query result object; call ``.to_table()`` to get an Astropy Table.

    Raises
    ------
    `~pyvo.dal.DALServiceError`
        If the server returns an error on the final attempt.
    """
    import time as _time
    from pyvo.dal import DALServiceError

    ## Set up TAP service
    tap_service = pyvo.dal.TAPService(IRSA_TAP_URL)
    
    for attempt in range(retries):
        try:
            return tap_service.search(adql, maxrec=maxrec)
        except DALServiceError as e:
            if attempt < retries - 1:
                print(f"TAP query failed ({e}); retrying in {delay}s… (attempt {attempt+1}/{retries})")
                _time.sleep(delay)
            else:
                raise


def process_cutout(row, ra, dec, size=None,
                   cache=False, uri_key="s3_uri", repo="S3"):

    """
    Retrieve a FITS image or cutout from a remote repository.
    
    This is a convenience wrapper that dispatches to the appropriate
    processing function depending on the requested repository.
    
    Parameters
    ----------
    row : dict-like
        A mutable table row (e.g., an ``astropy.table.Row``) that is updated
        in place with the downloaded HDUs and associated metadata.
    
    ra : float or `~astropy.units.Quantity`
        Right ascension (ICRS). If a scalar is provided, it is interpreted
        as decimal degrees. If a `~astropy.units.Quantity` is provided, it
        must have angular units.
    
    dec : float or `~astropy.units.Quantity`
        Declination (ICRS). If a scalar is provided, it is interpreted
        as decimal degrees. If a `~astropy.units.Quantity` is provided, it
        must have angular units.
    
    size : int, float, or `~astropy.units.Quantity`, optional
        Size of the requested cutout.
    
        - For ``repo="S3"``, a scalar is interpreted as pixels, while a
          `~astropy.units.Quantity` may be specified in pixels or angular
          units (e.g., ``11*u.pixel`` or ``30*u.arcsec``).
        - For ``repo="IRSA"``, the size represents the angular size of the
          requested cutout. A scalar is interpreted as decimal degrees,
          while a `~astropy.units.Quantity` must have angular units.
    
        The default is ``None`` (functions will choose if not user defined).
    
    cache : bool, optional
        Whether to cache the remote FITS file locally. Default is ``False``.
    
    uri_key : str, optional
        Name of the column in ``row`` containing the URI or base URI used
        to retrieve the FITS data. The interpretation depends on ``repo``.
    
    repo : {"S3", "IRSA"}, optional
        Repository from which to retrieve the FITS data. ``"S3"`` accesses
        cloud-hosted FITS files using Amazon S3, while ``"IRSA"`` retrieves
        cutouts from the IRSA archive. Default is ``"S3"``.
    
    Returns
    -------
    bool
        ``True`` if the retrieval and processing were successful,
        ``False`` otherwise.
    
    Raises
    ------
    ValueError
        If ``repo`` is not one of ``"S3"`` or ``"IRSA"``.
    
    Notes
    -----
    This function modifies ``row`` in place by adding the downloaded HDUs
    and associated spectral metadata.
    """

    
    if repo == "S3":
        return _process_cutout_s3(
            row=row, ra=ra, dec=dec,
            size=size, cache=cache, uri_key=uri_key
        )
    elif repo == "IRSA":
        return _process_cutout_irsa(
            row=row, ra=ra, dec=dec,
            size=size, cache=cache, uri_key=uri_key
        )
    else:
        raise ValueError("repo must be either 'S3' or 'IRSA'.")




def _process_cutout_s3(row, ra, dec, size=11*u.pixel, cache=False, uri_key="s3_uri"):
    """
    Download cutouts from a cloud-hosted FITS file and store them in a table row.

    The function opens a FITS file stored in an S3 bucket using
    ``astropy.io.fits`` with ``fsspec``, extracts a cutout centered on the
    specified sky coordinates for the ``IMAGE``, ``FLAGS``, and ``VARIANCE``
    extensions, and preserves the WCS in each cutout. It also computes the
    central wavelength and spectral bandwidth at the requested position using
    the spectral WCS.

    Parameters
    ----------
    row : dict-like
        A mutable table row (e.g., an ``astropy.table.Row``) that is updated
        in place. The following fields are added:

        - ``"hdus"`` : list of FITS HDUs containing the cutouts.
        - ``"central_wavelength"`` : Central wavelength in microns.
        - ``"bandwidth"`` : Spectral bandwidth in microns.

    ra : float or `~astropy.units.Quantity`
        Right ascension of the cutout center (ICRS). If a scalar is
        provided, it is interpreted as decimal degrees. If a
        `~astropy.units.Quantity` is provided, it must have angular units.

    dec : float or `~astropy.units.Quantity`
        Declination of the cutout center (ICRS). If a scalar is
        provided, it is interpreted as decimal degrees. If a
        `~astropy.units.Quantity` is provided, it must have angular units.

    size : int, float, or `~astropy.units.Quantity`, optional
        Size of the cutout. If a scalar is provided, it is interpreted
        as a size in pixels. Alternatively, an
        `~astropy.units.Quantity` may be supplied (e.g.,
        ``11*u.pixel`` or ``30*u.arcsec``). The default is
        ``11*u.pixel``.

    cache : bool, optional
        Whether to cache the remote FITS file locally. Passed to
        ``astropy.io.fits.open``. Default is ``False``.

    uri_key : str, optional
        Name of the column in ``row`` containing the S3 URI of the FITS file.
        Default is ``"s3_uri"``.

    Returns
    -------
    bool
        Returns ``True`` if the cutout and metadata were successfully
        generated.

    Raises
    ------
    TypeError
        If ``size`` is neither a scalar nor an
        `~astropy.units.Quantity`, or if ``ra`` or ``dec`` are neither
        scalars nor `~astropy.units.Quantity` objects.

    ValueError
        If ``ra`` or ``dec`` are given as
        `~astropy.units.Quantity` objects that do not have angular units.

    Notes
    -----
    This function modifies ``row`` in place. The FITS file is expected to
    contain ``IMAGE``, ``FLAGS``, and ``VARIANCE`` extensions, as well as a
    valid spectral WCS under the ``"W"`` coordinate key.
    """

    # Allow a scalar to be interpreted as pixels.
    if not isinstance(size, u.Quantity):
        if np.isscalar(size):
            size = size * u.pixel
        else:
            raise TypeError(
                "size must be either a scalar (interpreted as pixels) "
                "or an astropy.units.Quantity."
            )

    # Allow scalars to be interpreted as degrees.
    if not isinstance(ra, u.Quantity):
        if np.isscalar(ra):
            ra = ra * u.deg
        else:
            raise TypeError(
                "ra must be either a scalar (interpreted as degrees) "
                "or an astropy.units.Quantity."
            )
    elif not ra.unit.is_equivalent(u.deg):
        raise ValueError("ra must have angular units.")

    if not isinstance(dec, u.Quantity):
        if np.isscalar(dec):
            dec = dec * u.deg
        else:
            raise TypeError(
                "dec must be either a scalar (interpreted as degrees) "
                "or an astropy.units.Quantity."
            )
    elif not dec.unit.is_equivalent(u.deg):
        raise ValueError("dec must have angular units.")

    position = SkyCoord(ra=ra, dec=dec, frame="icrs")

    uri = row[uri_key]

    with fits.open(uri, fsspec_kwargs={"anon": True}, cache=cache) as hdul:

        keys = ["IMAGE", "FLAGS", "VARIANCE"]

        hdus = []
        for key in keys:
            cutout = Cutout2D(
                hdul[key].section,
                position=position,
                size=size,
                wcs=WCS(hdul[key].header),
                mode="partial"
            )

            hdu = fits.PrimaryHDU(data=cutout.data, header=hdul[key].header)
            hdu.header.update(cutout.wcs.to_header())
            hdu.header["EXTNAME"] = f"{hdu.header['EXTNAME']}{row['cutout_index']}"
            hdus.append(hdu)

        row["hdus"] = hdus

        spatial_wcs = WCS(hdul["IMAGE"].header)
        x, y = spatial_wcs.world_to_pixel(position)

        spectral_wcs = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
        spectral_wcs.sip = None
        wavelength, bandpass = spectral_wcs.pixel_to_world(x, y)

        row["central_wavelength"] = wavelength.to(u.um).value
        row["bandwidth"] = bandpass.to(u.um).value

    return(True)


def _process_cutout_irsa(row, ra, dec, size=60*u.arcsec, cache=False, uri_key="irsa_uri"):
    """
    Download a FITS cutout from the IRSA archive and store it in a table row.

    The function opens a remote FITS file hosted by the IRSA archive,
    reads the ``IMAGE``, ``FLAGS``, and ``VARIANCE`` extensions, and stores
    them as FITS HDUs in the supplied table row. It also computes the
    central wavelength and spectral bandwidth at the specified sky
    coordinates using the spectral WCS.

    Parameters
    ----------
    row : dict-like
        A mutable table row (e.g., an ``astropy.table.Row``) that is updated
        in place. The following fields are added:

        - ``"hdus"`` : list of FITS HDUs containing the downloaded images.
        - ``"central_wavelength"`` : Central wavelength in microns.
        - ``"bandwidth"`` : Spectral bandwidth in microns.

    ra : float or `~astropy.units.Quantity`
        Right ascension (ICRS). If a scalar is provided, it is interpreted
        as decimal degrees. If a `~astropy.units.Quantity` is provided, it
        must have angular units.

    dec : float or `~astropy.units.Quantity`
        Declination (ICRS). If a scalar is provided, it is interpreted
        as decimal degrees. If a `~astropy.units.Quantity` is provided, it
        must have angular units.

    size : float or `~astropy.units.Quantity`, optional
        Angular size of the requested cutout. If a scalar is provided,
        it is interpreted as decimal degrees. If a
        `~astropy.units.Quantity` is provided, it must have angular
        units. The default is ``60*u.arcsec``.

    cache : bool, optional
        Whether to cache the remote FITS file locally. Passed to
        ``astropy.io.fits.open``. Default is ``False``.

    uri_key : str, optional
        Name of the column in ``row`` containing the base IRSA cutout URL.
        The requested position and cutout size are appended as query
        parameters. Default is ``"irsa_uri"``.

    Returns
    -------
    bool
        Returns ``True`` if the FITS file was successfully downloaded and
        processed, and ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``ra``, ``dec``, or ``size`` are neither scalars nor
        `~astropy.units.Quantity` objects.
    
    ValueError
        If ``ra``, ``dec``, or ``size`` are given as
        `~astropy.units.Quantity` objects that do not have angular units.

    Notes
    -----
    This function modifies ``row`` in place. On success, the downloaded cutout HDUs
    and spectral metadata are stored in ``row``. If the FITS file
    cannot be opened or processed, the function returns ``False`` and
    leaves the row unchanged.
    """

    # Allow scalars to be interpreted as degrees.
    if not isinstance(size, u.Quantity):
        if np.isscalar(size):
            size = size * u.deg
        else:
            raise TypeError(
                "size must be either a scalar (interpreted as degrees) "
                "or an astropy.units.Quantity."
            )
    elif not size.unit.is_equivalent(u.deg):
        raise ValueError("size must have angular units.")
    
    # Allow scalars to be interpreted as degrees.
    if not isinstance(ra, u.Quantity):
        if np.isscalar(ra):
            ra = ra * u.deg
        else:
            raise TypeError(
                "ra must be either a scalar (interpreted as degrees) "
                "or an astropy.units.Quantity."
            )
    elif not ra.unit.is_equivalent(u.deg):
        raise ValueError("ra must have angular units.")

    if not isinstance(dec, u.Quantity):
        if np.isscalar(dec):
            dec = dec * u.deg
        else:
            raise TypeError(
                "dec must be either a scalar (interpreted as degrees) "
                "or an astropy.units.Quantity."
            )
    elif not dec.unit.is_equivalent(u.deg):
        raise ValueError("dec must have angular units.")


    position = SkyCoord(ra=ra, dec=dec, frame="icrs")

    # construct the URI
    # Note: from IRSA we can directly download the cutouts! 
    uri = row[uri_key]

    #uri = f"{uri}?center={ra.to(u.degree).value},{dec.to(u.degree).value}d&size={size.to(u.degree).value}"
    uri = (
        f"{uri}"
        f"?center={ra.to_value(u.deg)},{dec.to_value(u.deg)}d"
        f"&size={size.to_value(u.deg)}d"
    )

    # Note that some cutouts may not have an overlap. In this case
    # we just skip the row and leave the default values. We can filter
    # out later the images that couldn't be retrieved.
    try:
        with fits.open(uri, cache=cache) as hdul:
    
            keys = ["IMAGE", "FLAGS", "VARIANCE","WCS-WAVE"]
    
            hdus = []
            for key in keys:
                hdu = fits.PrimaryHDU(data=hdul[key].data, header=hdul[key].header)
                hdu.header["EXTNAME"] = f"{hdu.header['EXTNAME']}{row['cutout_index']}"
                hdus.append(hdu)
    
            row["hdus"] = hdus
    
            spatial_wcs = WCS(hdul["IMAGE"].header)
            x, y = spatial_wcs.world_to_pixel(position)
    
            spectral_wcs = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
            spectral_wcs.sip = None
            wavelength, bandpass = spectral_wcs.pixel_to_world(x, y)
    
            row["central_wavelength"] = wavelength.to(u.um).value
            row["bandwidth"] = bandpass.to(u.um).value

        return(True)
        
    except Exception as e:
        print(f"Failed to process {uri}: {e}")
        return(False)


def create_multiFITS(results_table, outdir=None, outname=None, savefits=False):

    """
    Create a multi-extension FITS (MEF) file from a table of retrieved cutouts.
    
    The function filters out unsuccessful cutout retrievals, creates a binary
    table summarizing the cutouts, and combines all image HDUs into a single
    `~astropy.io.fits.HDUList`. Optionally, the MEF can be written to disk.
    
    Parameters
    ----------
    results_table : `~astropy.table.Table`
        Table containing the retrieved cutouts. The table is expected to
        contain the columns ``hdus``, ``cutout_index``,
        ``time_bounds_lower``, ``central_wavelength``, and ``uri``.
    
    outdir : str, optional
        Output directory for the FITS file. Required if
        ``savefits=True``.
    
    outname : str, optional
        Name of the output FITS file. The ``.fits`` extension is added
        automatically if needed. Required if ``savefits=True``.
    
    savefits : bool, optional
        If `True`, write the multi-extension FITS file to disk.
        Default is `False`.
    
    Returns
    -------
    `~astropy.io.fits.HDUList`
        The assembled multi-extension FITS file. The first extension is a
        binary table named ``CUTOUT_INFO`` followed by the image HDUs from
        all successfully retrieved cutouts.
    
    Raises
    ------
    ValueError
        If ``savefits=True`` but ``outdir`` or ``outname`` is not provided.
    """

    ## First select valid images. Recognize the if hdu not equal to None
    sel_good = [h is not None for h in results_table["hdus"]]
    results_table = results_table[sel_good]
    print(f"Final number of cutouts retrieved: {len(results_table)}")

    ## Create a summary table HDU with renamed columns
    cols = fits.ColDefs([
        fits.Column(name="cutout_index", format="J", array=results_table["cutout_index"], unit=""),
        fits.Column(name="observation_date", format="D", array=results_table["time_bounds_lower"], unit="d"),
        fits.Column(name="central_wavelength", format="D", array=results_table["central_wavelength"], unit="um"),
        fits.Column(name="access_url", format="A200", array=results_table["uri"], unit=""),
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = "CUTOUT_INFO"

    ## Create final MEF
    primary_hdu = fits.PrimaryHDU()
    hdulist_list = [primary_hdu, table_hdu]
    hdulist_list.extend(hdu for fits_hdulist in results_table["hdus"] for hdu in fits_hdulist)
    combined_hdulist = fits.HDUList(hdulist_list)

    ## Save if user wants it
    if savefits:
        if outdir is None or outname is None:
            raise ValueError(
                "outdir and outname must be specified when savefits=True."
            )
        fn_out = os.path.join(outdir, f"{outname.removesuffix('.fits')}.fits")
        combined_hdulist.writeto(fn_out, overwrite=True)
    
    return(combined_hdulist)
    
def mjsr_to_jypixel(value_mjsr, pixel_size_arcsec):
    """
    Convert surface brightness from MJy/sr to Jy/pixel.

    Parameters
    ----------
    value_mjsr : float or array-like
        Value(s) in MJy/sr.
    pixel_size_arcsec : float
        Pixel size in arcseconds (assumed square pixels).

    Returns
    -------
    float or ndarray
        Equivalent value(s) in Jy/pixel.
    """
    # Constants
    arcsec_to_rad = np.pi / (180.0 * 3600.0)  # 1 arcsec in radians

    # Convert pixel size from arcsec^2 to steradian
    pixel_area_sr = (pixel_size_arcsec * arcsec_to_rad)**2

    # 1 MJy/sr = 1e6 Jy/sr
    value_jy_sr = value_mjsr * 1e6

    # Multiply by pixel solid angle
    value_jy_pixel = value_jy_sr * pixel_area_sr

    return value_jy_pixel


def measure_spherex_flux_helper(pars):
    """
    Measure the aperture flux in a SPHEREx cutout.

    The image is converted from MJy/sr to mJy/pixel, a sigma-clipped
    background level is estimated and subtracted, and circular aperture
    photometry is performed at the center of the cutout.

    Parameters
    ----------
    pars : tuple
        Tuple containing:

        - ``hdul`` (`~astropy.io.fits.HDUList`):
          FITS HDUList containing an ``IMAGE`` extension.
        - ``aperture_radius`` (float or `~astropy.units.Quantity`):
          Aperture radius. A scalar is interpreted as pixels. If an
          Astropy quantity is provided, it must have pixel units
          (e.g., ``3*u.pixel``).

    Returns
    -------
    float
        Background-subtracted aperture flux in mJy.

    Raises
    ------
    TypeError
        If ``aperture_radius`` is neither a scalar nor an
        `~astropy.units.Quantity`.

    ValueError
        If ``aperture_radius`` is given as a quantity with units other
        than pixels.
    """

    hdul, aperture_radius = pars

    # Allow scalars to be interpreted as pixels.
    if not isinstance(aperture_radius, u.Quantity):
        if np.isscalar(aperture_radius):
            aperture_radius = aperture_radius * u.pixel
        else:
            raise TypeError(
                "aperture_radius must be either a scalar (interpreted as pixels) "
                "or an astropy.units.Quantity."
            )
    elif not aperture_radius.unit.is_equivalent(u.pixel):
        raise ValueError(
            "aperture_radius must have pixel units."
        )

    aperture_radius_px = aperture_radius.to_value(u.pixel)

    ## Load
    img = hdul["IMAGE"].data

    ## Convert flux MJy/sr to mJy/px
    img = mjsr_to_jypixel(
        value_mjsr=img,
        pixel_size_arcsec=6.15
    )
    img = img * 1e3  # Jy/px -> mJy/px

    ## Measure aperture flux
    positions = [(img.shape[1] / 2, img.shape[0] / 2)]

    mean, median, stddev = sigma_clipped_stats(
        img,
        sigma=3,
        maxiters=5,
        mask=np.isnan(img)
    )

    aperture = CircularAperture(
        positions,
        r=aperture_radius_px
    )

    phot_table = aperture_photometry(
        img - median,
        aperture,
        subpixels=10
    )

    flux = phot_table["aperture_sum"][0]

    return flux

def measure_spherex_flux(
    hduls,
    aperture_radius=3*u.pixel,
    n_processes=5,
    chunk_size=5,
):
    """
    Measure aperture fluxes at the centers of SPHEREx cutouts.

    The function performs circular aperture photometry on the ``IMAGE``
    extension of each FITS HDUList in parallel using multiprocessing.
    Fluxes are background-subtracted using a sigma-clipped median and
    returned in mJy.

    Parameters
    ----------
    hduls : iterable of `~astropy.io.fits.HDUList`
        Collection of FITS HDULists containing SPHEREx cutout images.

    aperture_radius : float or `~astropy.units.Quantity`, optional
        Aperture radius. A scalar is interpreted as pixels. If an
        Astropy quantity is provided, it must have pixel units
        (e.g., ``3*u.pixel``). Default is ``3*u.pixel``.

    n_processes : int, optional
        Number of multiprocessing workers. Default is ``5``.

    chunk_size : int, optional
        Number of tasks distributed to each worker at a time in
        ``multiprocessing.Pool.map``. Increasing this value can reduce
        multiprocessing overhead for large numbers of cutouts.
        Default is ``5``.

    Returns
    -------
    `~astropy.units.Quantity`
        Array of aperture fluxes in mJy, with one value per input HDUList.

    Notes
    -----
    The input HDULists are passed to worker processes. For very large
    datasets, it may be more memory efficient to pass FITS filenames
    instead and open the files inside the worker function.
    """

    pars = [(hdul, aperture_radius) for hdul in hduls]

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(
            measure_spherex_flux_helper,
            pars,
            chunksize=chunk_size,
        )

    return np.asarray(results) * u.mJy


def bin_spherex_flux(lam, flux, lam_bins_width=5*u.micrometer):
    """
    Bin SPHEREx spectral flux measurements into wavelength bins.

    The function groups individual wavelength samples into fixed-width
    wavelength bins, applies sigma clipping to remove outliers, and
    computes the median flux and bootstrap uncertainty in each bin.

    Parameters
    ----------
    lam : `~astropy.units.Quantity`
        Wavelength measurements for the spectrum. Expected units are
        convertible to microns.

    flux : `~astropy.units.Quantity`
        Flux measurements corresponding to ``lam``. Expected units are
        convertible to mJy.

    lam_bins_width : `~astropy.units.Quantity`, optional
        Width of the wavelength bins. Default is ``5*u.micrometer``.

    Returns
    -------
    LAM_bin : `~astropy.units.Quantity`
        Median wavelength of each bin in microns.

    LAMERR_bin : `~astropy.units.Quantity`
        Wavelength bin width for each bin in microns.

    FLUX_bin : `~astropy.units.Quantity`
        Median sigma-clipped flux in each bin in mJy.

    FLUXERR_bin : `~astropy.units.Quantity`
        Bootstrap uncertainty on the median flux in each bin in mJy.

    Notes
    -----
    The wavelength grid is currently fixed to the SPHEREx wavelength
    range of 0.7--5 microns. A bin is retained only if it contains more
    than two wavelength samples and at least two valid flux measurements
    after sigma clipping.
    """

    bin_width = lam_bins_width.to_value(u.micrometer)

    xgrid = np.arange(0.7, 5, bin_width / 2)

    LAM_bin = []
    LAMERR_bin = []
    FLUX_bin = []
    FLUXERR_bin = []

    for xx in xgrid:

        sel = np.where(
            (lam.to_value(u.micrometer) >= xx - bin_width / 2)
            & (lam.to_value(u.micrometer) <= xx + bin_width / 2)
        )[0]

        if len(sel) > 2:
            lam_bin = np.nanmedian(lam[sel].value)

            f = flux.value[sel]
            f = f[~np.isnan(f)]

            msk = sigma_clip(f, sigma=3).mask
            f = f[~msk]

            if len(f) > 1:
                flux_bin = np.median(f)
                fluxerr_bin = bootstrap_median_std(
                    f,
                    num_bootstrap_samples=200
                )

                LAM_bin.append(lam_bin)
                LAMERR_bin.append(bin_width)
                FLUX_bin.append(flux_bin)
                FLUXERR_bin.append(fluxerr_bin)

    return (
        np.asarray(LAM_bin) * u.micrometer,
        np.asarray(LAMERR_bin) * u.micrometer,
        np.asarray(FLUX_bin) * u.mJy,
        np.asarray(FLUXERR_bin) * u.mJy,
    )


def bootstrap_median_std(X, num_bootstrap_samples=1000):
    """
    Estimate the uncertainty of the median using bootstrap resampling.

    Parameters
    ----------
    X : array-like
        Input data values.

    num_bootstrap_samples : int, optional
        Number of bootstrap realizations. Default is ``1000``.

    Returns
    -------
    float
        Standard deviation of the bootstrap distribution of the median.

    Notes
    -----
    Each bootstrap sample is generated by drawing ``len(X)`` values from
    the input array with replacement. The returned uncertainty represents
    the statistical uncertainty on the sample median.
    """

    X = np.asarray(X)
    n = len(X)

    samples = np.random.choice(
        X,
        size=(num_bootstrap_samples, n),
        replace=True
    )

    return np.std(np.median(samples, axis=1))
    

def extract_spectrum(combined_hdul,
                     aperture_radius = 3*u.pixel,
                     n_processes = 5,
                     chunk_size = 5,
                     lam_bins_width = 5*u.micrometer,
                     outdir = None,
                     outname = None,
                     savetable = False
                    ):


    """
    Extract a SPHEREx spectrum from a multi-extension FITS cutout file.
    
    This function reads the cutout metadata table from a multi-extension FITS
    file, extracts the image and flag extensions for each observation, performs
    aperture photometry at the center of each cutout, and combines the
    individual measurements into a binned spectrum. The resulting photometry
    tables can optionally be saved as a FITS file.
    
    Parameters
    ----------
    combined_hdul : `~astropy.io.fits.HDUList`
        Multi-extension FITS file containing the cutout images and the
        ``CUTOUT_INFO`` summary table. Image extensions should be named
        ``IMAGE<cutout_index>`` and ``FLAGS<cutout_index>``.
    
    aperture_radius : float or `~astropy.units.Quantity`, optional
        Aperture radius used for photometry. Scalars are interpreted as
        pixels. If a quantity is provided, it must have pixel units.
        Default is ``3*u.pixel``.
    
    n_processes : int, optional
        Number of multiprocessing workers used for aperture photometry.
        Default is ``5``.
    
    chunk_size : int, optional
        Number of cutouts distributed per multiprocessing task batch.
        Default is ``5``.
    
    lam_bins_width : `~astropy.units.Quantity`, optional
        Width of the wavelength bins used to construct the binned spectrum.
        Default is ``5*u.micrometer``.
    
    outdir : str, optional
        Output directory for the extracted spectrum FITS file. Required if
        ``savephoto=True``.
    
    outname : str, optional
        Output filename. The ``.fits`` extension is added automatically if
        needed. Required if ``savephoto=True``.
    
    savephoto : bool, optional
        If `True`, save the extracted photometry and binned spectrum tables
        to a FITS file. Default is ``True``.
    
    Returns
    -------
    tab1 : `~astropy.table.Table`
        Table containing individual measurements with columns:
    
        - ``lam_int`` : wavelength of each observation.
        - ``flux_int`` : aperture flux in mJy.
    
    tab2 : `~astropy.table.Table`
        Table containing the binned spectrum with columns:
    
        - ``lam_bin`` : median wavelength of each bin.
        - ``flux_bin`` : median sigma-clipped flux in each bin.
        - ``lamerr_bin`` : wavelength bin width.
        - ``fluxerr_bin`` : bootstrap uncertainty of the binned flux.
    
    Raises
    ------
    ValueError
        If ``savephoto=True`` but ``outdir`` or ``outname`` is not provided.
    
    Notes
    -----
    The extracted fluxes are aperture fluxes measured from the SPHEREx
    ``IMAGE`` extensions. Background subtraction and sigma clipping are
    performed during the photometry and binning steps.
    """
    
    ## get Summary Table
    summary_table = Table.read(combined_hdul , hdu=1)

    ## Get HDULs
    HDULS = []
    for cutout_id in summary_table["cutout_index"]:
        tmp = fits.HDUList( [combined_hdul[f"IMAGE{cutout_id}"].copy() , combined_hdul[f"FLAGS{cutout_id}"].copy()] ) 
        tmp[0].header["EXTNAME"] = "IMAGE"
        tmp[1].header["EXTNAME"] = "FLAGS"
        HDULS.append(tmp)

    ## Measure Photometry
    lam = summary_table["central_wavelength"].to(u.micrometer)
    flux = measure_spherex_flux(HDULS , aperture_radius, n_processes, chunk_size)

    ## Bin Flux/Lambda
    lam_bin, lamerr_bin, flux_bin , fluxerr_bin = bin_spherex_flux(lam = lam,
                                                           flux = flux ,
                                                           lam_bins_width = lam_bins_width)

    ## Combine in table
    tab1 = Table([lam,flux], names=["lam_int","flux_int"])
    tab2 = Table([lam_bin,flux_bin, lamerr_bin, fluxerr_bin], names=["lam_bin","flux_bin","lamerr_bin","fluxerr_bin"])


    ## Save Table
    if savetable:
        if outdir is None or outname is None:
            raise ValueError(
                "outdir and outname must be specified when savetable=True."
            )
        fn_out = os.path.join(outdir, f"{outname.removesuffix('.fits')}.fits")
        
        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.BinTableHDU(tab1, name='PRIM_CAT')
        hdu2 = fits.BinTableHDU(tab2, name='SECU_CAT')
        
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        hdul.writeto(fn_out, overwrite=True)
    
    return(tab1, tab2)

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
lvf_results = search_lvfs(ra = INPUT["RA"],
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
print("Starting to create cutouts (IRSA)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(
            _process_cutout_irsa,
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
print("Starting to create cutouts (S3)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(
            _process_cutout_s3,
            row=row,
            ra=INPUT["RA"],
            dec=INPUT["DEC"],
            size=INPUT["cutout_size"],
            cache=False,
            uri_key="s3_access",
        )
        for row in lvf_results[:30]
    ]
    concurrent.futures.wait(futures)
print("Done - time to create cutouts in parallel mode: {:2.2f} minutes.".format( (time.time()-t1)/60 ) )


```

```{code-cell} ipython3
## CREATE MULTIEXTENSION FITS
image_hdul = create_multiFITS(results_table=lvf_results, 
                outdir = "../output/",
                outname = "test",
                savefits = False)
```

```{code-cell} ipython3
%%time
## EXTRACT SPECTRUM FROM MULTIEXTENSION FITS
tmp1,tmp2 = extract_spectrum(combined_hdul = image_hdul)
```

```{code-cell} ipython3
plt.plot(tmp1["lam_int"], tmp1["flux_int"], "o")
#plt.plot(tmp2["lam_bin"], tmp2["flux_bin"], "o")
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
