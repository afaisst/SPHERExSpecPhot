<!-- #region -->
# SPHERExSpecPhot

Python tools for extracting low-resolution spectra (spectrophotometry) from **SPHEREx** observations at user-specified sky coordinates.

The package provides a simple interface for obtaining SPHEREx spectra from calibrated SPHEREx data products and includes a tutorial demonstrating a typical workflow.

## Features

* Extract SPHEREx spectra at arbitrary sky coordinates (RA, Dec)
* Handle SPHEREx spectral image products
* Return calibrated spectra and associated metadata
* Example notebook demonstrating a complete workflow

## Repository Structure

```text
SPHERExSpecPhot/
├── code/          # Core extraction routines
├── tutorial/      # Example notebook and tutorial
└── README.md
```

## Installation

Clone the repository

```bash
git clone https://github.com/afaisst/SPHERExSpecPhot.git
cd SPHERExSpecPhot
```

It is recommended to use a dedicated Python environment.

For example,

```bash
python -m venv spherex
source spherex/bin/activate
```

Install the required Python packages (if available):

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is included, install the dependencies manually.

## Quick Start

The main functionality is contained in the `code/` directory.

A complete example is provided in

```text
tutorial/
```

which demonstrates how to

1. Load SPHEREx data products.
2. Specify a target sky position.
3. Extract the spectrum.
4. Visualize and analyze the resulting spectrophotometry.

## Inputs

The extraction routines require

* SPHEREx calibrated data products
* Target sky coordinates (Right Ascension and Declination)

Additional options may be available depending on the extraction routine (e.g., extraction aperture, quality cuts, or background estimation).

## Output

The software returns the extracted SPHEREx spectrum, including

* wavelength
* flux density
* uncertainties (when available)
* quality flags and metadata

The exact output format is documented in the tutorial notebook.

## Tutorial

A step-by-step example is available in

```text
tutorial/
```

The notebook walks through the complete extraction process and serves as the recommended starting point for new users.

## Status

This repository is under active development. Interfaces and functionality may evolve as SPHEREx data products mature.

Contributions, bug reports, and feature requests are welcome through GitHub Issues and Pull Requests.

## Citation

If this software contributes to your scientific work, please cite:

* this GitHub repository
* the relevant SPHEREx mission and data release papers

A Zenodo DOI can be added here once the repository is archived.

## License

See the `LICENSE` file for licensing information.

<!-- #endregion -->

```python

```
