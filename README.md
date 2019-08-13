# crispec
A minimal data reduction pipeline for a tiny spectrograph called crispec

Data reduction package for CRISPEC

This script and library handles automatically the data reduction for crispec data. It finds the region of the images where the
 spectra fall, uses spectra of a Hg lamp to identify the strongest emission lines and fit a linear curve to calibrate in wavel
ength, collapses and wavelength calibrates all the spectra, and combines the RGB images into a single spectrum after taking ra
tios to a reference spectrum (usually that of an LED lamp). It also attempts to get the spectral energy distribution by using 
a solar model and an observation of the sun made with crispec.

The script expects a bunch of *.JPG images (crispec spectra) in the working directory, and two .txt files:

calfiles.txt, which lists the names of the .JPG images with spectra from the Hg calibration lamp to be used for wavelength cal
ibration

reffile.txt, which includes the name of a single .JPG image to be use as reference to take flux ratios.

The script produces flat files (.dat) with a header and a data table including:
wavelengths, the signal in the R, G, and B channels, a combine flux ratio relative to the reference spectrum and its uncertain
ty, and a calibrated spectral energy distribution based on a solar model.
It also produces plots with the resulting RGB spectra and the combined relative fluxes in PNG format, and a log file ('crislog
').

Example 
-------

python3 crispec.py

One can avoid the interactive graphical window by typing

python3 crispec.py ni

