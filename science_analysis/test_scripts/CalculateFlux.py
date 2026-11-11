from pathlib import Path

from astropy.io import fits
import numpy as np
from soxs import Spectrum
from xraydb import material_mu

from bc_caldb import CURRENT_CALDB_VER
from science_analysis import BCImageAnalysis, extract_paths


# General Constants
IMAGE_ANALYZER = BCImageAnalysis(caldb_version=CURRENT_CALDB_VER)
CODING_MAP = np.sum([
    exposure_map*10_000 for exposure_map
    in IMAGE_ANALYZER.imager.exposed_area_maps.values()
], axis=0)

# Data Constants
L1_PATH = Path("/archive/science_operations/level1")
OBSID = "202611092351"

# Energy Range Constants
EBINW_EV = 26
ERAW_LOW = 154
ERAW_HIGH = 327

# Mask Constants
MASK_PATTERN = IMAGE_ANALYZER.imager.instrument.mask_pattern
MASK_OPEN_CELL_FRACTION = np.sum(MASK_PATTERN) / np.prod(MASK_PATTERN.shape)
CODED_MASK_CALDB = IMAGE_ANALYZER.imager.instrument._coded_mask
MASK_OPEN_CELL_AREA = CODED_MASK_CALDB.cdelt1 * CODED_MASK_CALDB.cdelt2
MASK_OPEN_CELL_OPEN_AREA = CODED_MASK_CALDB.maskcelx * CODED_MASK_CALDB.maskcely
MASK_OPEN_CELL_OPENNESS = MASK_OPEN_CELL_OPEN_AREA / MASK_OPEN_CELL_AREA
MASK_OPENNESS = MASK_OPEN_CELL_FRACTION * MASK_OPEN_CELL_OPENNESS

# Badpix Constants
BADPIX_MAPS = (
    IMAGE_ANALYZER._badpix.badpix_0,
    IMAGE_ANALYZER._badpix.badpix_1,
    IMAGE_ANALYZER._badpix.badpix_2,
    IMAGE_ANALYZER._badpix.badpix_3,
)
GOOD_FRACTION = np.array([
    1 - IMAGE_ANALYZER.badpix_fraction[0],
    1 - IMAGE_ANALYZER.badpix_fraction[1],
    1 - IMAGE_ANALYZER.badpix_fraction[2],
    1 - IMAGE_ANALYZER.badpix_fraction[3],
], dtype=np.float64)
GOOD_AREA = GOOD_FRACTION * IMAGE_ANALYZER.imager.instrument.detector_area * 10_000
TOTAL_GOOD_FRACTION = np.sum(GOOD_FRACTION) / 4

# Transmission and Absorption Constants
ERAW_BIN_EDGES = np.arange(ERAW_LOW, ERAW_HIGH+2)
EV_BIN_EDGES = EBINW_EV * ERAW_BIN_EDGES
TRANSMISSION_ENERGIES_EV = (EV_BIN_EDGES[1:] + EV_BIN_EDGES[:-1]) / 2
MASK_POLYIMIDE_LAYER_CM = 6e-5
MASK_POLYIMIDE_TRANSMISSION = np.exp(
    -MASK_POLYIMIDE_LAYER_CM * material_mu("Polyimide", TRANSMISSION_ENERGIES_EV)
)
MASK_AL_LAYER_CM = 8.25e-6
MASK_ALUMINUM_TRANSMISSION = np.exp(
    -MASK_AL_LAYER_CM * material_mu("Al", TRANSMISSION_ENERGIES_EV)
)
AL_LAYER_CM = 5e-6
OBF_TRANSMISSION = np.exp(-AL_LAYER_CM * material_mu("Al", TRANSMISSION_ENERGIES_EV))
SIO2_LAYER_CM = 1.028e-5
INACTIVE_TRANSMISSION = np.exp(
    -SIO2_LAYER_CM * material_mu("SiO2", TRANSMISSION_ENERGIES_EV)
)
SI_LAYER_CM = 0.01
ACTIVE_TRANSMISSION = np.exp(-SI_LAYER_CM * material_mu("Si", TRANSMISSION_ENERGIES_EV))
ACTIVE_ABSORPTION = 1 - ACTIVE_TRANSMISSION
QE_BY_E = (
    MASK_POLYIMIDE_TRANSMISSION
    * MASK_ALUMINUM_TRANSMISSION
    * OBF_TRANSMISSION
    * INACTIVE_TRANSMISSION
    * ACTIVE_ABSORPTION
)

# Model Constants
# NOTE: From https://www.aanda.org/articles/aa/pdf/2001/01/aaxmm15.pdf
GAMMA = 2.108
ENORM = 9.59   # photons cm^-2 s^-1 keV^-1
NH = 0.345   # *10^22 cm^2
POWER_LAW_SPEC = Spectrum.from_powerlaw(
    photon_index=GAMMA,
    redshift=0,
    norm=ENORM,
    emin=ERAW_LOW * EBINW_EV / 1_000,
    emax=(ERAW_HIGH + 1) * EBINW_EV / 1_000,
    nbins=len(TRANSMISSION_ENERGIES_EV),
)
POWER_LAW_SPEC.apply_foreground_absorption(NH, model="tbabs")
SIM_GRADE = 0
MODEL_FLUX = POWER_LAW_SPEC.total_flux


# Extracting eventlists and attitude files.
evt_path, att_path, _ = extract_paths(L1_PATH, OBSID)
evt, head = fits.getdata(evt_path, 1, header=True)
att = fits.getdata(att_path, 1)

# Grabbing the stable attitudes and setting pointing to use them
stab_att = att[att["TIME"] >= head["TSTABLE"]]
points = stab_att["POINTING"]
IMAGE_ANALYZER.set_ra_dec_roll(*np.median(points, axis=0))

# Determining time range where all four detectors are fully active and
# pointing is known to be stable
min_time = stab_att["TIME"].min()
max_time = stab_att["TIME"].max()
for det in range(4):
    det_evt = evt[evt["DETID"] == det]
    min_time = max(det_evt["TIME"].min(), min_time)
    max_time = min(det_evt["TIME"].max(), max_time)
time_mask = (evt["TIME"] >= min_time)*(evt["ERAW"] <= max_time)
eraw_mask = (evt["ERAW"] >= ERAW_LOW)*(evt["ERAW"] <= ERAW_HIGH)
filtered_evt = evt[time_mask*eraw_mask]

# Calculating live time
min_time = filtered_evt["TIME"].min()
max_time = filtered_evt["TIME"].max()
livetime = (max_time - min_time) * IMAGE_ANALYZER.LIVETIME_FRACTION   # Hard coded for now.

# Image and find peak
image_hdu = IMAGE_ANALYZER.eventlist_to_image(counts=filtered_evt, header=head, add_wcs=True)
peaks = IMAGE_ANALYZER.find_peaks(image_hdu.data, image_hdu.header)

# We know the brightest peak is the Crab.
# (Peaks are internally sorted by significance).
peak = peaks[0]
peak_x, peak_y = peak["xy"]

# Weight eventlist
weights, flags = IMAGE_ANALYZER.weight_eventlist(counts=filtered_evt, header=head, image_xy=peak["xy"])
peak_counts = np.sum(weights[flags == 0])
bg_counts = len(filtered_evt) - peak_counts
b = bg_counts / livetime / np.sum(GOOD_AREA)
q = peak_counts / b
r = CODING_MAP[peak_y, peak_x] * livetime / b

flux = (1 / MASK_OPENNESS) * q / r
uncert = (1 / MASK_OPENNESS) * (1 / np.sqrt(r))
print(flux, uncert)
print(flux / MODEL_FLUX.value, np.median(QE_BY_E) * TOTAL_GOOD_FRACTION)
print()
