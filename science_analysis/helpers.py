"""Functions used by multiple files in bc_science_analysis."""

from os import PathLike
from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime


from science_analysis.bc_image_analyzer import BCImageAnalysis


def extract_paths(l1_dir: PathLike | str, obsid: str) -> tuple[Path, Path, Path]:
    """Extract paths to the stable eventlist, attitude file, and base
    observation directory.

    Arguments:
        - l1_dir: Path to the base directory holding level1 fits files.
        - obsid: Observation id to retrieve the paths for.
    """
    yyyy = obsid[:4]
    mm = obsid[4:6]
    dd = obsid[6:8]

    obsid_path = Path(l1_dir) / f"{yyyy}_{mm}" / f"{yyyy}{mm}{dd}" / obsid
    attitude_path = obsid_path / "auxil" / f"bl{obsid}.att.gz"
    events_path = obsid_path / "event" / f"bl{obsid}phpo_uf.evt.gz"

    if not events_path.exists():
        events_path = obsid_path / "event" / f"bl{obsid}pfpo_uf.evt.gz"

    return events_path, attitude_path, obsid_path


def grab_caldb(
    caldb: list[str],
) -> tuple[Optional[str], Optional[Path], Optional[Path]]:
    if len(caldb) == 1:
        return caldb[0], None, None

    if len(caldb) == 2:
        coded_mask = Path(caldb[0])
        teldef = Path(caldb[1])
        return None, coded_mask, teldef

    raise ValueError(
        "You must provide a caldb version, or paths to a coded mask file and teldef file."
    )


def initialize_imager(
    *,
    caldb: list[str],
    use_subpixel: bool = False,
    resolution: int = 1,
    global_balance: bool = False,
    show_frame: bool = False,
    overwrite: bool = False,
) -> BCImageAnalysis:
    caldb_version, coded_mask, teldef = grab_caldb(caldb)
    imager = BCImageAnalysis(
        caldb_version=caldb_version,
        coded_mask_file=coded_mask,
        teldef_file=teldef,
        use_subpixel=use_subpixel,
        resolution=resolution,
        balance_per_det=not global_balance,
        hide_frame=not show_frame,
        overwrite=overwrite,
    )
    return imager


def plot_image(image_hdu: fits.PrimaryHDU) -> tuple[Figure, Axes]:
    """Plot the image within an image hdu on RA/DEC axes if WCS is
    provided, or X/Y axes if not.
    """
    wcs = WCS(image_hdu.header)
    if wcs.has_celestial:
        fig, ax = plt.subplots(subplot_kw={"projection": wcs})
        ax.set_xlabel(f"{ax.get_xlabel().split('.')[-1].upper()} (J2000)")
        ax.set_ylabel(f"{ax.get_ylabel().split('.')[-1].upper()} (J2000)")
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel(f"Image X")
        ax.set_ylabel(f"Image Y")

    plotted_image = ax.imshow(image_hdu.data, origin="lower", aspect="equal")
    fig.colorbar(plotted_image, ax=ax, label="Counts")
    ax.grid(True, color="white", ls="dotted")

    return fig, ax

def write_lightcurve(
        event_file: PathLike | str,
        time_bine: np.ndarray,
        lc: np.ndarray,
        lc_err: np.ndarray,  
        obj_ra: float,
        obj_dec: float,
        e_min: float,
        e_max: float,
        outfile: Optional[str | PathLike] = None,
        reprocessing: bool = False,

):
    '''Create HDU list from event file, timebin edges, light curve, light curve error
        
        Parameters
        ----------
        event_file : PathLike
            Path to the event file.
        time_bine : np.ndarray
            Array of time edges.
        lc : np.ndarray
            Array of light curve values.
        lc_err : np.ndarray
            Array of light curve errors.
        obj_ra: float,
            Right ascension of the object.
        obj_dec: float,
            Declination of the object.
        e_min: float,
            Minimum energy of the light curve.
        e_max: float,
            Maximum energy of the light curve.
        outfile: Optional[str | PathLike] = None,
            Path to the output file. If None, the HDU list will be returned.
        reprocessing: bool = False,
            If True, the SEQPNUM will be incremented by 1. If False, it will remain the same.
        Returns
        -------
        fits.HDUList'''
    
    ### Pull event file header and data ###
    with fits.open(event_file) as hdul:
        evt_prim_ext_hdu = hdul[0]
        evt_prim_header = evt_prim_ext_hdu.header
        
        evt_first_ext_hdu = hdul[1]
        evt_first_header = evt_first_ext_hdu.header
        evt_first_data = evt_first_ext_hdu.data

    ### architecture to build headers ###
    COPY = object()   # marker meaning: take the value from evt_first_header

    ### Create primary Header ###
    prim_keywords = [
        ("TELESCOP", "BLACKCAT",                                    "Telescope mission name"),
        ("INSTRUME", "BLACKCAT",                                    "Instrument name"),
        ("DATAMODE", "PHOTON",                                      "Datamode"),
        ("OBS_ID",   COPY,                                          "Observation ID"),
        ("DATE-OBS", COPY,                                          "Start Date"),
        ("DATE-END", COPY,                                          "Stop Date"),
        ("ORIGIN",   "PENNSTATE",                                   "Origin of the FITS file"),
        ("DATE",     datetime.now().isoformat(timespec="seconds"),  "File creation date"), 
        ("CREATOR",  "BC_SCIENCE_ANALYSIS",                         "Software that created the file"),
    ]
    prim_header = fits.Header()

    for key, value, comment in prim_keywords:
        if value is COPY:
            value = evt_prim_header[key]   # pull the value from the event file
        prim_header[key] = (value, comment)

    prim_hdu = fits.PrimaryHDU(header=prim_header)

    ### Create primary HDU ###
    prim_hdu = fits.PrimaryHDU(header=prim_header)

    ext_data = fits.ColDefs([
        fits.Column(name="TIME",  format="1D", unit="s",       array=time_bine[:-1]),
        fits.Column(name="RATE",  format="1E", unit="count/s", array=lc),
        fits.Column(name="ERROR", format="1E", unit="count/s", array=lc_err),
    ])
    ext_hdu = fits.BinTableHDU.from_columns(ext_data, name="LIGHTCURVE")

    ext_hdu.header.comments["TTYPE1"] = "Mission time in sec since 01 Jan 2025 00:00:00"
    ext_hdu.header.comments["TTYPE2"] = "Background subtracted rate"
    ext_hdu.header.comments["TTYPE3"] = "Background subtracted rate error"

    ### create rest of extension header ###
    t_elapse = time_bine[-1]-time_bine[0]
    seqnum = evt_first_header['SEQPNUM']
    from . import __version__ as ver # Import the software version __init__.py file

    ext_keywords = [
        ("TELESCOP", "BLACKCAT",                                    "Telescope mission name"),
        ("INSTRUME", "BLACKCAT",                                    "Instrument name"),
        ("DATAMODE", "PHOTON",                                      "Data mode"),
        ("OBS_ID",   COPY,                                          "Observation ID"),
        ("OBS_MODE", COPY,                                          "Observation mode"),
        ("RA_OBJ",   obj_ra,                                        "[deg] Object R.A."),  
        ("DEC_OBJ",  obj_dec,                                        "[deg] Object Dec."), 
        ("RADECSYS", "FK5",                                         "Celestial coord system"),
        ("EQUINOX",  2000,                                          "Equinox of celestial coord system"),
        ("MJDREFI",  60676,                                         "MJD reference day 01 Jan 2025 00:00:00"),
        ("MJDREFF",  8.007407407407e-04,                            "MJD reference (fraction of day)"),
        ("TIMEREF",  "LOCAL",                                       "Reference Frame"),
        ("TASSIGN",  "SATELLITE",                                   "Time assigned"),
        ("TIMESYS",  "TT",                                          "Time system"),
        ("TIMEUNIT", "s",                                           "Time unit for timing header keywords"),
        ("TIMEPIXR", 1,                                             "Time stamps refer to end of each bin"),  
        ("TIMEDEL",  time_bine[1]-time_bine[0],                     "Time quantization"),  
        ("CLOCKAPP", False,                                         "If clock corrections are applied (T/F)"),
        ("DATE-OBS", COPY,                                          "Start Date UTC"),
        ("DATE-END", COPY,                                          "Stop Date UTC"),
        ("TSTART",   time_bine[0],                                  "Start time"),  
        ("TSTOP",    time_bine[-1],                                 "Stop time"),   
        ("DEADAPP",  False,                                         "If deadtime applied"),
        ("TELAPSE",  t_elapse,                                      "TSTOP-TSTART in sec"),  
        ("ONTIME",   evt_first_header.get('ONTIME', t_elapse),      "Sum good time interval in sec"),  #if there is no good time interval, set to T_ELAPSE
        ("TRIG_ID",  evt_first_header.get('TRIG_ID', None),         "Trigger id"), 
        ("TRIGGER",  evt_first_header.get('TRIGGER', None),         "Trigger number"), 
        ("TRIGTIME", evt_first_header.get('TRIGTIME', None),        "Trigger Time sec since 01 Jan 2025 00:00:00"),
        ("TRIGUTC",  evt_first_header.get('TRIGUTC', None),         "Time of Trigger Time in UTC"),
        ("E_MIN",    e_min,                                         "Lower energy boundary"), 
        ("E_MAX",    e_min,                                         "High energy boundary"), 
        ("E_UNIT",   "keV",                                         "Unit for the energy"),
        ("HDUCLASS", "OGIP",                                        "Format conforms to OGIP standard"),
        ("HDUCLAS1", "LIGHTCURVE",                                  "hduclass1"),
        ("CREATOR",  "BC_SCIENCE_ANALYSIS",                         "Software that created the file"),
        ("PROCVER",  ver+'.'+evt_first_header['CALDBVER'],          "Processing version"), #combine caldb & software version
        ("CALDBVER", COPY,                                          "CALDB version"),
        ("SEQPNUM",  t_elapse + 1 if reprocessing else t_elapse,    "Number of times the dataset has been processed"), #Tim needs to think #optional parameter (reprocessing  = True/False) if True take number and add 1 
        ("SOFTVER",  ver,                                           "Software version"),  #read init.py for value
        ("ORIGIN",   "GSFC",                                        "Origin of the FITS file"),
        ("DATE",     datetime.now().isoformat(timespec="seconds"),  "File creation date"), 
    ]


    for key, value, comment in ext_keywords:
        if value is COPY:
            value = evt_first_header[key]   # pull the value from the event file
        ext_hdu.header[key] = (value, comment)

    #Create extension HDU
    hdul_lc = fits.HDUList([prim_hdu, ext_hdu])
    if outfile is not None:
        hdul_lc.writeto(Path(outfile), overwrite=True, checksum=True)
    else:
        return hdul_lc