"""Functions used by multiple files in bc_science_analysis."""

from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime


from science_analysis.bc_image_analyzer import BCImageAnalysis


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
    evt_prim_header = fits.getheader(event_file, 0)
    evt_first_data, evt_first_header = fits.getdata(event_file, 1, header=True)



    ### Create primary Header ###
    prim_header = evt_prim_header #Copy over most values from the event file primary header
    prim_header['DATE'] = datetime.now().isoformat(timespec="seconds")
    prim_header['CREATOR'] = "BC_SCIENCE_ANALYSIS"

    ### Create primary HDU ###
    prim_hdu = fits.PrimaryHDU(header=prim_header)

    ### Create extension data ###
    ext_data = fits.ColDefs([
        fits.Column(name="TIME",  format="1D", unit="s",       array=time_bine[1:]),
        fits.Column(name="RATE",  format="1E", unit="count/s", array=lc),
        fits.Column(name="ERROR", format="1E", unit="count/s", array=lc_err),
    ])
    ext_hdu = fits.BinTableHDU.from_columns(ext_data, name="LIGHTCURVE")

    ext_hdu.header.comments["TTYPE1"] = "Mission time in sec since 01 Jan 2025 00:00:00"
    ext_hdu.header.comments["TTYPE2"] = "Background subtracted rate"
    ext_hdu.header.comments["TTYPE3"] = "Background subtracted rate error"

    ### create rest of extension header ###
    from . import __version__ as ver # Import the software version __init__.py file

    ext_hdu.header["TELESCOP"] = evt_first_header.cards["TELESCOP"]
    ext_hdu.header["INSTRUME"] = evt_first_header.cards["INSTRUME"]
    ext_hdu.header["DATAMODE"] = evt_first_header.cards["DATAMODE"]
    ext_hdu.header["OBS_ID"]   = evt_first_header.cards["OBS_ID"]
    ext_hdu.header["OBS_MODE"] = evt_first_header.cards["OBS_MODE"]
    ext_hdu.header["RA_OBJ"]   = (obj_ra, "[deg] Object R.A.")
    ext_hdu.header["DEC_OBJ"]  = (obj_dec, "[deg] Object Dec.")
    ext_hdu.header["RADECSYS"] = evt_first_header.cards["RADECSYS"]
    ext_hdu.header["EQUINOX"]  = evt_first_header.cards["EQUINOX"]
    ext_hdu.header["MJDREFI"]  = evt_first_header.cards["MJDREFI"]
    ext_hdu.header["MJDREFF"]  = evt_first_header.cards["MJDREFF"]
    ext_hdu.header["TIMEREF"]  = evt_first_header.cards["TIMEREF"]
    ext_hdu.header["TASSIGN"]  = evt_first_header.cards["TASSIGN"]
    ext_hdu.header["TIMESYS"]  = evt_first_header.cards["TIMESYS"]
    ext_hdu.header["TIMEUNIT"] = evt_first_header.cards["TIMEUNIT"]
    ext_hdu.header["TIMEPIXR"] = evt_first_header.cards["TIMEPIXR"]
    ext_hdu.header["TIMEDEL"]  = (time_bine[1] - time_bine[0], "Time quantization")
    ext_hdu.header["CLOCKAPP"] = evt_first_header.cards["CLOCKAPP"]
    ext_hdu.header["DATE-OBS"] = evt_first_header.cards["DATE-OBS"]
    ext_hdu.header["DATE-END"] = evt_first_header.cards["DATE-END"]
    ext_hdu.header["TSTART"]   = (time_bine[0], "Start time")
    ext_hdu.header["TSTOP"]    = (time_bine[-1], "Stop time")
    ext_hdu.header["DEADAPP"]  = evt_first_header.cards["DEADAPP"]
    ext_hdu.header["TELAPSE"]  = (time_bine[-1] - time_bine[0], "TSTOP-TSTART in sec")
    ext_hdu.header["ONTIME"]   = (evt_first_header.get('ONTIME', time_bine[-1] - time_bine[0]), "Sum good time interval in sec")
    ext_hdu.header["TRIG_ID"]  = (evt_first_header.get('TRIG_ID', None), "Trigger id")
    ext_hdu.header["TRIGGER"]  = (evt_first_header.get('TRIGGER', None), "Trigger number")
    ext_hdu.header["TRIGTIME"] = (evt_first_header.get('TRIGTIME', None), "Trigger Time sec since 01 Jan 2025 00:00:00")
    ext_hdu.header["TRIGUTC"]  = (evt_first_header.get('TRIGUTC', None), "Time of Trigger Time in UTC")
    ext_hdu.header["E_MIN"]    = (e_min, "Lower energy boundary")
    ext_hdu.header["E_MAX"]    = (e_max, "High energy boundary")
    ext_hdu.header["E_UNIT"]   = ("keV", "Unit for the energy")
    ext_hdu.header["HDUCLASS"] = evt_first_header.cards["HDUCLASS"]
    ext_hdu.header["HDUCLAS1"] = evt_first_header.cards["HDUCLAS1"]
    ext_hdu.header["HDUCLAS1"] = "LIGHTCURVE"
    ext_hdu.header["CREATOR"]  = prim_header.cards["CREATOR"]
    ext_hdu.header["PROCVER"]  = (ver + '.' + evt_first_header['CALDBVER'], "Processing version")
    ext_hdu.header["CALDBVER"] = evt_first_header.cards["CALDBVER"]
    ext_hdu.header["SEQPNUM"]  = (time_bine[-1] - time_bine[0] + 1 if reprocessing else time_bine[-1] - time_bine[0], "Number of times the dataset has been processed")
    ext_hdu.header["SOFTVER"]  = (ver, "Software version")
    ext_hdu.header["ORIGIN"]   = ("GSFC", "Origin of the FITS file")
    ext_hdu.header["DATE"]     = (datetime.now().isoformat(timespec="seconds"), "File creation date")
    

    #Create extension HDU
    hdul_lc = fits.HDUList([prim_hdu, ext_hdu])
    if outfile is not None:
        hdul_lc.writeto(Path(outfile), overwrite=True, checksum=True)
    
    return hdul_lc