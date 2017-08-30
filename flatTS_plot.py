""" Script to plot the results from the comparison between sets of weighted
images
"""

import os
import socket
import argparse
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import MonthLocator, DayLocator, DateFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class Plot():
    def __init__(self, filelist=None, plot_Nticks=None, outname=None):
        kw = {"dtype":np.dtype([("path", 'S200')])}
        self.tabs = np.genfromtxt(filelist, **kw)
        self.day_step = plot_Nticks
        self.outnm = outname

    def all_bands(self):
        """ Create plots for requested bands
        """
        # Setup the plot based in the number of bands 
        Ntabs = self.tabs.shape[0]
        # Using gridspec_kw
        plt.close("all")
        gspec_dict = {
            "height_ratios": None, 
            "width_ratios": [3, 1],
            "left": None,
            "bottom": None,
            "right": 0.915,
            "top": 0.94,
            "wspace": 0.07,
            "hspace": 0.09,
            }
        fig, ax = plt.subplots(figsize=(8,10), nrows=Ntabs, ncols=2, 
                               sharex="col", sharey="row", 
                               gridspec_kw=gspec_dict)
        for idx, filename in enumerate(self.tabs["path"]):
            res = np.load(filename, mmap_mode="r", allow_pickle=True)
            logging.info(res.dtype)
            # Plot by band
            # Time Series
            band = np.unique(res["band"])[0]
            if (len(np.unique(res["band"])) > 1):
                logging.error("Non-unique band for {0}".format(filename))
            # Change from statistics normalized value to values around 0
            varY = (res["mean"] - 1.) * 100.
            xdate = [datetime.datetime.strptime(str(date), "%Y%m%d")
                     for date in np.sort(res["nite_img"])]
            #
            im = ax[idx, 0].scatter(xdate, varY, s=10, 
                                    c=res["nite_ref"], cmap="summer", 
                                    label=band)
            # Time axis
            days4 = DayLocator(interval=self.day_step)
            days = DayLocator()
            dateFmt = DateFormatter("%b/%d/%y")
            months2 = MonthLocator()
            days2 = DayLocator()
            dateFmt2 = DateFormatter("%b/%d/%y")
            ax[idx, 0].xaxis.set_major_locator(days4)
            ax[idx, 0].xaxis.set_major_formatter(dateFmt)
            ax[idx, 0].xaxis.set_minor_locator(days)
            ax[idx, 0].autoscale_view()
            xlabels_ax0 = ax[idx, 0].get_xticklabels()
            plt.setp(xlabels_ax0, rotation=30, fontsize=10)
            ndays = datetime.timedelta(days=1)
            ax[idx, 0].set_xlim([xdate[0] - ndays, xdate[-1] + ndays])
            #
            handles, labels = ax[idx, 0].get_legend_handles_labels()
            ax[idx, 0].legend(handles, labels,
                              loc="upper left", ncol=1, fontsize=10,
                              scatterpoints=1, markerscale=2,
                              frameon=False, framealpha=0.5)
            #
            # Some percentiles
            for p in [25, 50, 75]:
                ax[idx, 1].axhline(np.percentile(varY, p), c="orange", 
                                   linestyle="solid", lw=0.8)
            for p in [5, 95]:
                ax[idx, 1].axhline(np.percentile(varY, p), c="green", 
                                   linestyle="solid", lw=0.8)
            # Vertical histogram
            n, bins, patches = ax[idx, 1].hist(
                varY, bins="auto", normed=False, histtype="step", align="mid", 
                color="navy", orientation="horizontal"
                )
            # Ticks every 1%
            # majorLocator_y1 = MultipleLocator(5) # (0.05)
            # majorFormatter_y1 = FormatStrFormatter("%d") # ("%.2f")
            minorLocator_y1 = MultipleLocator(1) # (0.01)
            # ax[idx, 0].yaxis.set_major_locator(majorLocator_y1)
            ax[idx, 0].yaxis.set_minor_locator(minorLocator_y1)
            # ax[idx, 0].yaxis.set_major_formatter(majorFormatter_y1)
            # Frequency (histo) axis
            # majorLocator_x1 = MultipleLocator(10)
            # ax[0, 0].xaxis.set_major_locator(majorLocator_x1)
            majorFormatter_x1 = FormatStrFormatter("%d")
            minorLocator_x1 = MultipleLocator(2)
            ax[idx, 1].xaxis.set_minor_locator(minorLocator_x1)
            ax[idx, 1].xaxis.set_major_formatter(majorFormatter_x1)
            # Grid and labels for both set of plots
            kw_grid0 = { 
                "color": "lightgray",
                "linestyle": "dotted",
                "dash_capstyle": "round",
                "which": "minor",
                "axis": "y",
                "alpha": 0.7,}
            ax[idx, 0].grid(**kw_grid0)
            ax[idx, 1].grid(**kw_grid0)
            ax[idx, 0].set_xlabel("Observation night", fontsize=10)
            ax[idx, 1].set_xlabel("N", fontsize=10)
            fig.text(0.04, 0.5, "Percentage of variability", 
                     va="center", rotation="vertical", fontsize=10) 
            #
            # Activate all spines for both sets of plots
            ax[idx, 0].spines["right"].set_visible(True)
            ax[idx, 1].spines["right"].set_visible(True)
            ax[idx, 0].spines["top"].set_visible(True)
            ax[idx, 1].spines["top"].set_visible(True)
            # Set ticks of histogram to the right
            ax[idx, 1].yaxis.set_ticks_position("right")
            #
        # Suptitle 
        gtitle = "Time Series from comparison of domeflats (cb:green->yellow)"
        fig.suptitle(gtitle, fontsize=13)
        # Display and save
        plt.savefig(self.outnm, dpi=400, facecolor="w", edgecolor="w",
                    orientation="portrait", papertype=None, format="pdf",
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
        plt.show()


if (__name__ == "__main__"):
    print socket.gethostname()
    # Test run
    # P = Plot(filelist="y4e2superc_y4e1superc.lst", plot_Nticks=3, 
    #          outname="y4e2superc_y4e1superc.pdf")
    # P.all_bands()
    
    #
    helptxt = "Plot the result tables coming from the comparison"
    aft = argparse.ArgumentParser(description=helptxt)
    #
    t1 = "File containing the list of .npy files to be plotted"
    aft.add_argument("input_list", help=t1)
    #
    t2 = "Out filename for the .pdf plot"
    aft.add_argument("out", help=t2)
    #
    t3 = "Space in days between the ticks to be plotted in the time series."
    t3 += " Default: 3"
    aft.add_argument("--days", "-d", help=t3, metavar="", default=3, type=int)
    # 
    val = aft.parse_args()
    kw = dict()
    kw["filelist"] = val.input_list
    kw["plot_Nticks"] = val.days
    kw["outname"] = val.out
    P = Plot(**kw)
    P.all_bands()
