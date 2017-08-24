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
    def __init__(self, filelist=None):
        kw = {"dtype":np.dtype([("path", 'S200')])}
        self.tabs = np.genfromtxt(filelist, **kw)
        
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
            "right": None,
            "top": None,
            "wspace": None,
            "hspace": 0.03,
            }
        fig, ax = plt.subplots(figsize=(8,8), nrows=Ntabs, ncols=2, 
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
            vary = res["mean"] - 1. 
            # Time axis
            days4 = DayLocator(interval=4)
            days = DayLocator()
            dateFmt = DateFormatter("%b/%d/%y")
            months2 = MonthLocator()
            days2 = DayLocator()
            dateFmt2 = DateFormatter("%b/%d/%y")
            xdate = [datetime.datetime.strptime(str(date), "%Y%m%d")
                     for date in np.sort(res["nite_img"])]
            #
            ax[idx, 0].scatter(xdate, vary, s=10, 
                               c=res["nite_ref"], cmap="plasma", label=band)
            ax[idx, 0].xaxis.set_major_locator(days4)
            ax[idx, 0].xaxis.set_major_formatter(dateFmt)
            ax[idx, 0].xaxis.set_minor_locator(days)
            ax[idx, 0].autoscale_view()
            kw_grid0 = { 
                "color": "lightgray",
                "linestyle": "dotted",
                "dash_capstyle": "round",
                "alpha": 0.7,}
            ax[idx, 0].grid(**kw_grid0)
            xlabels_ax0 = ax[idx, 0].get_xticklabels()
            plt.setp(xlabels_ax0, rotation=30, fontsize=10)
            ndays = datetime.timedelta(days=2)
            ax[idx, 0].set_xlim([xdate[0] - ndays, xdate[-1] + ndays])
            #
            handles, labels = ax[idx, 0].get_legend_handles_labels()
            ax[idx, 0].legend(handles, labels,
                              loc="upper left", ncol=1, fontsize=10,
                              scatterpoints=1, markerscale=2,
                              frameon=False, framealpha=0.5)
            # Vertical histogram
            n, bins, patches = ax[idx, 1].hist(
                vary, bins="auto", normed=False, histtype="step", align="mid", 
                color="navy", orientation="horizontal"
                )
             
        
        
        plt.show()
        exit()
        # Read the file and load the listed files
        for tab in self.tabs["path"]:
            print tab
            aux_tab = np.load(tab)
            print aux_tab.shape

        # Next: plot in one box per band, in percentage y-axis


if (__name__ == "__main__"):
    print socket.gethostname()
    # Test run
    P = Plot(filelist="y4e2superc_y4e1superc.lst")
    P.all_bands()
    
    exit()
    #
    helptxt = "Plot the result tables coming from the comparison"
    aft = argparse.ArgumentParser(description=helptxt)
    #
    t1 = "File containing the list of .npy files to be plotted"
    aft.add_argument("input_list", help=t1)
    #
    t2 = "List of bands to be plotted (case-insensitive). Example: u, G, r"
    aft.add_argument("bands", help=t2, metavar="")
    # 
    val = aft.parse_args()
    kw = dict()
    # kw[""] = val.input_list

