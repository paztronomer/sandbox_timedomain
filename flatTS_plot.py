""" Script to plot the results from the comparison between sets of weighted
images
"""

import os
import socket
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
                               sharex="col", sharey=False, 
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
            ax[idx, 0].scatter(res["nite_img"], vary, s=10, 
                               c=res["nite_ref"], label=band)
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

