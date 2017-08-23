""" Script to plot the results from the comparison between sets of weighted
images
"""

import os
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Plot():
    def __init__(self, filelist=None):
        kw = {"dtype":np.dtype([("path", 'S200')])}
        self.tabs = np.genfromtxt(filelist, **kw)
        
    def all_bands(self):
        """ Create plots for requested bands
        """
        # Setup the plot based in the number of bands 

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

