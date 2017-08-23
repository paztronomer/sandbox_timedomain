""" Code to create a set of time series from binned focal plane images
of filetype compare_binned_fp
"""
import os
import errno
import socket
import argparse
import logging
import numpy as np
import pandas as pd
import fitsio


class FPBinned():
    def __init__(self, fullpath, dim_read=0):
        """ Simple method to open focal plane binned images
        """
        M_header = fitsio.read_header(fullpath)
        M_hdu = fitsio.FITS(fullpath)[dim_read]
        self.data = M_hdu.read()
        self.header = M_header


class Construct():
    def __init__(self, bench=None, prefix_weight=None, 
                 suffix_weight=None, dir_weighted=None,
                 norm= None,
                 cols=["pfw_attempt_id","reqnum","unitname","attnum",
                       "expnum","band","nite","filename","path"]):
        """ Initialize the constructor
        Inputs
        - bench: str, filename of the table containing the paths plus some
        additional info of the images to be used as benchmark
        - prefix_weight: str, prefix to be used on each of the weighted images
        filenames
        - cols: list of str, columns to be used when reading the DF
        - suffix_weight: str, suffix for the images filename
        - dir_weighted: str, directory where to save the weighted images
        """
        try:
            d = {"sep":"\s+", "engine":"python", "names":cols}
            self.df_ben = pd.read_table(bench, **d)
        except Exception as e:
            raise
            print "Error loading tables"
            exit(1)
        if (suffix_weight is None):
            self.suffix = "pid{0}".format(os.getpid())
        else:
            self.suffix = suffix_weight
        self.prefix = prefix_weight
        self.dir_wimg = dir_weighted
        self.norm = norm

    def weight_image(self, rootpath="/archive_data/desarchive", 
                     stat=np.median,
                     dir_wimg=None):
        """ Create weight images by some statistics of the distribution, pixel
        by pixel, and save them
        Inputs
        - rootpath: str, root path to the OPS, ACT, DTS directory
        - stat: function, norm to be used for create a weighted image based
        on pixel statistics
        """
        band = self.df_ben.drop_duplicates(subset=["band"], inplace=False)
        band = band["band"]
        nite = self.df_ben.drop_duplicates(subset=["nite"], inplace=False)
        nite = nite["nite"]
        # Iterate over band for the entrire available data range, creating
        # a weighted image per night
        for b in band:
            for idx,n in enumerate(nite):
                print b, n
                auxsel = self.df_ben[(self.df_ben["band"]==b) & 
                                     (self.df_ben["nite"]==n)]
                auxsel.reset_index(drop=True, inplace=True)
                # Check if there are no duplicate entries for the same 
                # exposure 
                c1 = np.unique(auxsel["expnum"]).shape[0] == len( auxsel.index)
                if (not c1):
                    logging.error("Multiple entries for some exposures")
                    exit(1)
                # Now create the median of the night, per band
                for index, row in auxsel.iterrows():
                    pth = row["path"]
                    fnm = row["filename"]
                    if index == 0:
                        if (rootpath is not None):
                            auxfnm = os.path.join(rootpath, pth)
                            auxfnm = os.path.join(auxfnm, fnm)
                        else:
                            auxfnm = os.path.join(pth, fnm)
                        M = FPBinned(auxfnm).data
                        if (self.norm == 0):
                            M /= np.median(M)
                        elif (self.norm == 1):
                            M /= np.mean(M)
                        #============================
                        #for compare_dflat_binned_fp
                        #must remove the median norm
                        # M /= np.median(M)
                        #============================
                    else:
                        if (rootpath is not None):
                            auxfnm = os.path.join(rootpath, pth)
                            auxfnm = os.path.join(auxfnm, fnm)
                        else:
                            auxfnm = os.path.join(pth, fnm)
                        N = FPBinned(auxfnm).data
                        if (self.norm == 0):
                            N /= np.median(N)
                        elif (self.norm == 1):
                            N /= np.mean(N)
                        #============================
                        #for compare_dflat_binned_fp
                        #must remove the median norm
                        #============================
                        M = np.dstack((M,N))
                # Call the weighted images method
                M_w = self.stat_cube(M, (lambda: stat)())
                # Check/crete the destination folder
                if (self.dir_wimg is None):
                    self.dir_wimg = os.path.join(os.getcwd(), "weighted/") 
                try:
                    os.makedirs(self.dir_wimg)
                except OSError as exception:
                    if (exception.errno != errno.EEXIST):
                        raise
                        logging.error("{0} cannot be created".format(dir_img))
                # Name the output weighted images
                if (self.prefix is None):
                    fnm = "wimg_{1}_{2}_{3}.fits".format(row["nite"], 
                                                         row["band"], 
                                                         self.suffix)
                else:
                    fnm = "{0}_wimg_{1}_{2}_{3}.fits".format(self.prefix, 
                                                             row["nite"], 
                                                             row["band"], 
                                                             self.suffix)
                outpath = os.path.join(self.dir_wimg, fnm)
                logging.info("Saving weighted image: {0}".format(outpath))
                print "Saving weighted image: {0}".format(outpath)
                #
                # Here save information on the header! Example: range of
                # nights, set of expnums used for generate it, band, 
                # pfw_attempt for the processed images, etc
                #
                fits = fitsio.FITS(outpath, "rw")
                fits.write(M_w)
                fits[-1].write_checksum()
                fits.close()
        return True

    def stat_cube(self,arr3,f):
        '''Receives a data cube (3D array) and performs the given statistics
        over the third dimension, pixel by pixel
        Uses numpy iteration tools
        '''
        out = np.zeros_like(arr3[:, :, 0])
        it = np.nditer(arr3[:, :, 0], flags=['multi_index'])
        while not it.finished:
            i1, i2 = it.multi_index
            out[i1, i2] = f(arr3[i1, i2, :])
            it.iternext()
        return out

if __name__ == "__main__":
    #
    print "Running on {0}".format(socket.gethostname())
    #
    h = "Code to create weighted images, per night, per band."
    bow = argparse.ArgumentParser(description=h)
    # Positional
    #
    txt2 = "Full path to the tables from DB containing information of the"
    txt2 += " binned_fp to be used as benchmark"
    bow.add_argument("benchmark", help=txt2)
    #
    # Optional
    txt3 = "Prefix for the benchmark set weighted outputimages. If no value,"
    txt3 += " images will start with 'wimg'"
    bow.add_argument("--prefix", help=txt3, metavar="")
    #
    txt4 = "Suffix to be used in naming the output weighted images from the"
    txt4 += " benchmark dataset. If no value, PID will be used"
    bow.add_argument("--suffix", help=txt4, metavar="")
    #
    txt5 = "Folder where to save the weighted images."
    txt5 += " Default: <current_directory>/weighted"
    bow.add_argument("--dir_w", help=txt5, metavar="")
    #
    txt6 = "Flag to normalize each image before combine. If no flag, images"
    txt6 += " aren't normalized. Values: 0-median, 1-mean"
    bow.add_argument("--norm", help=txt6, metavar="", type=int, choices=[0, 1])
    #
    val = bow.parse_args()
    kw = dict()
    kw["bench"] = val.benchmark
    kw["prefix_weight"] = val.prefix
    kw["suffix_weight"] = val.suffix
    kw["dir_weighted"] = val.dir_w
    kw["norm"] = val.norm
    #
    C = Construct(**kw)
    C.weight_image()
