""" Script takes 2 lists of images and compare each one in the sample list
against the images in the benchmark list
"""

import os
import errno
import socket
import time
import argparse
import copy
import logging
import uuid
import pandas as pd
import numpy as np
import scipy.stats
import fitsio
# For print in screen
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# For save to file
if False:
    logging.basicConfig(filename= str(uuid.uuid4()) + ".log",
                        level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
# For print to file and save to file at same time
if False:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('log_filename.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# PENDING:
# 1) if compare sets of images not being weighted images, get the
# flag from the header


class Toolbox():
    def split_path(self, path):
        """ Method to return parent path and the filename
        Inputs
        - path: complete path as in full/path/to/file.extension
        Returns 
        - tuple with 2 elements being (path, filename) 
        """
        relroot, filename = os.path.split(path)
        return (relroot, filename)


class LoadFITS():
    def __init__(self, fullpath, dim_read=0):
        """ Simple method to open focal plane binned images
        """
        M_header = fitsio.read_header(fullpath)
        M_hdu = fitsio.FITS(fullpath)[dim_read]
        # Note here we're copying a ndarray, and an object
        self.data = np.ndarray.copy(M_hdu.read())
        self.header = copy.deepcopy(M_header)
        fitsio.FITS(fullpath).close()


class Compare():
    def __init__(self, images=None, reference=None, norm=None, outfile=None,
                 combine=None):
        kw = {"dtype":np.dtype([('fullpath', 'S200')])}
        self.im = np.genfromtxt(images, **kw)
        self.ref = np.genfromtxt(reference, **kw)
        self.norm = norm
        self.outnm = outfile
        self.combine = combine

    def compare_wimg(self):
        """ Iteratively compare a set of weighted images to the set of 
        (also weighted) benchmarks 
        """
        # Compare images with references
        # ratio, bad = [], []
        res = []
        T = Toolbox()
        # In case of 1-element
        if (self.im.ndim == 0):
            im_path_aux = list([str(self.im["fullpath"])])
        else:
            im_path_aux = self.im["fullpath"]
        if (self.ref.ndim == 0):
            ref_path_aux = list([str(self.ref["fullpath"])])
        else:
            ref_path_aux = self.ref["fullpath"]
        #

        #
        # The outer region has values of -1. So I has some options... do not
        # use negative values for data_cube or to use masked array
        #
        #
        for im in im_path_aux:
            # Folder and filename
            folder_im, fnm_im = T.split_path(im)
            # Load data
            fp_bin = LoadFITS(im)
            xdata = fp_bin.data
            xdata_h = fp_bin.header
            del fp_bin
            # Normalize each one of the images by its own statistics
            if (self.norm == 0):
                xdata /= np.median(xdata)
                flag_norm = "median"
            elif (self.norm == 1):
                xdata /= np.mean(xdata)
                flag_norm = "mean"
            else:
                flag_norm = "none"
            for idx, ref in enumerate(ref_path_aux):
                # Folder and filename for reference
                folder_ref, fnm_ref = T.split_path(ref)
                if (im == ref):
                    continue
                # Reuse fpb
                fp_bin = LoadFITS(ref)
                xref = fp_bin.data
                if (self.norm == 0):
                    xref /= np.median(xref)
                elif (self.norm == 1):
                    xref /= np.mean(xref)
                xref_h = fp_bin.header
                del fp_bin 
                # If combine flag was set, then get a median image per set of 
                # compared images
                if self.combine:
                    if (idx == 0):
                        X = xdata / xref
                    else:
                        X_tmp = xdata / xref
                        X = np.dstack((X, X_tmp))
                s = self.get_stat(xdata, xref)
                # Include description of compared data
                # IMG, REF, NORM 
                flag_im = fnm_im.split("_")[0]
                flag_ref = fnm_ref.split("_")[0]
                nite1 = int(fnm_im.split("_")[2])
                nite2 = int(fnm_ref.split("_")[2])
                band = fnm_im.split("_")[3]
                comm = fnm_im.split("_")[1] + "-" + fnm_im.split("_")[4][:-5]
                i = [flag_im, flag_ref, flag_norm, nite1, nite2, band, comm]
                s = i + s
                s = tuple(s)
                res.append(s)
                del s
                # ratio.append(s[0])
                # if ((s[0] > 1.05) or (s[0] < 0.95)):
                #    bad.append(s[0])
            if self.combine:
                # Create the combined image per set of comparisons
                stat2use = np.median
                Xres = self.stat_cube(X, (lambda: stat2use)())
                # And save as FITS
                self.dir_combine = os.path.join(os.getcwd(), "combined/") 
                try:
                    os.makedirs(self.dir_combine)
                except OSError as exception:
                    if (exception.errno != errno.EEXIST):
                        raise
                        logging.error("Error creating combined folder")
                relroot_im, fname_im = os.path.split(im)
                relroot_ref, fname_ref = os.path.split(ref)
                # Assume the filenames have the same format:
                # <some_prefix>_wimg_<nite>_<band>_pixcorMedN.fits
                m = "Assuming filenames as"
                m += " <some_prefix>_wimg_<nite>_<band>_pixcorMedN.fits"
                logging.warning(m)
                a1 = fname_im[: fname_im.find("_")]
                b1 = fname_ref[: fname_ref.find("_")]
                p1 = fname_im.find("wimg_") + 5
                p2 = fname_im.find("_pixcor")
                a2_nite_band = fname_im[p1 : p2] 
                outfnm = a1 + "_" + b1 + "_" + a2_nite_band + "_combined.fits"
                outfnm = os.path.join(self.dir_combine, outfnm)
                fits_comb = fitsio.FITS(outfnm, "rw")
                fits_comb.write(Xres)
                fits_comb[-1].write_checksum()
                fits_comb.close()
        # bad = np.array(bad)
        # ratio = np.array(ratio)
        # remove nan
        # bad = bad[~np.isnan(bad)]
        # ratio = ratio[~np.isnan(ratio)]
        # print len(bad), np.mean(bad)
        # print len(ratio), np.mean(ratio)
        #
        # Saving the stats, needs the output filename
        dt = np.dtype([("flag_img", "|S100"), ("flag_ref", "|S100"),
                       ("norm_flag", "|S10"), ("nite_img", "i8"),
                       ("nite_ref", "i8"), ("band", "|S5"), 
                       ("comments", "|S50"), ("mean", "f8"), 
                       ("median", "f8"), ("range", "f8"), ("rms", "f8"), 
                       ("mad", "f8"), ("kurt", "f8"), ("skew", "f8"),
                       ("entropy", "f8")])
        recarr = np.rec.array(res, dtype=dt)
        np.save(self.outnm, recarr)
        msg = "Saved: {0}".format(self.outnm)
        logging.info(msg)
        return True

    def get_stat(self, data, reference):
        """ Method that make the statistics of the comparison
        """
        tmp = data/reference
        # Simple statistics
        # <x>, med, ptp, rms, mad, kurt, skew, entropy
        s1 = np.mean(tmp)
        s2 = np.median(tmp)
        s3 = np.ptp(tmp.ravel())
        s4 = np.sqrt(np.mean(np.square(tmp.ravel())))
        s5 = np.median(np.abs(tmp - np.median(tmp)))
        s6 = scipy.stats.skew(tmp.ravel())
        s7 = scipy.stats.kurtosis(tmp.ravel(), fisher=False, bias=True)
        s8 = scipy.stats.entropy(tmp.ravel())
        return [s1, s2, s3, s4, s5, s6, s7, s8]

    def stat_cube(self, arr3, f):
        """ FROM: flatTS_weighted.py
        Receives a data cube (3D array) and performs the given statistics
        over the third dimension, pixel by pixel
        Uses numpy iteration tools
        """
        out = np.zeros_like(arr3[:, :, 0])
        it = np.nditer(arr3[:, :, 0], flags=["multi_index"])
        while not it.finished:
            i1, i2 = it.multi_index
            out[i1, i2] = f(arr3[i1, i2, :])
            it.iternext()
        return out


if __name__ == "__main__":
    print socket.gethostname()
    #
    h = "Code to compare a set of flats against other sample and get"
    h += " statistics"
    aft = argparse.ArgumentParser(description=h)
    #
    h1 = "Filename of the list of images to compare (full path if not cwd)"
    aft.add_argument("imlist", help=h1)
    # 
    h2 = "Filename of the list of images against to compare, benchmark"
    aft.add_argument("imref", help=h2)
    #
    h3 = "Filename (or full path) to the output .npy file"
    aft.add_argument("out", help=h3)
    #
    h4 = "Normalize each image? (0-median, 1-mean)."
    h4 += " Default is not normalization"
    aft.add_argument("--norm", help=h4, metavar="")
    #
    h5 = "Wheter to create a median image for each set of comparison"
    h5 += " \'image vs reference\'."
    aft.add_argument("--combine", help=h5, action="store_true")
    #
    val = aft.parse_args()
    kw = dict()
    kw["images"] = val.imlist
    kw["reference"] = val.imref
    kw["outfile"] = val.out
    kw["norm"] = val.norm
    kw["combine"] = val.combine
    #
    C = Compare(**kw)
    #
    C.compare_wimg()
