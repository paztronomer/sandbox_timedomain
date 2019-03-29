""" Script to iteratively plot a set of binned focal plane images
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from astropy import time
from astropy.io import fits
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, ZScaleInterval)


def animate(list_fnm, df_info, out_anim='anim_set02'):
    anim_interval = 4000
    # 4000
    anim_repeat = False
    anim_save = True
    anim_blit = True
    fig, ax = plt.subplots()
    ims = []
    # Populate a list 
    for idx, i in enumerate(list_fnm):
        tmp = fits.open(i)
        aux_x = tmp[0].data
        expnum = tmp[0].header['EXPNUM']
        im_norm = ImageNormalize(
            aux_x,
            interval=ZScaleInterval(),
            stretch=SqrtStretch(),
        )
        kw = {
            'origin': 'lower',
            'animated': True,
            'cmap': 'bone',
            'norm': im_norm,
        }
        # Plotting
        im = ax.imshow(aux_x, label=idx, **kw)
        # Additional information to be shown
        df = df_info.loc[df_info['expnum'] == expnum]
        print(df)
        t = '{0},'.format(df['nite'].iloc[0]) 
        t += ' enum={0},'.format(df['expnum'].iloc[0])
        t += '\n etime={0:.0f},'.format(df['exptime'].iloc[0])
        t += ' N={0:.0f}'.format(df['nobjects'].iloc[0])
        txt_aux = ax.text(0.95, 2, t,
                          fontweight='bold', color='w', fontsize=12)
        # Append to the list to be converted into frames
        ims.append([im, txt_aux])
    # With the list of objects create the animation
    anima = animation.ArtistAnimation(
        fig, ims, interval=anim_interval, blit=anim_blit, 
        repeat_delay=anim_interval, repeat=anim_repeat,
    )
    plt.tight_layout()
    plt.legend()
    # anima.save(out_anim)
    plt.show()
    return True

if __name__ == '__main__':
    fnm = 'r4124_binned_path.csv'
    fnm2 = 'r4124_nobjects.csv'
    lst = np.loadtxt(fnm, dtype='str')
    info = pd.read_csv(fnm2)
    info.columns = info.columns.map(str.lower)
    animate(lst, info)
