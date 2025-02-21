import os
import sys
import glob
import datetime
from collections import OrderedDict
from tqdm import tqdm
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
mpl.use('Agg')


from astropy.io import fits
import er3t
import cam

def read_fits(
        fname,
        iselect=0,
        plot=False,
        verbose=True,
        ):

    info_list = ['#╭────────────────────────────────────────────────╮#', fname]
    with fits.open(fname) as f:
        f0 = f[iselect]
        for key in f0.header.keys():
            info0 = '%d, %s, %s' % (iselect, key, f0.header[key])
            info_list.append(info0)

        data0 = f0.data.copy()
    info_list.append('#╰────────────────────────────────────────────────╯#')

    if verbose:
        info = '\n'.join(info_list)
        print(info)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(data0.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.text(100, 1000, info, fontsize=10)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_%s_%d.png' % (_metadata_['Function'], info_list[0].replace('/', '-').replace('.fits', ''), i), bbox_inches='tight', metadata=_metadata_)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return data0

def cdata_led(
        fnames,
        int_time,
        ):

    light0 = read_fits(fnames[0], verbose=True)

    N = len(fnames)
    shape = light0.shape + (N,)
    light = np.zeros(shape, dtype=light0.dtype)
    ifile = np.zeros(N, dtype=np.int16)
    for i, fname in tqdm(enumerate(fnames)):
        filename = os.path.basename(fname)
        ifname = filename.split('_')[2]
        light[..., i] = read_fits(fname, verbose=False)
        ifile[i] = ifname

    indices_sort = np.argsort(ifile)
    ifile = ifile[indices_sort]
    light = light[..., indices_sort]

    f = h5py.File('led_%dms.h5' % int_time, 'w')
    dset_light = f.create_dataset('light', data=light, compression='gzip', compression_opts=9, chunks=True)
    dset_ifile = f.create_dataset('ifile', data=ifile, compression='gzip', compression_opts=9, chunks=True)
    f.close()

def check_led_1ms():
    fname = 'data/aux/dark_1ms.h5'
    f = h5py.File(fname, 'r')
    dark = np.nanmean(f['dark'][...], axis=-1)
    f.close()

    fname = 'data/aux/led_1ms.h5'
    f = h5py.File(fname, 'r')
    data = f['light'][...]-dark[..., np.newaxis]
    ifile = f['ifile'][...]
    f.close()

    vmin = 000
    vmax = 200
    rcParams['font.size'] = 14

    for i in tqdm(range(data.shape[-1])):
        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(14, 2.5))
            fig.suptitle('LED (1ms)', y=1.05)
            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(141)
            cs = ax1.imshow(data[..., i].T, origin='lower', cmap='jet', zorder=0, vmin=vmin, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
            ax1.axis('off')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax1.set_title('Light at File #%d' % ifile[i], color='r')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax2 = fig.add_subplot(142)
            mean = np.nanmean(data[..., :i+1], axis=-1)
            cs = ax2.imshow(mean.T, origin='lower', cmap='jet', zorder=0, vmin=vmin, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
            ax2.axis('off')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax2.set_title('Mean until File #%d' % ifile[i], color='k')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax3 = fig.add_subplot(143)
            std = np.nanstd(data[..., :i+1], axis=-1)
            cs = ax3.imshow(std.T, origin='lower', cmap='jet', zorder=0, vmin=0, vmax=25) #, extent=extent, vmin=0.0, vmax=0.5)
            ax3.axis('off')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax3.set_title('STD until File #%d' % ifile[i], color='blue')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax4 = fig.add_subplot(144)
            ax4.plot(data[:, 1024, i], color='r', lw=0.5) #, extent=extent, vmin=0.0, vmax=0.5)
            ax4.plot(mean[:, 1024], color='k', lw=0.5) #, extent=extent, vmin=0.0, vmax=0.5)
            ax4_ = ax4.twinx()
            ax4_.plot(std[:, 1024], color='b', lw=0.5)
            ax4.set_ylim((vmin, vmax))
            ax4.set_xlim((0, 2047))
            ax4_.set_ylim((0, 25))
            #╰──────────────────────────────────────────────────────────────╯#

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            plt.savefig('%s_%3.3d.png' % (_metadata_['Function'], ifile[i]), bbox_inches='tight', metadata=_metadata_)
            #╰──────────────────────────────────────────────────────────────╯#
            # plt.show()
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

def check_led_20ms():

    fname = 'data/aux/led_20ms.h5'
    f = h5py.File(fname, 'r')
    data = f['light'][...]
    ifile = f['ifile'][...]
    f.close()
    vmin = 500
    vmax = 3500
    rcParams['font.size'] = 14

    for i in tqdm(range(data.shape[-1])):
        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(14, 2.5))
            fig.suptitle('LED (20ms)', y=1.05)
            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(141)
            cs = ax1.imshow(data[..., i].T, origin='lower', cmap='jet', zorder=0, vmin=vmin, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
            ax1.axis('off')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax1.set_title('Light at File #%d' % ifile[i], color='r')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax2 = fig.add_subplot(142)
            mean = np.nanmean(data[..., :i+1], axis=-1)
            cs = ax2.imshow(mean.T, origin='lower', cmap='jet', zorder=0, vmin=vmin, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
            ax2.axis('off')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax2.set_title('Mean until File #%d' % ifile[i], color='k')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax3 = fig.add_subplot(143)
            std = np.nanstd(data[..., :i+1], axis=-1)
            cs = ax3.imshow(std.T, origin='lower', cmap='jet', zorder=0, vmin=0, vmax=100) #, extent=extent, vmin=0.0, vmax=0.5)
            ax3.axis('off')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax3.set_title('STD until File #%d' % ifile[i], color='blue')
            #╰──────────────────────────────────────────────────────────────╯#

            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax4 = fig.add_subplot(144)
            ax4.plot(data[:, 1024, i], color='r', lw=0.5) #, extent=extent, vmin=0.0, vmax=0.5)
            ax4.plot(mean[:, 1024], color='k', lw=0.5) #, extent=extent, vmin=0.0, vmax=0.5)
            ax4_ = ax4.twinx()
            ax4_.plot(std[:, 1024], color='b', lw=0.5)
            ax4.set_ylim((vmin, vmax))
            ax4.set_xlim((0, 2047))
            ax4_.set_ylim((0, 100))
            #╰──────────────────────────────────────────────────────────────╯#

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            plt.savefig('%s_%3.3d.png' % (_metadata_['Function'], ifile[i]), bbox_inches='tight', metadata=_metadata_)
            #╰──────────────────────────────────────────────────────────────╯#
            # plt.show()
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

if __name__ == '__main__':

    check_led_1ms()
    # check_led_20ms()

    # fdir = 'data/aux/led'
    # # int_time = 1
    # int_time = 20
    # fnames = sorted(glob.glob('%s/g??_%d_*.fits' % (fdir, int_time)))
    # cdata_led(fnames, int_time)

    pass
