import datetime
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# mpl.use('Agg')
from astropy.io import fits
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


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

def cdata_darks(fnames, int_time):

    dark0 = read_fits(fnames[0], verbose=True)

    N = len(fnames)
    shape = dark0.shape + (N,)
    dark = np.zeros(shape, dtype=dark0.dtype)
    ifile = np.zeros(N, dtype=np.int16)
    for i, fname in tqdm(enumerate(fnames)):
        filename = os.path.basename(fname)
        ifname = filename.split('_')[2]
        dark[..., i] = read_fits(fname, verbose=False)
        ifile[i] = ifname

    indices_sort = np.argsort(ifile)
    ifile = ifile[indices_sort]
    dark = dark[..., indices_sort]

    f = h5py.File('dark_%dms.h5' % int_time, 'w')
    dset_dark = f.create_dataset('dark', data=dark, compression='gzip', compression_opts=9, chunks=True)
    dset_ifile = f.create_dataset('ifile', data=ifile, compression='gzip', compression_opts=9, chunks=True)
    f.close()

def check_dark_1ms():

    fname = 'data/aux/dark_1ms.h5'
    f = h5py.File(fname, 'r')
    data = f['dark'][...]
    ifile = f['ifile'][...]
    f.close()
    vmin = 200
    vmax = 250
    rcParams['font.size'] = 14

    for i in tqdm(range(data.shape[-1])):
        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(14, 2.5))
            fig.suptitle('Darks (1ms)', y=1.05)
            # plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(141)
            cs = ax1.imshow(data[..., i].T, origin='lower', cmap='jet', zorder=0, vmin=vmin, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
            ax1.axis('off')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            ax1.set_title('Dark at File #%d' % ifile[i], color='r')
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

def check_dark_kss():

    # check data for KSS
    #╭────────────────────────────────────────────────────────────────────────────╮#
    darks = read_fits('data/20250116/dark_1_0_0.fits')
    lights = read_fits('data/20250116/room_post_focus_5_0_0.fits')

    logic = (lights<=500.0)
    scale_factor = lights[logic].mean()/darks[logic].mean()
    darks = darks * scale_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('Cable Schematics (scaled)')

        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131)
        # cs = ax1.imshow(darks.T, origin='upper', cmap='jet', zorder=0, vmin=150, vmax=250)
        cs = ax1.imshow(darks.T, origin='upper', cmap='jet', zorder=0, vmin=450, vmax=650)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)

        ax1.set_title('A. dark_1_0_0.fits')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(132)
        cs = ax2.imshow(lights.T, origin='upper', cmap='Greys_r', zorder=0, vmin=0, vmax=6000)
        # cs = ax2.imshow(lights.T, origin='upper', cmap='jet', zorder=0, vmin=450, vmax=750)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)

        ax2.set_title('B. room_post_focus_5_0_0.fits')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(133)
        cs = ax3.imshow((lights-darks).T, origin='upper', cmap='Greys_r', zorder=0, vmin=0, vmax=6000)
        # cs = ax3.imshow((lights-darks).T, origin='upper', cmap='jet', zorder=0, vmin=450, vmax=750)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)

        ax3.set_title('B - A')
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s_%s.png' % (_metadata_['Date'], _metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    pass

if __name__ == '__main__':

    data = read_fits('data/20250116/dark_1_0_0.fits')
