import os
import sys
import glob
import datetime
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


import cam


def cam_dn2rad(cnt_obs, int_time):

    # dark correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dark_offset = cam.corr.get_dark_offset_syn(int_time)
    cnt_dark_corr = cnt_obs - dark_offset
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # non-linearity correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    nlin_factor = cam.corr.get_nlin_factor_syn(cnt_dark_corr)
    cnt_nlin_corr = cnt_dark_corr * nlin_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # flat-fielding correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    flfd_factor = cam.corr.get_flfd_factor_syn(cnt_nlin_corr)
    cnt_flfd_corr = cnt_nlin_corr * flfd_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # apply radiometric response
    #╭────────────────────────────────────────────────────────────────────────────╮#
    rad_factor = cam.corr.get_rad_factor_syn(int_time)
    rad_cam = cnt_flfd_corr * rad_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return rad_cam

def cdata_cam_rad_from_cam_cnt(
        fdir_rad='./data/rad-cam_seg',
        int_times=[1.0, 20.0], # millisecs
        ):

    if not os.path.exists(fdir_rad):
        os.makedirs(fdir_rad)

    fnames = sorted(glob.glob('%s/*.h5' % fdir_rad.replace('rad-cam', 'cnt-cam')))

    for i, fname_h5_ in enumerate(tqdm(fnames)):

        data = cam.util.load_h5(fname_h5_)
        data_true = cam.util.load_h5(fname_h5_.replace('cnt-cam', 'rad-viirs'))

        data = cam.util.load_h5(fname_h5_)
        jday0 = data['_metadata_/jday0']
        lon0 = data['_metadata_/lon0']
        lat0 = data['_metadata_/lat0']
        dtime0 = cam.util.jday_to_dtime(jday0)
        dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

        for ii, int_time in enumerate(int_times):

            # radiance to counts conversion
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rad_true = data_true['rad1']

            cnt_obs = data['cnt_obs%d' % ii]
            rad_obs = cam_dn2rad(cnt_obs, int_time)
            rad_obs[np.isnan(rad_true)] = np.nan
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # figure
            #╭────────────────────────────────────────────────────────────────────────────╮#
            if True:
                plt.close('all')
                fig = plt.figure(figsize=(16, 5))
                fig.suptitle('%s (IntTime=%dms)' % (dtime0_s.replace('_', ' '), int_time))
                # plot
                #╭──────────────────────────────────────────────────────────────╮#
                ax1 = fig.add_subplot(131)
                cs = ax1.imshow(rad_true.T, origin='lower', cmap='jet', zorder=0, vmin=0.0, vmax=0.5)
                ax1.set_title('Radiance ("Ground Truth")')
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax2 = fig.add_subplot(132)
                cs = ax2.imshow(rad_obs.T, origin='lower', cmap='jet', zorder=0, vmin=0.0, vmax=0.5)
                ax2.set_title('Radiance ("Observed")')
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax3 = fig.add_subplot(133)
                cs = ax3.imshow(((rad_obs-rad_true)/rad_true*100.0).T, origin='lower', cmap='seismic', zorder=0, vmin=-30.0, vmax=30.0)
                ax3.set_title('("Observed"$-$"Ground Truth")/"Ground Truth" [%]')
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)
                #╰──────────────────────────────────────────────────────────────╯#
                # save figure
                #╭──────────────────────────────────────────────────────────────╮#
                fig.subplots_adjust(hspace=0.6, wspace=0.6)
                _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                plt.savefig('%s_%s_%2.2d.png' % (_metadata_['Function'], dtime0_s, ii), bbox_inches='tight', metadata=_metadata_)
                #╰──────────────────────────────────────────────────────────────╯#
                # plt.show()
                # sys.exit()
                plt.close(fig)
                plt.clf()
            #╰────────────────────────────────────────────────────────────────────────────╯#

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # fname_h5 = fname_h5_.replace('rad-', 'cnt-')
        # f = h5py.File(fname_h5, 'w')

        # for key in data.keys():
        #     if data[key].size == 1:
        #         f.create_dataset(key, data=data[key])
        #     else:
        #         if 'rad' not in key.lower():
        #             f.create_dataset(key, data=data[key], compression='gzip', compression_opts=9, chunks=True)

        # f['_metadata_/band_width'] = band_width
        # f['_metadata_/scale_factor'] = scale_factor
        # f['_metadata_/int_times'] = np.array(int_times)

        # for ii, int_time in enumerate(int_times):
        #     cnt = data['rad'] * band_width * scale_factor * (int_time/1000.0)
        #     f.create_dataset('cnt%d' % ii, data=cnt.astype(np.int16), compression='gzip', compression_opts=9, chunks=True)

        # f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

if __name__ == '__main__':

    cdata_cam_rad_from_cam_cnt()

    pass
