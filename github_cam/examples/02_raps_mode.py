import os
import sys
import glob
import datetime
import multiprocessing as mp
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

_date_ = datetime.datetime(2021, 5, 18)
_fdir_raw_ = './data/rad-viirs_raw'

def check_viirs_unagg():

    fdir_geo = '%s/ICDBG/%4.4d/%3.3d' % (_fdir_raw_, _date_.year, _date_.timetuple().tm_yday)
    fdir_rad = '%s/IVCDB/%4.4d/%3.3d' % (_fdir_raw_, _date_.year, _date_.timetuple().tm_yday)

    fnames_geo = sorted(glob.glob('%s/*.h5' % fdir_geo))
    # fnames_rad = sorted(glob.glob('%s/*.h5' % fdir_rad))
    # patterns = []
    # print(len(fnames_rad))

    for i, fname_geo in enumerate(tqdm(fnames_geo)):
        pattern = '_'.join(os.path.basename(fname_geo).split('_')[1:6])
        dtime0_s = datetime.datetime.strptime('_'.join(pattern.split('_')[1:3])[:-1], 'd%Y%m%d_t%H%M%S').strftime('%Y-%m-%d_%H:%M:%S')
        fnames_rad = sorted(glob.glob('%s/*%s*.h5' % (fdir_rad, pattern)))
        if len(fnames_rad) == 1:
            fname_rad = fnames_rad[0]
            with h5py.File(fname_geo, 'r') as f_geo:
                lon = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Longitude'][...]
                lat = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Latitude'][...]
                logic_nan = (lon>180.0) | (lon<-180.0) | (lat>90.0) | (lat<-90.0)
                lon[logic_nan] = np.nan
                lat[logic_nan] = np.nan
                lon_ = lon[_index_xs_:_index_xe_, ...]
                lat_ = lat[_index_xs_:_index_xe_, ...]
            with h5py.File(fname_rad, 'r') as f_rad:
                rad = f_rad['All_Data/VIIRS-DualGain-Cal-IP_All/radiance_3'][...]/1000.0
                rad[rad<0.0] = 0.0
                rad_ = rad[_index_xs_:_index_xe_, ...]

            # figure
            #╭────────────────────────────────────────────────────────────────────────────╮#
            if True:
                lon0 = lon[int(lon.shape[0]//2), int(lon.shape[1]//2)]
                lat0 = 0.0
                plt.close('all')
                fig = plt.figure(figsize=(8, 6))
                proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
                ax1 = fig.add_subplot(111, projection=proj)
                ax1.scatter(lon[::2, ::4], lat[::2, ::4], c=rad[::2, ::4], vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='Greys_r', alpha=0.1)
                ax1.scatter(lon_, lat_, c=rad_, vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='jet')
                ax1.set_global()
                ax1.add_feature(cartopy.feature.OCEAN, zorder=0)
                ax1.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
                ax1.coastlines(color='gray', lw=0.5)
                g1 = ax1.gridlines()
                g1.xlocator = FixedLocator(np.arange(-180, 181, 60))
                g1.ylocator = FixedLocator(np.arange(-80, 81, 20))
                ax1.set_title(dtime0_s.replace('_', ' '))
                # save figure
                #╭──────────────────────────────────────────────────────────────╮#
                fig.subplots_adjust(hspace=0.3, wspace=0.3)
                _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                fig.savefig('viirs-rad-unagg_%s.png' % pattern, bbox_inches='tight', metadata=_metadata)
                plt.close()
                # plt.show()
                # sys.exit()
                #╰──────────────────────────────────────────────────────────────╯#
            #╰────────────────────────────────────────────────────────────────────────────╯#

            print(i, pattern, np.nanmin(rad), np.nanmax(rad))

    return



def cdata_raps_ratio(
        index,
        fdir_rad='./data/rad-cam_raps',
        fdir_cnt='./data/cnt-cam_raps',
        fdir_raw='./data/rad-viirs_raw',
        # int_times=[1.0, 20.0], # millisecs
        int_times=[20.0], # millisecs
        ):

    if not os.path.exists(fdir_rad):
        os.makedirs(fdir_rad)

    # retrieve data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('%s/noaa20_orbit_2021-05-18.h5' % fdir_raw, 'r')
    jday0 = f['jday'][...][index]
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # time treatment
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # ratio
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_ = sorted(glob.glob('%s/*.h5' % (fdir_cnt)))
    fname_ = sorted(glob.glob('%s/*%s*.h5' % (fdir_cnt, dtime0_s)))[0]

    index_fname_ = fnames_.index(fname_)
    if index_fname_ == 0:

        _ratio_ = np.ones((cam.common.FPA_NX, cam.common.FPA_NY), dtype=np.float32)
        _ratio_[...] = np.nan

        for fname_ in fnames_:
            dtime0_s_ = '_'.join(os.path.basename(fname_).split('.')[0].split('_')[-2:])
            dtime0_ = datetime.datetime.strptime(dtime0_s_, '%Y-%m-%d_%H:%M:%S')
            if dtime0_ <= dtime0:
                fname__ = fname_.replace('cnt-cam', 'rad-viirs')
                with h5py.File(fname_, 'r') as f_:
                    hed0 = f_['_metadata_/hed0'][...]
                    hed0_ = f_['_metadata_/hed0_'][...]
                    cnt_obs = f_['cnt_obs1'][...]
                    rad_obs = cam.corr.dn2rad(cnt_obs, 20.0)

                with h5py.File(fname__, 'r') as f__:
                    rad_true= f__['rad1'][...]

                sr = (hed0 - hed0_)
                mask = cam.mask.mask_c_stripe_ss(sr=sr)
                logic_mask = (mask==1)

                _ratio_[logic_mask] = rad_obs[logic_mask]/rad_true[logic_mask]
    else:
        fname_prev_ = sorted(glob.glob('%s/*.h5' % (fdir_rad)))[index_fname_-1]
        with h5py.File(fname_prev_, 'r') as f_:
            _ratio_ = f_['ratio'][...]

        fname__ = fname_.replace('cnt-cam', 'rad-viirs')

        with h5py.File(fname_, 'r') as f_:
            hed0 = f_['_metadata_/hed0'][...]
            hed0_ = f_['_metadata_/hed0_'][...]
            cnt_obs = f_['cnt_obs1'][...]
            rad_obs = cam.corr.dn2rad(cnt_obs, 20.0)

        with h5py.File(fname__, 'r') as f__:
            rad_true= f__['rad1'][...]

        sr = (hed0 - hed0_)
        mask = cam.mask.mask_c_stripe_ss(sr=sr)
        logic_mask = (mask==1)

        _ratio_[logic_mask] = rad_obs[logic_mask]/rad_true[logic_mask]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # save data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = fname__.replace('rad-viirs', 'rad-cam')
    f = h5py.File(fname_h5, 'w')
    f.create_dataset('rad_obs1', data=rad_obs.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
    f.create_dataset('ratio', data=_ratio_.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
    f.create_dataset('mask', data=mask.astype(np.int16), compression='gzip', compression_opts=9, chunks=True)
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    print(dtime0_s)
    print(fname_h5)

def figure_raps_mode(
        index,
        fdir_rad='./data/rad-cam_raps',
        fdir_raw='./data/rad-viirs_raw',
        # int_times=[1.0, 20.0], # millisecs
        int_times=[20.0], # millisecs
        ):

    # if not os.path.exists(fdir_rad):
    #     os.makedirs(fdir_rad)

    # retrieve data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('%s/noaa20_orbit_2021-05-18.h5' % fdir_raw, 'r')
    jday0 = f['jday'][...][index]
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # time treatment
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # get rad-viirs file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = sorted(glob.glob('%s/*%s*.h5' % (fdir_rad, dtime0_s)))[0]
    data = cam.util.load_h5(fname_h5)
    fname_h5_ = sorted(glob.glob('%s/*%s*.h5' % (fdir_rad.replace('rad-cam', 'rad-viirs'), dtime0_s)))[0]
    data_true = cam.util.load_h5(fname_h5_.replace('rad-cam', 'rad-viirs'))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # retrieve data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    jday0 = data_true['_metadata_/jday0']
    lon0 = data_true['_metadata_/lon0']
    lat0 = data_true['_metadata_/lat0']
    hed0 = data_true['_metadata_/hed0']
    hed0_ = data_true['_metadata_/hed0_']
    lon_cam = data_true['lon']
    lat_cam = data_true['lat']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # rotate central stripe
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # sr = (hed0 - hed0_)
    # mask = cam.mask.mask_c_stripe_ss(sr=sr)
    mask = data['mask']
    logic_mask = (mask==1)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for ii, int_time in enumerate(int_times):

        # radiance to counts conversion
        #╭────────────────────────────────────────────────────────────────────────────╮#
        rad_true = data_true['rad1']

        # cnt_obs = data['cnt_obs%d' % (ii)]
        # cnt_obs = data['cnt_obs%d' % (ii+1)]
        # rad_obs = cam.corr.dn2rad(cnt_obs, int_time)
        rad_obs = data['rad_obs1']
        rad_obs[np.isnan(rad_true)] = np.nan
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # ratio for flat-fielding
        #╭────────────────────────────────────────────────────────────────────────────╮#
        _ratio_ = data['ratio']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if True:
            plt.close('all')
            fig = plt.figure(figsize=(14, 14))
            fig.suptitle(dtime0_s.replace('_', ' '), fontsize=24, y=0.96)

            # pixel mask
            #╭────────────────────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(221, aspect='equal')

            ax1.imshow(mask.T, origin='lower', aspect='auto', cmap='binary', vmin=0, vmax=1, zorder=0)
            ax1.axhline(1024, color='b', lw=1.0, ls='--', zorder=1)
            ax1.axvline(1024, color='r', lw=1.0, ls='--', zorder=1)

            ax1.set_xlim((0, 2048))
            ax1.set_ylim((0, 2048))
            ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 2049, 512)))
            ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 2049, 512)))
            ax1.set_xlabel('FPA X')
            ax1.set_ylabel('FPA Y')
            ax1.set_title('Pixel Mask')
            patches_legend = [
                              mpatches.Patch(color='black' , label='Imagery Stripe'), \
                             ]
            ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # flat-fielding factor
            #╭────────────────────────────────────────────────────────────────────────────╮#
            ax2 = fig.add_subplot(222, aspect='equal')

            cs = ax2.imshow(_ratio_.T, origin='lower', aspect='auto', cmap='seismic', vmin=0.6, vmax=1.4, zorder=0)
            ax2.axhline(1024, color='b', lw=1.0, ls='--', zorder=1)
            ax2.axvline(1024, color='r', lw=1.0, ls='--', zorder=1)

            ax2.set_xlim((0, 2048))
            ax2.set_ylim((0, 2048))
            ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, 2049, 512)))
            ax2.yaxis.set_major_locator(FixedLocator(np.arange(0, 2049, 512)))
            ax2.set_xlabel('FPA X')
            ax2.set_ylabel('FPA Y')
            ax2.set_title('Flat-Fielding Factor')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax)
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # camera radiance
            #╭────────────────────────────────────────────────────────────────────────────╮#
            proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=22.0)
            ax3 = fig.add_subplot(223, projection=proj)
            ax3.scatter(lon_cam            , lat_cam            , c=rad_obs            , s=0.5, lw=0.0, transform=ccrs.PlateCarree(), vmin=0.0, vmax=0.5, cmap='Greys_r', alpha=0.5)
            ax3.scatter(lon_cam[logic_mask], lat_cam[logic_mask], c=rad_obs[logic_mask], s=1.0, lw=0.0, transform=ccrs.PlateCarree(), vmin=0.0, vmax=0.5, cmap='jet')
            ax3.plot(lon_cam[:, 1023]   , lat_cam[:, 1023]   , color='b', lw=0.5, ls='--', transform=ccrs.PlateCarree())
            ax3.plot(lon_cam[1023, :]   , lat_cam[1023, :]   , color='r', lw=0.5, ls='--', transform=ccrs.PlateCarree())
            ax3.set_global()
            ax3.add_feature(cartopy.feature.OCEAN, zorder=0)
            ax3.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
            ax3.coastlines(color='gray', lw=0.5)
            g3 = ax3.gridlines()
            g3.xlocator = FixedLocator(np.arange(-180, 181, 60))
            g3.ylocator = FixedLocator(np.arange(-80, 81, 20))
            ax3.set_title('Radiance (WFOV Camera L1B)')
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # true radiance
            #╭────────────────────────────────────────────────────────────────────────────╮#
            ax4 = fig.add_subplot(224, projection=proj)
            ax4.scatter(lon_cam[logic_mask], lat_cam[logic_mask], c=rad_true[logic_mask], s=1.0, lw=0.0, transform=ccrs.PlateCarree(), vmin=0.0, vmax=0.5, cmap='jet')
            ax4.set_global()
            ax4.add_feature(cartopy.feature.OCEAN, zorder=0)
            ax4.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
            ax4.coastlines(color='gray', lw=0.5)
            g4 = ax4.gridlines()
            g4.xlocator = FixedLocator(np.arange(-180, 181, 60))
            g4.ylocator = FixedLocator(np.arange(-80, 81, 20))
            ax4.set_title('Radiance (Unaggregated VIIRS M4)')
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
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

def main_raps(
        tmhr_range=[0, 24],
        ):

    # retrieve data indices
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('%s/noaa20_orbit_2021-05-18.h5' % _fdir_raw_, 'r')
    jday_ = f['jday'][...]
    tmhr = (jday_-int(jday_[0]))*24.0
    indices = np.where((tmhr>=tmhr_range[0]) & (tmhr<=tmhr_range[-1]))[0]
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # data processing
    #╭────────────────────────────────────────────────────────────────────────────╮#
    for index in tqdm(indices):
        cdata_raps_ratio(index)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # multiprocessing
    #╭────────────────────────────────────────────────────────────────────────────╮#
    Ncpu = 14
    with mp.Pool(processes=Ncpu) as pool:
        r = list(tqdm(pool.imap(figure_raps_mode, indices), total=indices.size))
    #╰────────────────────────────────────────────────────────────────────────────╯#




def read_viirs_unagg(
        fname_geo,
        fname_rad,
        Ndet=16,
        ):

    patterns_geo = os.path.basename(fname_geo).replace('.h5', '').split('_')[1:5]
    patterns_rad = os.path.basename(fname_rad).replace('.h5', '').split('_')[1:5]

    if patterns_geo != patterns_rad:
        msg = 'Error [read_viirs_unagg]: geolocation-file and radiance-file does not match.'
        raise OSError(msg)

    with h5py.File(fname_geo, 'r') as f_geo:
        lon = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Longitude'][...]
        lat = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Latitude'][...]
        logic_nan = (lon>180.0) | (lon<-180.0) | (lat>90.0) | (lat<-90.0)
        lon[logic_nan] = np.nan
        lat[logic_nan] = np.nan

    with h5py.File(fname_rad, 'r') as f_rad:
        rad = f_rad['All_Data/VIIRS-DualGain-Cal-IP_All/radiance_3'][...]/1000.0
        rad[rad<0.0] = 0.0

if __name__ == '__main__':

    # read_viirs_unagg()
    # check_viirs_unagg()
    # figure_raps_mode(8888)

    main_raps(tmhr_range=[10.5, 10.7])
    main_raps(tmhr_range=[12.2333, 12.4334])
    main_raps(tmhr_range=[13.9, 14.1])

    pass
