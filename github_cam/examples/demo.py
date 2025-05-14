import datetime
import glob
import os
import sys

# mpl.use('Agg')
import cam
import cartopy
import cartopy.crs as ccrs
import h5py
import matplotlib.image as mpl_img
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

_Ncoarsen_ = 32
_fdir_local_ = './data/orbit-sim'


def cdata_cam_rad_from_viirs_m4_all(
        fdir_rad='./data/rad-viirs_all',
        ):

    if not os.path.exists(fdir_rad):
        os.makedirs(fdir_rad)

    f = h5py.File('data/orbit-sim/noaa20_orbit_2021-05-18_sample.h5', 'r')
    jday = f['jday'][...]
    lon  = f['lon'][...]
    lat  = f['lat'][...]
    alt  = f['alt'][...]
    heading = f['heading'][...]
    f.close()

    for i, jday0 in enumerate(tqdm(jday)):

        dtime0 = er3t.util.jday_to_dtime(jday0)
        dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')
        lon0 = lon[i]
        lat0 = lat[i]
        alt0 = alt[i]
        hed0 = heading[i]

        libera = cam.platform(lon=lon0, lat=lat0, alt=alt0, dtime=dtime0, ang_pit=0.0, ang_rol=0.0, ang_hed=hed0)
        libera.add_camera('camera')
        libera.geomap(which_mask=None)

        cam_lon = libera.sensor['camera']['longitude']
        cam_lat = libera.sensor['camera']['latitude']

        fname_tags = er3t.util.get_satfile_tag(dtime0, cam_lon[::_Ncoarsen_, ::_Ncoarsen_], cam_lat[::_Ncoarsen_, ::_Ncoarsen_], satellite='noaa20', instrument='viirs', fdir_local=_fdir_local_)

        fnames_02 = []
        fnames_03 = []
        for fname_tag in fname_tags:
            fnames_02 += sorted(glob.glob('%s/VJ102MOD/2021/138/*%s*.nc' % (_fdir_local_, fname_tag)))
            fnames_03 += sorted(glob.glob('%s/VJ103MOD/2021/138/*%s*.nc' % (_fdir_local_, fname_tag)))

        cam_rad = cam_lon.copy()
        cam_rad[~np.isnan(cam_rad)] = 0.0

        cam_lon_ = cam_lon.copy()
        cam_lat_ = cam_lat.copy()
        cam_lon_[np.isnan(cam_lon)] = 0.0
        cam_lat_[np.isnan(cam_lat)] = -100.0
        if len(fname_tags) > 0:
            f03 = er3t.util.viirs_03(fnames=fnames_03)
            f02 = er3t.util.viirs_l1b(fnames=fnames_02, f03=f03, bands=['M04'])

            lon_ = f03.data['lon']['data']
            lat_ = f03.data['lat']['data']
            rad_ = f02.data['rad']['data']

            cam_rad = er3t.util.find_nearest(lon_, lat_, rad_, cam_lon_, cam_lat_, Ngrid_limit=10, fill_value=-1.0)

        cam_rad[np.isnan(cam_rad) & ~np.isnan(cam_lon)] = 0.0

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_h5 = '%s/rad-viirs_%s.h5' % (fdir_rad, dtime0_s)
        f = h5py.File(fname_h5, 'w')
        g = f.create_group('_metadata_')
        g['jday0'] = jday0
        g['hed0'] = hed0
        g['alt0'] = alt0
        g['lon0'] = lon0
        g['lat0'] = lat0

        f.create_dataset('lon', data=cam_lon.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
        f.create_dataset('lat', data=cam_lat.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
        f.create_dataset('rad', data=cam_rad.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        print(dtime0_s)

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if True:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
            ax1 = fig.add_subplot(111, projection=proj)
            ax1.scatter(cam_lon, cam_lat, c=cam_rad, vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='jet')
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
            fig.savefig(fname_h5.replace(fdir_rad, '.').replace('.h5', '.png'), bbox_inches='tight', metadata=_metadata)
            plt.close()
            #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#

        print('%3.3d/%3.3d' % (i+1, jday.size))

def cdata_cam_rad_from_viirs_m4_seg(
        tmhr_range=[5.4, 5.8],
        fdir_rad='./data/rad-viirs_seg',
        ):

    if not os.path.exists(fdir_rad):
        os.makedirs(fdir_rad)

    f = h5py.File('data/orbit-sim/noaa20_orbit_2021-05-18.h5', 'r')
    jday_ = f['jday'][...]
    tmhr = (jday_-int(jday_[0]))*24.0
    logic = (tmhr>=tmhr_range[0]) & (tmhr<=tmhr_range[-1])

    jday = jday_[logic]
    lon  = f['lon'][...][logic]
    lat  = f['lat'][...][logic]
    alt  = f['alt'][...][logic]
    heading = f['heading'][...][logic]
    f.close()

    for i, jday0 in enumerate(tqdm(jday)):

        dtime0 = er3t.util.jday_to_dtime(jday0)
        dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')
        lon0 = lon[i]
        lat0 = lat[i]
        alt0 = alt[i]
        hed0 = heading[i]

        libera = cam.platform(lon=lon0, lat=lat0, alt=alt0, dtime=dtime0, ang_pit=0.0, ang_rol=0.0, ang_hed=hed0)
        libera.add_camera('camera')
        libera.geomap(which_mask=None)

        cam_lon = libera.sensor['camera']['longitude']
        cam_lat = libera.sensor['camera']['latitude']

        fname_tags = er3t.util.get_satfile_tag(dtime0, cam_lon[::_Ncoarsen_, ::_Ncoarsen_], cam_lat[::_Ncoarsen_, ::_Ncoarsen_], satellite='noaa20', instrument='viirs', fdir_local=_fdir_local_)

        fnames_02 = []
        fnames_03 = []
        for fname_tag in fname_tags:
            fnames_02 += sorted(glob.glob('%s/VJ102MOD/2021/138/*%s*.nc' % (_fdir_local_, fname_tag)))
            fnames_03 += sorted(glob.glob('%s/VJ103MOD/2021/138/*%s*.nc' % (_fdir_local_, fname_tag)))

        cam_rad = cam_lon.copy()
        cam_rad[~np.isnan(cam_rad)] = 0.0

        cam_lon_ = cam_lon.copy()
        cam_lat_ = cam_lat.copy()
        cam_lon_[np.isnan(cam_lon)] = 0.0
        cam_lat_[np.isnan(cam_lat)] = -100.0
        if len(fname_tags) > 0:
            f03 = er3t.util.viirs_03(fnames=fnames_03)
            f02 = er3t.util.viirs_l1b(fnames=fnames_02, f03=f03, bands=['M04'])

            lon_ = f03.data['lon']['data']
            lat_ = f03.data['lat']['data']
            rad_ = f02.data['rad']['data']

            cam_rad = er3t.util.find_nearest(lon_, lat_, rad_, cam_lon_, cam_lat_, Ngrid_limit=10, fill_value=-1.0)

        cam_rad[np.isnan(cam_rad) & ~np.isnan(cam_lon)] = 0.0

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_h5 = '%s/rad-viirs_%s.h5' % (fdir_rad, dtime0_s)
        f = h5py.File(fname_h5, 'w')
        g = f.create_group('_metadata_')
        g['jday0'] = jday0
        g['hed0'] = hed0
        g['alt0'] = alt0
        g['lon0'] = lon0
        g['lat0'] = lat0

        f.create_dataset('lon', data=cam_lon.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
        f.create_dataset('lat', data=cam_lat.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
        f.create_dataset('rad', data=cam_rad.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        print(dtime0_s)

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if True:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
            ax1 = fig.add_subplot(111, projection=proj)
            ax1.scatter(cam_lon, cam_lat, c=cam_rad, vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='jet')
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
            fig.savefig(fname_h5.replace(fdir_rad, '.').replace('.h5', '.png'), bbox_inches='tight', metadata=_metadata)
            plt.close()
            #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#

        print('%3.3d/%3.3d' % (i+1, jday.size))

def smooth_cam_rad_from_viirs_m4(
        fdir_rad='./data/rad-viirs_seg',
        ):

    if not os.path.exists(fdir_rad):
        os.makedirs(fdir_rad)

    fnames = sorted(glob.glob('%s/*.h5' % fdir_rad))
    for i, fname_h5 in enumerate(tqdm(fnames)):

        f = h5py.File(fname_h5, 'r')
        jday0 = f['/_metadata_/jday0'][...]
        dtime0 = er3t.util.jday_to_dtime(jday0)
        dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

        lon0 = f['/_metadata_/lon0'][...]
        lat0 = f['/_metadata_/lat0'][...]
        alt0 = f['/_metadata_/alt0'][...]
        cam_lon = f['lon'][...]
        cam_lat = f['lat'][...]
        cam_rad_ = f['rad'][...]
        f.close()

        logic_nan = np.isnan(cam_lon) | np.isnan(cam_lat)
        logic_valid = (~logic_nan) & (cam_rad_>0.0) & (cam_rad_<=0.7)

        cam_lon_ = cam_lon.copy()
        cam_lat_ = cam_lat.copy()
        cam_lon_[np.isnan(cam_lon)] = 0.0
        cam_lat_[np.isnan(cam_lat)] = -100.0

        cam_rad = np.zeros_like(cam_rad_)
        cam_rad = er3t.util.find_nearest(cam_lon[logic_valid], cam_lat[logic_valid], cam_rad_[logic_valid], cam_lon_, cam_lat_, Ngrid_limit=100, fill_value=-1.0)

        cam_rad[logic_nan] = np.nan
        if (np.nanmin(cam_rad) <= 0.0) | (np.nanmin(cam_rad) >= 0.7):
            print(i, '!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(i, np.nanmin(cam_rad))
            print(i, np.nanmax(cam_rad))
            print(i, '!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print(i, np.nanmin(cam_rad))
            print(i, np.nanmax(cam_rad))

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if True:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
            ax1 = fig.add_subplot(111, projection=proj)
            ax1.scatter(cam_lon, cam_lat, c=cam_rad, vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='jet')
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
            fig.savefig(fname_h5.replace(fdir_rad, '.').replace('.h5', '.png'), bbox_inches='tight', metadata=_metadata)
            plt.close()
            # plt.show()
            # sys.exit()
            #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f = h5py.File(fname_h5, 'r+')
        f.create_dataset('rad1', data=cam_rad.astype(np.float32), compression='gzip', compression_opts=9, chunks=True)
        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#





def cdata_cnt_led(
        int_time, # millisecs
        Nbit=12,
        fname_led_pattern='data/aux/led.png',
        scale=0.1,
        ):

    _data_ = cam.util.cal_vza_vaa(cam.common.FPA_NX, cam.common.FPA_NY, cam.common.FPA_DX, cam.common.FPA_DY, coef_dist2ang=cam.common.COEF_DIST2ANG)

    data = mpl_img.imread(fname_nlin_pattern)
    data_gray = np.dot(data[..., :3], [0.5870, 0.2989, 0.1140])
    cnt_nlin_ = np.int_(2.0**Nbit * data_gray * scale)

    x1d_ = np.linspace(0.0, 1.0, data.shape[1])
    y1d_ = np.linspace(0.0, 1.0, data.shape[0])
    x2d_, y2d_ = np.meshgrid(x1d_, y1d_)

    interp = RegularGridInterpolator((y1d_, x1d_), cnt_nlin_, method='nearest')

    x1d = np.linspace(0.0, 1.0, _data_['vza'].shape[0])
    y1d = np.linspace(0.0, 1.0, _data_['vza'].shape[1])
    x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')

    cnt_nlin = interp((x2d, y2d))
    cnt_nlin[_data_['vza']>cam.common.VZA_LIMIT] = np.nan
    print(np.nanmin(cnt_nlin))
    print(np.nanmax(cnt_nlin))

    print(cnt_nlin.shape)


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(data_gray.T, origin='lower', cmap='Grays', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        cs = ax1.imshow(cnt_nlin.T, origin='lower', cmap='jet', zorder=0, vmin=200.0, vmax=400.0) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.scatter(x, y, s=6, c='k', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    cnt_dark_noise_per_ms = np.int_(np.random.randint(scale*100, high=100, size=_data_['vza'].shape)/100.0 * (2.0**Nbit * rad_unct * scale))

    cnt_dark = cnt_dark_base + cnt_dark_noise_per_ms * int_time

    return cnt_dark



def cdata_cam_cnt_from_viirs_rad(
        fdir_cnt='./data/cnt-cam_seg',
        band_width=1.0,  # band width [nm], used to be 20.0
        scale_factor=4.8e5, # scale factor
        int_times=[1.0, 20.0], # millisecs
        Nbit=12,
        ):

    if not os.path.exists(fdir_cnt):
        os.makedirs(fdir_cnt)

    scaler = {
            1.0: 1.0,
            20.0: 10.0,
            }

    cnt_limit = 2.0**Nbit-1

    fnames = sorted(glob.glob('%s/*.h5' % fdir_cnt.replace('cnt-cam', 'rad-viirs')))
    for i, fname_h5_ in enumerate(tqdm(fnames)):

        data = cam.util.load_h5(fname_h5_)
        jday0 = data['_metadata_/jday0']
        lon0 = data['_metadata_/lon0']
        lat0 = data['_metadata_/lat0']
        dtime0 = er3t.util.jday_to_dtime(jday0)
        dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_h5 = fname_h5_.replace('rad-viirs', 'cnt-cam')
        f = h5py.File(fname_h5, 'w')

        for key in data.keys():
            if data[key].size == 1:
                f.create_dataset(key, data=data[key])
            else:
                if 'rad' not in key.lower():
                    f.create_dataset(key, data=data[key], compression='gzip', compression_opts=9, chunks=True)

        f['_metadata_/band_width'] = band_width
        f['_metadata_/scale_factor'] = scale_factor
        f['_metadata_/int_times'] = np.array(int_times)

        for ii, int_time in enumerate(int_times):

            # radiance to counts conversion
            #╭────────────────────────────────────────────────────────────────────────────╮#
            rad_factor = cam.cal.get_rad_factor_syn(int_time)
            cnt_true = data['rad1'] / rad_factor
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # mask out out-of-fov data
            #╭────────────────────────────────────────────────────────────────────────────╮#
            cnt_true = np.ma.masked_array(cnt_true, mask=(cnt_true<=0.0))
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # flat-fielding
            #╭────────────────────────────────────────────────────────────────────────────╮#
            flfd_factor = cam.cal.get_flfd_factor_syn()
            cnt_flfd = cnt_true / flfd_factor
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # non-linearity
            #╭────────────────────────────────────────────────────────────────────────────╮#
            nlin_factor = cam.cal.get_nlin_factor_syn_r(cnt_flfd)
            cnt_nlin = cnt_flfd / nlin_factor
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # dark
            #╭────────────────────────────────────────────────────────────────────────────╮#
            dark_offset = cam.cal.get_dark_offset_syn(int_time)
            cnt_dark = cnt_nlin + dark_offset
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # figure
            #╭────────────────────────────────────────────────────────────────────────────╮#
            if True:
                rcParams['font.size'] = 12

                plt.close('all')
                fig = plt.figure(figsize=(14, 6))
                fig.suptitle('%s (IntTime=%dms)' % (dtime0_s.replace('_', ' '), int_time))
                # plot
                #╭──────────────────────────────────────────────────────────────╮#
                vmax = 400.0 * scaler[int_time]
                cmap = 'jet'

                proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
                ax1 = fig.add_subplot(241, projection=proj)
                cs = ax1.scatter(data['lon'], data['lat'], c=cnt_true, vmin=0.0, vmax=vmax, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap=cmap)
                ax1.set_global()
                ax1.add_feature(cartopy.feature.OCEAN, zorder=0)
                ax1.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
                ax1.coastlines(color='gray', lw=0.5)
                g1 = ax1.gridlines()
                g1.xlocator = FixedLocator(np.arange(-180, 181, 60))
                g1.ylocator = FixedLocator(np.arange(-80, 81, 20))
                ax1.set_title('$DN_{true}$ ($Rad_{true} \\div Fac_{Rad}$)')

                ax2 = fig.add_subplot(242)
                cs = ax2.imshow((cnt_true/cnt_flfd).T, origin='lower', cmap=cmap, zorder=0) #, vmin=0.0, vmax=1.0) #, extent=extent, vmin=0.0, vmax=0.5)
                ax2.set_title('Flat-Fielding Factor ($Fac_{flfd}$)')
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax3 = fig.add_subplot(243)
                cs = ax3.imshow((cnt_flfd/cnt_nlin).T, origin='lower', cmap=cmap, zorder=0, vmin=0.5, vmax=1.5) #, extent=extent, vmin=0.0, vmax=0.5)
                ax3.set_title('Non-Linearity Factor ($Fac_{nlin}$)')
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax4 = fig.add_subplot(244)
                cs = ax4.imshow((cnt_dark-cnt_nlin).T, origin='lower', cmap=cmap, zorder=0, vmin=0.0, vmax=400.0) #, extent=extent, vmin=0.0, vmax=0.5)
                ax4.set_title('Dark DN ($Offset_{dark}$)')
                divider = make_axes_locatable(ax4)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax5 = fig.add_subplot(245)
                cs = ax5.imshow(cnt_true.T, origin='lower', cmap=cmap, zorder=0, vmin=0.0, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
                ax5.set_title('$DN_{true}$ ($Rad_{true} \\div Fac_{Rad}$)')
                divider = make_axes_locatable(ax5)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax6 = fig.add_subplot(246)
                cs = ax6.imshow(cnt_flfd.T, origin='lower', cmap=cmap, zorder=0, vmin=0.0, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
                ax6.set_title('$DN_{flfd}$ ($DN_{true} \\div Fac_{flfd}$)')
                divider = make_axes_locatable(ax6)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax7 = fig.add_subplot(247)
                cs = ax7.imshow(cnt_nlin.T, origin='lower', cmap=cmap, zorder=0, vmin=0.0, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
                ax7.set_title('$DN_{nlin}$ ($DN_{flfd} \\div Fac_{nlin}$)')
                divider = make_axes_locatable(ax7)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)

                ax8 = fig.add_subplot(248)
                cs = ax8.imshow(cnt_dark.T, origin='lower', cmap=cmap, zorder=0, vmin=0.0, vmax=vmax) #, extent=extent, vmin=0.0, vmax=0.5)
                ax8.set_title('$DN_{obs}$ ($DN_{nlin} + Offset_{dark}$)')
                divider = make_axes_locatable(ax8)
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

            logic_bad = (cnt_true.mask==1) | (np.isnan(cnt_true.data)) | (np.isinf(cnt_true.data))

            cnt_true[cnt_true>=cnt_limit] = cnt_limit
            cnt_true_data = cnt_true.data
            cnt_true_data[logic_bad] = 0.0

            cnt_dark[cnt_dark>=cnt_limit] = cnt_limit
            cnt_dark_data = cnt_dark.data
            cnt_dark_data[logic_bad] = 0.0

            f.create_dataset('cnt_true%d' % ii, data=cnt_true_data.astype(np.int16), compression='gzip', compression_opts=9, chunks=True)
            f.create_dataset('cnt_obs%d'  % ii, data=cnt_dark_data.astype(np.int16), compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#





def figure_viirs_globe(index):

    img = mpl_img.imread('data/orbit-sim/snapshot-2021-05-18T00_00_00Z.jpg')[:, :, 1]
    extent = [-180, 180, -90, 90]

    plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img, cmap='Greys_r', vmin=0.0, vmax=255.0, extent=extent)
    ax1.axis('off')

    fname = sorted(glob.glob('data/rad-viirs_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon = f['lon'][...]
    lat = f['lat'][...]
    rad = f['rad1'][...]
    f.close()
    ax1.scatter(lon, lat, c=rad, s=1, vmin=0.0, vmax=0.5, cmap='jet')
    ax1.plot(lon[1023, :], lat[1023, :], color='k', lw=1.0)
    ax1.plot(lon[:, 1023], lat[:, 1023], color='k', lw=1.0)

    # save figure
    #╭──────────────────────────────────────────────────────────────╮#
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
    #╰──────────────────────────────────────────────────────────────╯#
    # plt.show()
    # sys.exit()
    plt.close(fig)
    plt.clf()

def figure_cam_rad_true(index):

    fname = sorted(glob.glob('data/rad-viirs_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_rad = f['rad1'][...]
    f.close()

    dtime0 = er3t.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        proj = ccrs.NearsidePerspective(central_longitude=lon0, central_latitude=lat0)
        ax1 = fig.add_subplot(111, projection=proj)
        ax1.scatter(cam_lon, cam_lat, c=cam_rad, vmin=0.0, vmax=0.5, s=0.1, lw=0.0, transform=ccrs.PlateCarree(), cmap='jet')
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
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#




def figure_cam_cnt_true(index):

    fname = sorted(glob.glob('data/cnt-cam_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_cnt0= f['cnt_true0'][...].astype(np.float32)
    cam_cnt1= f['cnt_true1'][...].astype(np.float32)
    f.close()

    cam_cnt0[cam_cnt0==0.0] = np.nan
    cam_cnt1[cam_cnt1==0.0] = np.nan

    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))

        ax1 = fig.add_subplot(211)
        ax1.imshow(cam_cnt0.T, vmin=0.0, vmax=512, cmap='jet', origin='lower')

        ax2 = fig.add_subplot(212)
        ax2.imshow(cam_cnt1.T, vmin=0.0, vmax=4096, cmap='jet', origin='lower')

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#

def figure_cam_cnt_flfd(index):

    fname = sorted(glob.glob('data/cnt-cam_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    int_time = f['_metadata_/int_times'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_cnt0= f['cnt_true0'][...].astype(np.float32)
    cam_cnt1= f['cnt_true1'][...].astype(np.float32)
    f.close()

    fac0 = cam.corr.get_flfd_factor_syn()
    fac1 = cam.corr.get_flfd_factor_syn()
    cam_cnt0 /= fac0
    cam_cnt1 /= fac1

    cam_cnt0[cam_cnt0==0.0] = np.nan
    cam_cnt1[cam_cnt1==0.0] = np.nan

    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))

        ax1 = fig.add_subplot(211)
        ax1.imshow(cam_cnt0.T, vmin=0.0, vmax=512, cmap='jet', origin='lower')

        ax2 = fig.add_subplot(212)
        ax2.imshow(cam_cnt1.T, vmin=0.0, vmax=4096, cmap='jet', origin='lower')

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#

def figure_cam_cnt_nlin(index):

    fname = sorted(glob.glob('data/cnt-cam_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    int_time = f['_metadata_/int_times'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_cnt0= f['cnt_true0'][...].astype(np.float32)
    cam_cnt1= f['cnt_true1'][...].astype(np.float32)
    f.close()

    fac0 = cam.corr.get_flfd_factor_syn()
    fac1 = cam.corr.get_flfd_factor_syn()
    cam_cnt0 /= fac0
    cam_cnt1 /= fac1

    fac0 = cam.corr.get_nlin_factor_syn_r(cam_cnt0)
    fac1 = cam.corr.get_nlin_factor_syn_r(cam_cnt1)
    cam_cnt0 /= fac0
    cam_cnt1 /= fac1

    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))

        ax1 = fig.add_subplot(211)
        ax1.imshow(cam_cnt0.T, vmin=0.0, vmax=512, cmap='jet', origin='lower')

        ax2 = fig.add_subplot(212)
        ax2.imshow(cam_cnt1.T, vmin=0.0, vmax=4096, cmap='jet', origin='lower')

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#

def figure_cam_cnt_dark(index):

    fname = sorted(glob.glob('data/cnt-cam_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    int_time = f['_metadata_/int_times'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_cnt0= f['cnt_true0'][...].astype(np.float32)
    cam_cnt1= f['cnt_true1'][...].astype(np.float32)
    f.close()

    fac0 = cam.corr.get_flfd_factor_syn()
    fac1 = cam.corr.get_flfd_factor_syn()
    cam_cnt0 /= fac0
    cam_cnt1 /= fac1

    fac0 = cam.corr.get_nlin_factor_syn_r(cam_cnt0)
    fac1 = cam.corr.get_nlin_factor_syn_r(cam_cnt1)
    cam_cnt0 /= fac0
    cam_cnt1 /= fac1

    off0 = cam.corr.get_dark_offset_syn(int_time[0])
    off1 = cam.corr.get_dark_offset_syn(int_time[1])

    cam_cnt0 += off0
    cam_cnt1 += off1

    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))

        ax1 = fig.add_subplot(211)
        cs = ax1.imshow(cam_cnt0.T, vmin=0.0, vmax=512, cmap='jet', origin='lower')
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', '5%', pad='3%')
        # cbar = fig.colorbar(cs, cax=cax)

        ax2 = fig.add_subplot(212)
        cs = ax2.imshow(cam_cnt1.T, vmin=0.0, vmax=4096, cmap='jet', origin='lower')
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes('right', '5%', pad='3%')
        # cbar = fig.colorbar(cs, cax=cax)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#


def figure_dark(index):

    fname = sorted(glob.glob('data/cnt-cam_seg/*.h5'))[index]
    f = h5py.File(fname, 'r')
    lon0 = f['_metadata_/lon0'][...]
    lat0 = f['_metadata_/lat0'][...]
    jday0 = f['_metadata_/jday0'][...]
    int_time = f['_metadata_/int_times'][...]
    cam_lon = f['lon'][...]
    cam_lat = f['lat'][...]
    cam_cnt0= f['cnt_true0'][...].astype(np.float32)
    cam_cnt1= f['cnt_true1'][...].astype(np.float32)
    f.close()

    off0 = cam.corr.get_dark_offset_syn(int_time[0])
    off1 = cam.corr.get_dark_offset_syn(int_time[1])

    dtime0 = cam.util.jday_to_dtime(jday0)
    dtime0_s = dtime0.strftime('%Y-%m-%d_%H:%M:%S')

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))

        ax1 = fig.add_subplot(211)
        cs = ax1.imshow(off0.T, vmin=0.0, vmax=200, cmap='jet', origin='lower')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)

        ax2 = fig.add_subplot(212)
        cs = ax2.imshow(off1.T, vmin=0.0, vmax=400, cmap='jet', origin='lower')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        plt.close(fig)
        plt.clf()
        #╰──────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':


    # figure_viirs_globe(100)
    # figure_cam_rad_true(100)

    # figure_dark(100)

    # figure_cam_cnt_true(100)
    # figure_cam_cnt_flfd(100)
    # figure_cam_cnt_nlin(100)
    figure_cam_cnt_dark(100)

    pass
