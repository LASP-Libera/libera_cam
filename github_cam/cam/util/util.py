import datetime
import fnmatch
import os

import h5py
import numpy as np
import pysolar
from scipy import interpolate

__all__ = [
        'polyval',
        'cal_vza_vaa',
        'get_all_files',
        'get_all_folders',
        'if_file_exists',
        'cal_heading',
        'cal_solar_angles',
        'cal_solar_factor',
        'cal_step_offset',
        'prh2za',
        'muslope',
        'dtime_to_jday',
        'jday_to_dtime',
        'interp',
        'load_h5',
        'save_h5',
        'get_solar_kurudz',
        'get_slit_func',
        'cal_weighted_flux',
        'cal_geodesic_dist',
        'cal_geodesic_lonlat',
        ]

def polyval(p, x):

    Ndeg = p.shape[-1]

    y = np.zeros_like(x)

    for ideg in range(Ndeg):
        y = y + p[..., ideg]*np.power(x, Ndeg-ideg-1)

    return y

def cal_vza_vaa(
        fpa_nx,
        fpa_ny,
        fpa_dx,
        fpa_dy,
        coef_dist2ang=np.array([1.397428e-02, -1.500364e-01, 6.352646e-01, -9.551312e-01,    9.042359, 0]),
        ):

    # calculate viewing zenith angles (vaa) and viewing azimuth angles (vza)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fpa_cx = fpa_nx//2
    fpa_cy = fpa_ny//2

    fpa_ix = np.arange(fpa_nx)
    fpa_iy = np.arange(fpa_ny)

    fpa_x0 = (fpa_ix-fpa_cx + 0.5) * fpa_dx
    fpa_y0 = (fpa_iy-fpa_cy + 0.5) * fpa_dy

    fpa_x, fpa_y = np.meshgrid(fpa_x0, fpa_y0, indexing='ij')

    vaa = np.rad2deg(np.arctan2(fpa_x, fpa_y))

    fpa_dist = np.sqrt(fpa_x**2 + fpa_y**2)
    vza = np.polyval(coef_dist2ang, fpa_dist)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    data = {
            'fpa_cx': np.array(fpa_cx),
            'fpa_cy': np.array(fpa_cy),
            'vza'   : vza,
            'vaa'   : vaa,
            }

    return data

def get_all_files(fdir, pattern='*'):

    fnames = []
    for fdir_root, fdir_sub, fnames_tmp in os.walk(fdir):
        for fname_tmp in fnames_tmp:
            if fnmatch.fnmatch(fname_tmp, pattern):
                fnames.append(os.path.join(fdir_root, fname_tmp))
    return sorted(fnames)

def get_all_folders(fdir, pattern='*'):

    fnames = get_all_files(fdir)

    folders = []
    for fname in fnames:
        folder_tmp = os.path.abspath(os.path.dirname(os.path.relpath(fname)))
        if (folder_tmp not in folders) and fnmatch.fnmatch(folder_tmp, pattern):
                folders.append(folder_tmp)

    return folders

def if_file_exists(fname, exitTag=True):

    """
    Check whether file exists.

    Input:
        fname: file path of the data
    Output:
        None
    """

    if not os.path.exists(fname):
        if exitTag is True:
            exit(f"Error   [if_file_exists]: cannot find '{fname}'")
        else:
            print(f"Warning [if_file_exists]: cannot find '{fname}'")

def cal_heading(lon, lat):

    dx = lat[1:]-lat[:-1]
    dy = lon[1:]-lon[:-1]

    heading = np.rad2deg(np.arctan2(dy, dx))
    heading = np.append(heading[0], heading) % 360.0

    return heading

def cal_solar_angles(julian_day, longitude, latitude, altitude):

    dateRef = datetime.datetime(1, 1, 1)
    jdayRef = 1.0

    sza = np.zeros_like(julian_day)
    saa = np.zeros_like(julian_day)

    for i in range(julian_day.size):

        jday = julian_day[i]

        dtime_i = (dateRef + datetime.timedelta(days=jday-jdayRef)).replace(tzinfo=datetime.UTC)

        sza_i = 90.0 - pysolar.solar.get_altitude(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        if sza_i < 0.0 or sza_i > 90.0:
            sza_i = np.nan
        sza[i] = sza_i

        saa_i = pysolar.solar.get_azimuth(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        if saa_i >= 0.0:
            if 0.0<=saa_i<=180.0:
                saa_i = 180.0 - saa_i
            elif 180.0<saa_i<=360.0:
                saa_i = 540.0 - saa_i
            else:
                saa_i = np.nan
        elif saa_i < 0.0:
            if -180.0<=saa_i<0.0:
                saa_i = -saa_i + 180.0
            elif -360.0<=saa_i<-180.0:
                saa_i = -saa_i - 180.0
            else:
                saa_i = np.nan
        saa[i] = saa_i

    return sza, saa

def cal_solar_factor(dtime):

    """
    Calculate solar factor that accounts for Sun-Earth distance
    Input:
        dtime: datetime.datetime object
    Output:
        solfac: solar factor
    """

    doy = dtime.timetuple().tm_yday
    eps = 0.0167086
    perh= 4.0
    rsun = (1.0 - eps*np.cos(0.017202124161707175*(doy-perh)))
    solfac = 1.0/(rsun**2)

    return solfac

def cal_step_offset(x_ref, x_target, fill_value=np.nan, offset_range=[-900, 900]):

    from scipy.ndimage import shift
    from scipy.stats import pearsonr

    logic = (np.logical_not(np.isnan(x_ref))) & (np.logical_not(np.isnan(x_target)))
    x_ref    = x_ref[logic]
    x_target = x_target[logic]
    x_ref    = x_ref/x_ref.max()
    x_target = x_target/x_target.max()

    offsets = np.arange(offset_range[0], offset_range[1], dtype=np.float64)
    cross_corr = np.zeros_like(offsets)
    for i, offset in enumerate(offsets):
        x0 = shift(x_target, offset, cval=fill_value)
        logic = (np.logical_not(np.isnan(x0))) & (np.logical_not(np.isnan(x_ref)))
        coef = pearsonr(x_ref[logic], x0[logic])
        cross_corr[i] = coef[0]

    step_offset = offsets[np.argmax(cross_corr)]

    return step_offset

def prh2za_test(ang_pit, ang_rol, ang_head, is_rad=False, face_down=False):

    """
    input:
    ang_pit   (Pitch)   [deg]: positive (+) values indicate nose up (tail down)
    ang_rol   (Roll)    [deg]: positive (+) values indicate right wing down (left side up)
    ang_head  (Heading) [deg]: positive (+) values clockwise, w.r.t. north

    "vec": normal vector of the surface of the sensor

    return:
    ang_zenith : angle of "vec" [deg]
    ang_azimuth: angle of "vec" [deg]: positive (+) values clockwise, w.r.t. north
    """

    if not is_rad:
        rad_pit  = np.deg2rad(ang_pit)
        rad_rol  = np.deg2rad(ang_rol)
        rad_head = np.deg2rad(ang_head)

    uz =  np.cos(rad_rol)*np.cos(rad_pit)
    ux =  np.sin(rad_rol)
    uy = -np.cos(rad_rol)*np.sin(rad_pit)

    vz = uz.copy()
    vx = ux*np.cos(rad_head) + uy*np.sin(rad_head)
    vy = uy*np.cos(rad_head) - ux*np.sin(rad_head)

    ang_zenith  = np.rad2deg(np.arccos(vz))
    if face_down:
        ang_azimuth = np.rad2deg(np.arctan2(-vx,-vy))
    else:
        ang_azimuth = np.rad2deg(np.arctan2(vx,vy))

    ang_azimuth = (ang_azimuth + 360.0) % 360.0

    return ang_zenith, ang_azimuth

def prh2za(ang_pit, ang_rol, ang_head, is_rad=False, face_down=False):

    """
    input:
    ang_pit   (Pitch)   [deg]: positive (+) values indicate nose up (tail down)
    ang_rol   (Roll)    [deg]: positive (+) values indicate right wing down (left side up)
    ang_head  (Heading) [deg]: positive (+) values clockwise, w.r.t. north

    "vec": normal vector of the surface of the sensor

    return:
    ang_zenith : angle of "vec" [deg]
    ang_azimuth: angle of "vec" [deg]: positive (+) values clockwise, w.r.t. north
    """

    if not is_rad:
        rad_pit  = np.deg2rad(ang_pit)
        rad_rol  = np.deg2rad(ang_rol)
        rad_head = np.deg2rad(ang_head)

    uz =  np.cos(rad_rol)*np.cos(rad_pit)
    ux =  np.sin(rad_rol)
    uy = -np.cos(rad_rol)*np.sin(rad_pit)

    vz = uz.copy()
    vx = ux*np.cos(rad_head) + uy*np.sin(rad_head)
    vy = uy*np.cos(rad_head) - ux*np.sin(rad_head)

    ang_zenith  = np.rad2deg(np.arccos(vz))
    if face_down:
        ang_azimuth = np.rad2deg(np.arctan2(-vx,-vy))
    else:
        ang_azimuth = np.rad2deg(np.arctan2(vx,vy))

    ang_azimuth = (ang_azimuth + 360.0) % 360.0

    return ang_zenith, ang_azimuth

def muslope(sza, saa, iza, iaa, is_rad=False):

    if not is_rad:
        rad_sza = np.deg2rad(sza)
        rad_saa = np.deg2rad(saa)
        rad_iza = np.deg2rad(iza)
        rad_iaa = np.deg2rad(iaa)

    zs = np.cos(rad_sza)
    ys = np.sin(rad_sza) * np.cos(rad_saa)
    xs = np.sin(rad_sza) * np.sin(rad_saa)

    zi = np.cos(rad_iza)
    yi = np.sin(rad_iza) * np.cos(rad_iaa)
    xi = np.sin(rad_iza) * np.sin(rad_iaa)

    mu = xs*xi + ys*yi + zs*zi

    return mu

def load_h5(fname):

    def get_variable_names(obj, prefix=''):

        """
        Purpose: Walk through the file and extract information of data groups and data variables

        Input: h5py file object <f>, e.g., f = h5py.File('file.h5', 'r')

        Outputs:
            data variable path in the format of <['group1/variable1']> to
            mimic the style of accessing HDF5 data variables using h5py, e.g.,
            <f['group1/variable1']>
        """

        for key in obj.keys():

            item = obj[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):
                yield path
            elif isinstance(item, h5py.Group):
                yield from get_variable_names(item, prefix=path)

    data = {}
    f = h5py.File(fname, 'r')
    keys = get_variable_names(f)
    for key in keys:
        data[key[1:]] = f[key[1:]][...]
    f.close()
    return data

def save_h5(fname, data):

    data = {}
    f = h5py.File(fname, 'w')
    for key in data.keys():
        f[key] = data[key]
    f.close()

    print('Message [save_h5]: Data has been successfully saved into \'%s\'.' % fname)

def get_slit_func(wvl, slit_func_file=None, wvl_joint=950.0):

    if slit_func_file is None:
        if wvl <= wvl_joint:
            slit_func_file = '%s/slit/vis_0.1nm_s.dat' % ssfr.common.fdir_data
        else:
            slit_func_file = '%s/slit/nir_0.1nm_s.dat' % ssfr.common.fdir_data

    data_slt = np.loadtxt(slit_func_file)

    return data_slt

def get_solar_kurudz(kurudz_file=None):

    if kurudz_file is None:
        kurudz_file = '%s/solar/kurudz_0.1nm.dat' % ssfr.common.fdir_data

    data_sol = np.loadtxt(kurudz_file)
    data_sol[:, 1] /= 1000.0

    return data_sol

def cal_weighted_flux(wvl, data_wvl, data_flux, slit_func_file=None, wvl_joint=950.0):

    data_slt = get_slit_func(wvl, slit_func_file=slit_func_file, wvl_joint=wvl_joint)

    weights  = data_slt[:, 1]
    wvl_x    = data_slt[:, 0] + wvl
    flux     = np.average(np.interp(wvl_x, data_wvl, data_flux), weights=weights)

    return flux

def dtime_to_jday(dtime):

    jday = (dtime - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    return jday

def jday_to_dtime(jday):

    dtime = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=np.round(((jday-1)*86400.0), decimals=0))

    return dtime

def interp(x, x0, y0, mode='linear'):

    if mode == 'nearest':
        f = interpolate.interp1d(x0, y0, bounds_error=False, kind=mode)
    else:
        logic = (~np.isnan(x0) & ~np.isnan(y0))
        f = interpolate.interp1d(x0[logic], y0[logic], bounds_error=False, kind=mode)

    return f(x)

def cal_geodesic_dist(lon0, lat0, lon1, lat1):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_dist]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    lon1 = np.array(lon1).ravel()
    lat1 = np.array(lat1).ravel()

    geo0 = cg.Geodesic()

    points0 = np.transpose(np.vstack((lon0, lat0)))

    points1 = np.transpose(np.vstack((lon1, lat1)))

    output = np.squeeze(np.asarray(geo0.inverse(points0, points1)))

    dist = output[..., 0]

    return dist

def cal_geodesic_lonlat(lon0, lat0, dist, azimuth):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_lonlat]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    dist = np.array(dist).ravel()
    azimuth = np.array(azimuth).ravel()

    points = np.transpose(np.vstack((lon0, lat0)))

    geo0 = cg.Geodesic()

    output = np.squeeze(np.asarray(geo0.direct(points, azimuth, dist)))

    lon1 = output[..., 0]
    lat1 = output[..., 1]

    return lon1, lat1

if __name__ == '__main__':

    pass
