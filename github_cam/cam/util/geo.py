import numpy as np

import cam.util

__all__ = [
        'geolocate',
           ]



def geolocate(
        vza,
        vaa,
        ang_pit=0.0,
        ang_rol=0.0,
        ang_hed=0.0,
        mask=None,
        lon0=0.0,
        lat0=0.0,
        heading=45.0,
        altitude=824000.0,
        planet_radius=6378137.0,
        vza_limit=90.0,
        ):

    # calculate sensor zenith angle and sensor azimuth angle
    # adding pitch, roll, heading effects of the mounting platform
    # /------------------------------------------------------------\ #
    # iza, iaa = cam.util.prh2za(self.pitch_angle+pitch_angle_offset, self.roll_angle+roll_angle_offset, self.heading+heading_offset, face_down=True)

    # vx = np.sin(np.deg2rad(vza))*np.sin(np.deg2rad(vaa))
    # vy = np.sin(np.deg2rad(vza))*np.cos(np.deg2rad(vaa))
    # vz = np.cos(np.deg2rad(vza))

    # ix = np.sin(np.deg2rad(iza))*np.sin(np.deg2rad(iaa))
    # iy = np.sin(np.deg2rad(iza))*np.cos(np.deg2rad(iaa))
    # iz = np.cos(np.deg2rad(iza))

    # vx = vx * np.sin(np.deg2rad(iza))*np.sin(np.deg2rad(iaa))
    # vy = vy * np.sin(np.deg2rad(iza))*np.cos(np.deg2rad(iaa))
    # vz = vz * np.cos(np.deg2rad(iza))

    # vza = np.rad2deg(np.arctan2(np.sqrt((vx+ix)**2 + (vy+iy)**2), np.repeat(iz, vza.size).reshape(vza.shape)))
    # vza = np.rad2deg(np.arctan2(np.sqrt((vx+ix)**2 + (vy+iy)**2), vz))
    # vza = np.rad2deg(np.arccos(vz*np.cos(np.deg2rad(iza)))) - iza
    # vza = np.rad2deg(np.arccos(vz))
    # vaa = np.rad2deg(np.arctan2(vx+ix, vy+iy))
    # \------------------------------------------------------------/ #


    # method from Sebastian Schmidt's CAM code
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # ang_sat2viewPoint2earthCenter = 180.0 - np.rad2deg(np.arcsin((planet_radius+altitude)/planet_radius*np.sin(np.deg2rad(vza))))
    # ang_sat2earthCenter2viewPoint = 180.0 - ang_sat2viewPoint2earthCenter - vza
    # radial_distance = 2.0 * np.pi * planet_radius * ang_sat2earthCenter2viewPoint/360.0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # another method
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tan2 = np.tan(np.deg2rad(vza))**2
    A0 = 1.0 + tan2
    B0 = 2.0*tan2*altitude - 2.0*planet_radius
    C0 = tan2*altitude**2

    h = (-B0-np.sqrt(B0**2 - 4.0*A0*C0)) / (2.0*A0)
    degrees = np.rad2deg(np.arccos((planet_radius-h)/planet_radius))
    radial_distance = 2.0 * np.pi * planet_radius * degrees/360.0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if mask is None:
        mask = np.zeros_like(vza, dtype=np.int32)

    indices = np.where((vza<=vza_limit) & (mask==0))
    indices_x = indices[0]
    indices_y = indices[1]

    azimuth = vaa + heading
    lon_, lat_ = cam.util.cal_geodesic_lonlat(lon0, lat0, radial_distance[indices_x, indices_y], azimuth[indices_x, indices_y])

    lon = np.zeros_like(vza, dtype=np.float64)
    lat = np.zeros_like(vza, dtype=np.float64)
    lon[...] = np.nan
    lat[...] = np.nan
    lon[indices_x, indices_y] = lon_
    lat[indices_x, indices_y] = lat_

    return lon, lat



if __name__ == '__main__':

    pass
