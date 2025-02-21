import os
import sys
import datetime
import numpy as np


import cam


__all__ = ['platform']



class platform(object):

    """
    Purpose: generate an platform object

    Input:
        alt=: altitude of the platform, units in meter
    """

    def __init__(self,
            lon=0.0,
            lat=0.0,
            alt=824000.0,
            dtime=datetime.datetime.now(),
            ang_pit=0.0,
            ang_rol=0.0,
            ang_hed=0.0,
            planet_radius=cam.common.PLANET_RADIUS
            ):

        self.ang_pit = ang_pit
        self.ang_rol = ang_rol
        self.ang_hed = ang_hed

        self.lon = lon
        self.lat = lat
        self.alt = alt

        self.dtime = dtime
        self.jday  = cam.util.dtime_to_jday(dtime)

        self.planet_radius = planet_radius

        self.sensor = {}

    def add_camera(self,
            name,
            ang_pit_offset=0.0,
            ang_rol_offset=0.0,
            ang_hed_offset=0.0,
            fpa_nx=cam.common.FPA_NX,
            fpa_ny=cam.common.FPA_NY,
            fpa_dx=cam.common.FPA_DX,
            fpa_dy=cam.common.FPA_DY,
            coef_dist2ang=cam.common.COEF_DIST2ANG,
            coef_ang2dist=cam.common.COEF_ANG2DIST,
            ):

        self.sensor[name] = cam.util.cal_vza_vaa(fpa_nx, fpa_ny, fpa_dx, fpa_dy, coef_dist2ang=coef_dist2ang)

    def add_image_mask(self,
            sensors = None,
            delta_vza = 30.0,
            nline_stripe = 21,
            viewing_zenith_angle_limit = 62.0,
            # crop_delta_degree = 0.14,
            crop_delta_degree = None,
            rotate_degree = 0.0,
            straighten = False,
            ):

        if sensors is None:

            sensors = self.sensor

        for sensor0 in sensors:
            vza0 = self.sensor[sensor0]['vza']
            vaa0 = self.sensor[sensor0]['vaa']

            index_y_center = self.sensor[sensor0]['fpa_cy']
            index_x_center = self.sensor[sensor0]['fpa_cx']
            index_y_edge   = np.argmin(np.abs(vza0[self.sensor[sensor0]['fpa_cx'], index_y_center:]-delta_vza)) + index_y_center
            indices_y = [index_y_edge]

            lon, lat = cam.util.geolocate(vza0, vaa0, mask=None, lon0=self.lon, lat0=self.lat, heading=self.ang_hed, altitude=self.alt, planet_radius=self.planet_radius)

            # central horizontal stripe
            #╭────────────────────────────────────────────────────────────────────────────╮#
            mask0 = np.ones(vza0.shape, dtype=np.int16)
            if crop_delta_degree is not None:
                lon0 = lon[:, index_y_center-nline_stripe:index_y_center+nline_stripe]
                lat0 = lat[:, index_y_center-nline_stripe:index_y_center+nline_stripe]
                lon_c = np.repeat(np.nanmean(lon0[:, [nline_stripe-1, nline_stripe]], axis=1), lon0.shape[1]).reshape(lon0.shape)
                lat_c = np.repeat(np.nanmean(lat0[:, [nline_stripe-1, nline_stripe]], axis=1), lat0.shape[1]).reshape(lat0.shape)
                logic = np.sqrt((lon0-lon_c)**2 + (lat0-lat_c)**2) <= crop_delta_degree
                mask0[:, index_y_center-nline_stripe:index_y_center+nline_stripe][logic] = 0
            else:
                mask0[:, index_y_center-nline_stripe:index_y_center+nline_stripe] = 0
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # central vertical stripe
            #╭────────────────────────────────────────────────────────────────────────────╮#
            mask1 = np.ones(vza0.shape, dtype=np.int16)
            if crop_delta_degree is not None:
                lon0 = lon[index_x_center-nline_stripe:index_x_center+nline_stripe, :]
                lat0 = lat[index_x_center-nline_stripe:index_x_center+nline_stripe, :]
                lon_c = np.repeat(np.nanmean(lon0[[nline_stripe-1, nline_stripe], :], axis=0), lon0.shape[0]).reshape(lon0.shape)
                lat_c = np.repeat(np.nanmean(lat0[[nline_stripe-1, nline_stripe], :], axis=0), lat0.shape[0]).reshape(lat0.shape)
                logic = np.sqrt((lon0-lon_c)**2 + (lat0-lat_c)**2) <= crop_delta_degree
                mask1[index_x_center-nline_stripe:index_x_center+nline_stripe, :][logic] = 0
            else:
                mask1[index_x_center-nline_stripe:index_x_center+nline_stripe, :] = 0
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # forward stripe
            #╭────────────────────────────────────────────────────────────────────────────╮#
            mask2 = np.ones(vza0.shape, dtype=np.int16)
            for index_y in indices_y:
                if straighten:
                    delta_lon0 = lon[self.sensor[sensor0]['fpa_cx'], index_y] - lon[self.sensor[sensor0]['fpa_cx'], self.sensor[sensor0]['fpa_cy']]
                    delta_lat0 = lat[self.sensor[sensor0]['fpa_cx'], index_y] - lat[self.sensor[sensor0]['fpa_cx'], self.sensor[sensor0]['fpa_cy']]
                    degree0 = np.arctan(delta_lat0/delta_lon0)

                    lon_s0 = lon_c[:, 0] + delta_lon0 - np.abs(crop_delta_degree * np.cos(degree0))
                    lon_e0 = lon_c[:, 0] + delta_lon0 + np.abs(crop_delta_degree * np.cos(degree0))

                    lat_s0 = lat_c[:, 0] + delta_lat0 - np.abs(crop_delta_degree * np.sin(degree0))
                    lat_e0 = lat_c[:, 0] + delta_lat0 + np.abs(crop_delta_degree * np.sin(degree0))

                    lon_s_ = np.repeat(lon_s0, lon.shape[1]).reshape(lon.shape)
                    lat_s_ = np.repeat(lat_s0, lat.shape[1]).reshape(lat.shape)
                    lon_e_ = np.repeat(lon_e0, lon.shape[1]).reshape(lon.shape)
                    lat_e_ = np.repeat(lat_e0, lat.shape[1]).reshape(lat.shape)

                    logic = (lat>=lat_s_) & (lat<=lat_e_)

                    mask2[logic] = 0
                else:
                    mask2[:, index_y-nline_stripe:index_y+nline_stripe] = 0
            #╰────────────────────────────────────────────────────────────────────────────╯#

            self.sensor[sensor0]['mask_central_h'] = (mask0 == 1) | (vza0 > viewing_zenith_angle_limit)
            self.sensor[sensor0]['mask_central_v'] = (mask1 == 1) | (vza0 > viewing_zenith_angle_limit)
            self.sensor[sensor0]['mask_forward_h'] = (mask2 == 1) | (vza0 > viewing_zenith_angle_limit)
            self.sensor[sensor0]['mask'] = ~(~self.sensor[sensor0]['mask_central_h'] | ~self.sensor[sensor0]['mask_central_v'] | ~self.sensor[sensor0]['mask_forward_h'])

    def add_adm_mask(self,
            sensors = None,
            mask = None,
            ):

        if sensors is None:
            sensors = self.sensor

        for sensor0 in sensors:

            vza0 = self.sensor[sensor0]['vza']
            vaa0 = self.sensor[sensor0]['vaa']

            lon, lat = cam.util.geolocate(vza0, vaa0, mask=mask, lon0=self.lon, lat0=self.lat, altitude=self.alt, heading=self.ang_hed, planet_radius=self.planet_radius)

            self.sensor[sensor0]['mask_adm'] = mask

    def geomap(self,
            which_mask=None,
            sensors=None,
            delta_latitude=None,
            ):

        if sensors is None:
            sensors = self.sensor

        for sensor0 in sensors:
            vza0 = self.sensor[sensor0]['vza']
            vaa0 = self.sensor[sensor0]['vaa']

            if which_mask is None:
                longitude, latitude = cam.util.geolocate(vza0, vaa0, mask=which_mask, lon0=self.lon, lat0=self.lat, altitude=self.alt, heading=self.ang_hed, planet_radius=self.planet_radius)
            else:
                longitude, latitude = cam.util.geolocate(vza0, vaa0, mask=self.sensor[sensor0][which_mask], lon0=self.lon, lat0=self.lat, heading=self.ang_hed, altitude=self.alt, planet_radius=self.planet_radius)

            self.sensor[sensor0]['longitude'] = longitude
            self.sensor[sensor0]['latitude']  = latitude

    def travel(self,
            delta_t=5.0,
            heading=None,
            speed=None,
            planet_rotation_period=86164.09053,
            planet_rotation_direction=90.0,
            ):

        if speed is None:
            speed = self.speed

        if heading is None:
            heading = self.ang_hed

        dist = speed * delta_t

        degree_travel = dist / (np.pi*(self.planet_radius+self.alt)) * 180.0
        degree_rotate = delta_t * 360.0 / planet_rotation_period

        self.lon += degree_travel * np.sin(np.deg2rad(heading)) - degree_rotate * np.sin(np.deg2rad(planet_rotation_direction))
        self.lat += degree_travel * np.cos(np.deg2rad(heading)) - degree_rotate * np.cos(np.deg2rad(planet_rotation_direction))
        self.dtime     += datetime.timedelta(seconds=delta_t)

    def divide_into_tiles(self,
            sensors=None,
            nx=24,
            ny=24
            ):

        if sensors is None:
            sensors = self.sensor

        self.tiles = {}
        for sensor0 in sensors:
            tiles0 = {}
            for vname0 in self.sensor[sensor0].keys():
                params0 = {}
                if self.sensor[sensor0][vname0].ndim == 2:
                    Nfpa_x, Nfpa_y = self.sensor[sensor0][vname0].shape
                    Ntile_x = Nfpa_x // nx
                    Ntile_y = Nfpa_y // ny
                    params0[vname0] = np.zeros((Ntile_x, Ntile_y), dtype=self.sensor[sensor0][vname0].dtype)
                    for i in range(Ntile_x):
                        for j in range(Ntile_y):
                            data0 = self.sensor[sensor0][vname0][nx*i:nx*(i+1), ny*j:ny*(j+1)]
                            params0[vname0][i, j] = np.nanmean(data0)

                    tiles0[vname0] = params0

            self.tiles[sensor0] = tiles0



if __name__ == '__main__':

    pass
