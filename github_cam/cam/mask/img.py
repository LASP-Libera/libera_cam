import sys

import numpy as np

import cam.common
import cam.util

__all__ = [
        'mask_c_stripe_ss',
        'mask_c_stripe',
        'mask_v_stripe',
        'mask_f_stripe',
        'mask_img',
        ]

_data_ = cam.util.cal_vza_vaa(cam.common.FPA_NX, cam.common.FPA_NY, cam.common.FPA_DX, cam.common.FPA_DY, coef_dist2ang=cam.common.COEF_DIST2ANG)
_nline_stripe_      = 21
_crop_delta_degree_ = None
_delta_vza_         = 30.0
_straighten_        = False
_vza_limit_         = cam.common.VZA_LIMIT


def mask_c_stripe_ss(
        sr=0,    # stripe rotation angle (in degrees)
        su=50,   # stripe add-on + (to generate non-zero width)
        sd=50,   # stripe add-on - (to generate non-zero width)
        fr=1000, # radius of FOV to be populated with pixels
        sy=0,    # y distance of stripe from origin (FPA center)
        plot=True,
        fill=True,
        xp=cam.common.FPA_NX,
        yp=cam.common.FPA_NY,
        ):

    xc = int(xp//2) # center pixel with respect to FPA coordinate system
    yc = int(yp//2)

    # create empty index array as spaceholder - initialized with -1
    xi = np.zeros((xp,1+su+sd))-1
    yi = np.zeros((xp,1+su+sd))-1

    # populate spaceholder with pixel mask data
    # start with generator line
    xg = np.arange(xc-(fr),xc+(fr+1),1)
    xi[np.int_(xg),0] = xg
    yi[np.int_(xg),0] = yc+sy
    pop=1
    # now positive add-on (stripe width)
    if su>0:
        for i in range(su):
           xi[np.int_(xg),pop] = xg
           yi[np.int_(xg),pop] = yc+sy+i+1
           pop=pop+1


    # now negative add-on (stripe width)
    if sd>0:
        for i in range(sd):
            xi[np.int_(xg),pop] = xg
            yi[np.int_(xg),pop] = yc+sy-(i+1)
            pop=pop+1


    # Now rotation of the whole object about the prescribed angle
    ct = np.cos(sr*np.pi/180.) # cosine of angle
    st = np.sin(sr*np.pi/180.) # sine of angle
    R  = np.array([[ct,-st],[st,ct]])

    xr=np.zeros_like(xi)-1
    yr=np.zeros_like(yi)-1
    for i in range(pop):
         v0=np.array([xi[:,i]-xc,yi[:,i]-yc])
         vt = np.matmul(R,v0)
         xr[:,i]=vt[0,:]+xc
         yr[:,i]=vt[1,:]+yc
         flt=np.where(np.power(xr[:,i]-xc,2) + np.power(yr[:,i]-yc,2)>fr**2)
         xr[flt,i]=-1
         yr[flt,i]=-1


    mask = np.zeros((xp,yp))
    flt=np.where(xr>=0)
    xr=xr[flt]
    yr=yr[flt]
    n =len(xr)

    for i in range(n):
        ii=xr[i]
        jj=yr[i]
        i_=round(ii)
        j_=round(jj)
        if (mask[i_,j_]==0): # if mask is still empty in that location
            mask[i_,j_]=1
        else:
            if fill:
                idx=np.array([[i_+1,j_],[i_-1,j_],[i_,j_-1],[i_,j_+1],[i_-1,j_-1],[i_+1,j_+1],[i_-1,j_+1],[i_+1,j_-1]])
                flt=np.where((idx[:,0]>=0) & (idx[:,1]>=0) & (idx[:,0]<xp) & (idx[:,1]<yp))
                idx=idx[flt]
                nx =len(idx)
                if nx>0: mask[idx[:,0],idx[:,1]]=1

    return mask

def mask_c_stripe_hc(
        ang_rot=0,    # stripe rotation angle (in degrees)
        su=50,   # stripe add-on + (to generate non-zero width)
        sd=50,   # stripe add-on - (to generate non-zero width)
        fr=1000, # radius of FOV to be populated with pixels
        sy=0,    # y distance of stripe from origin (FPA center)
        fill=True,
        xp=cam.common.FPA_NX,
        yp=cam.common.FPA_NY,
        ):

    # center pixel with respect to FPA coordinate system
    #╭────────────────────────────────────────────────────────────────────────────╮#
    xc = int(xp//2)
    yc = int(yp//2)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # create empty index array as spaceholder - initialized with -1
    xi = np.zeros((xp,1+su+sd))-1
    yi = np.zeros((xp,1+su+sd))-1

    # populate spaceholder with pixel mask data
    # start with generator line
    xg = np.arange(xc-(fr),xc+(fr+1),1)
    xi[np.int_(xg),0] = xg
    yi[np.int_(xg),0] = yc+sy
    pop=1
    # now positive add-on (stripe width)
    if su>0:
        for i in range(su):
           xi[np.int_(xg),pop] = xg
           yi[np.int_(xg),pop] = yc+sy+i+1
           pop=pop+1


    # now negative add-on (stripe width)
    if sd>0:
        for i in range(sd):
            xi[np.int_(xg),pop] = xg
            yi[np.int_(xg),pop] = yc+sy-(i+1)
            pop=pop+1


    # Now rotation of the whole object about the prescribed angle
    ct = np.cos(np.deg2rad(ang_rot)) # cosine of angle
    st = np.sin(np.deg2rad(ang_rot)) # sine of angle
    R  = np.array([[ct,-st],[st,ct]])

    xr=np.zeros_like(xi)-1
    yr=np.zeros_like(yi)-1
    for i in range(pop):
         v0=np.array([xi[:,i]-xc,yi[:,i]-yc])
         vt = np.matmul(R,v0)
         xr[:,i]=vt[0,:]+xc
         yr[:,i]=vt[1,:]+yc
         flt=np.where(np.power(xr[:,i]-xc,2) + np.power(yr[:,i]-yc,2)>fr**2)
         xr[flt,i]=-1
         yr[flt,i]=-1


    mask = np.zeros((xp,yp))
    flt=np.where(xr>=0)
    xr=xr[flt]
    yr=yr[flt]
    n =len(xr)

    for i in range(n):
        ii=xr[i]
        jj=yr[i]
        i_=round(ii)
        j_=round(jj)
        if (mask[i_,j_]==0): # if mask is still empty in that location
            mask[i_,j_]=1
        else:
            if fill:
                idx=np.array([[i_+1,j_],[i_-1,j_],[i_,j_-1],[i_,j_+1],[i_-1,j_-1],[i_+1,j_+1],[i_-1,j_+1],[i_+1,j_-1]])
                flt=np.where((idx[:,0]>=0) & (idx[:,1]>=0) & (idx[:,0]<xp) & (idx[:,1]<yp))
                idx=idx[flt]
                nx =len(idx)
                if nx>0: mask[idx[:,0],idx[:,1]]=1

    return mask

def mask_c_stripe(
        mask=None,
        nline_stripe=21,
        crop_delta_degree=None,
        vza_limit=62.0,
        ):

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    index_x_center = int(Nx//2)
    index_y_center = int(Ny//2)

    # central horizontal stripe
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if crop_delta_degree is not None:

        # lon0 = lon[:, index_y_center-nline_stripe:index_y_center+nline_stripe]
        # lat0 = lat[:, index_y_center-nline_stripe:index_y_center+nline_stripe]
        # lon_c = np.repeat(np.nanmean(lon0[:, [nline_stripe-1, nline_stripe]], axis=1), lon0.shape[1]).reshape(lon0.shape)
        # lat_c = np.repeat(np.nanmean(lat0[:, [nline_stripe-1, nline_stripe]], axis=1), lat0.shape[1]).reshape(lat0.shape)
        # logic = np.sqrt((lon0-lon_c)**2 + (lat0-lat_c)**2) <= crop_delta_degree
        # mask[:, index_y_center-nline_stripe:index_y_center+nline_stripe][logic] = 0

        msg = 'Error [cam.mask.mask_c_stripe]: <crop_delta_degree=> has not been implemented yet.'
        raise OSError(msg)

    else:

        mask[:, index_y_center-nline_stripe:index_y_center+nline_stripe] = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if vza_limit is not None:
        mask[_data_['vza']>vza_limit] = 1

    return mask

def mask_v_stripe(
        mask=None,
        nline_stripe=21,
        crop_delta_degree=None,
        vza_limit=62.0,
        ):

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    index_x_center = int(Nx//2)
    index_y_center = int(Ny//2)

    # central horizontal stripe
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if crop_delta_degree is not None:

        # lon0 = lon[index_x_center-nline_stripe:index_x_center+nline_stripe, :]
        # lat0 = lat[index_x_center-nline_stripe:index_x_center+nline_stripe, :]
        # lon_c = np.repeat(np.nanmean(lon0[[nline_stripe-1, nline_stripe], :], axis=0), lon0.shape[0]).reshape(lon0.shape)
        # lat_c = np.repeat(np.nanmean(lat0[[nline_stripe-1, nline_stripe], :], axis=0), lat0.shape[0]).reshape(lat0.shape)
        # logic = np.sqrt((lon0-lon_c)**2 + (lat0-lat_c)**2) <= crop_delta_degree
        # mask[index_x_center-nline_stripe:index_x_center+nline_stripe, :][logic] = 0

        msg = 'Error [cam.mask.mask_v_stripe]: <crop_delta_degree=> has not been implemented yet.'
        raise OSError(msg)

    else:

        mask[index_x_center-nline_stripe:index_x_center+nline_stripe, :] = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if vza_limit is not None:
        mask[_data_['vza']>vza_limit] = 1

    return mask

def mask_f_stripe(
        mask=None,
        nline_stripe=21,
        delta_vza=30.0,
        straighten=False,
        vza_limit=62.0,
        ):

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    index_x_center = int(Nx//2)
    index_y_center = int(Ny//2)
    index_y_pick   = np.argmin(np.abs(_data_['vza'][index_x_center, index_y_center:]-delta_vza)) + index_y_center

    # forward stripe
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if straighten:
        # delta_lon0 = lon[self.sensor[sensor0]['fpa_cx'], index_y] - lon[self.sensor[sensor0]['fpa_cx'], self.sensor[sensor0]['fpa_cy']]
        # delta_lat0 = lat[self.sensor[sensor0]['fpa_cx'], index_y] - lat[self.sensor[sensor0]['fpa_cx'], self.sensor[sensor0]['fpa_cy']]
        # degree0 = np.arctan(delta_lat0/delta_lon0)

        # lon_s0 = lon_c[:, 0] + delta_lon0 - np.abs(crop_delta_degree * np.cos(degree0))
        # lon_e0 = lon_c[:, 0] + delta_lon0 + np.abs(crop_delta_degree * np.cos(degree0))

        # lat_s0 = lat_c[:, 0] + delta_lat0 - np.abs(crop_delta_degree * np.sin(degree0))
        # lat_e0 = lat_c[:, 0] + delta_lat0 + np.abs(crop_delta_degree * np.sin(degree0))

        # lon_s_ = np.repeat(lon_s0, lon.shape[1]).reshape(lon.shape)
        # lat_s_ = np.repeat(lat_s0, lat.shape[1]).reshape(lat.shape)
        # lon_e_ = np.repeat(lon_e0, lon.shape[1]).reshape(lon.shape)
        # lat_e_ = np.repeat(lat_e0, lat.shape[1]).reshape(lat.shape)

        # logic = (lat>=lat_s_) & (lat<=lat_e_)

        # mask[logic] = 0
        msg = 'Error [cam.mask.mask_f_stripe]: <straighten=True> has not been implemented yet.'
        raise OSError(msg)
    else:
        mask[:, index_y_pick-nline_stripe:index_y_pick+nline_stripe] = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if vza_limit is not None:
        mask[_data_['vza']>vza_limit] = 1

    return mask



def mask_img(
        mask=None,
        nline_stripe=_nline_stripe_,
        crop_delta_degree=_crop_delta_degree_,
        delta_vza=_delta_vza_,
        straighten=_straighten_,
        vza_limit=_vza_limit_,
        ):

    mask = mask_c_stripe(mask=mask, nline_stripe=nline_stripe, vza_limit=vza_limit)
    mask = mask_v_stripe(mask=mask, nline_stripe=nline_stripe, vza_limit=vza_limit)
    mask = mask_f_stripe(mask=mask, nline_stripe=nline_stripe, vza_limit=vza_limit)

    return mask



if __name__ == '__main__':

    import time
    time0 = time.time()
    mask1 = mask_c_stripe_ss()
    time1 = time.time()
    print(time1-time0)
    mask2 = mask_c_stripe_hc()
    time2 = time.time()
    print(time2-time1)

    x = np.arange(2048)
    y = np.arange(2048)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    mask = mask_f_stripe(vza_limit=63.0)

    indices_y = np.where(mask[1024, :]==0)[0]
    for i, index_y in enumerate(indices_y):
        logic_x = (mask[:, index_y]==0)
        print(i, xx[logic_x, index_y])

    sys.exit()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        import matplotlib.pyplot as plt
        # import cartopy.crs as ccrs
        # mpl.use('Agg')

        plt.close('all')
        fig = plt.figure(figsize=(12, 4))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131)
        cs = ax1.imshow(mask1.T, origin='lower', cmap='jet', zorder=0, interpolation='none') #, extent=extent, vmin=0.0, vmax=0.5)

        ax2 = fig.add_subplot(132)
        cs = ax2.imshow(mask2.T, origin='lower', cmap='jet', zorder=0, interpolation='none') #, extent=extent, vmin=0.0, vmax=0.5)

        ax3 = fig.add_subplot(133)
        cs = ax3.imshow((mask2-mask1).T, origin='lower', cmap='jet', zorder=0, interpolation='none') #, extent=extent, vmin=0.0, vmax=0.5)
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
    pass
