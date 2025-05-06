import numpy as np

from libera_cam.constants import (
    VZA_LIMIT,
    PIXEL_COUNT_X,
    PIXEL_COUNT_Y,
    ADM_PIXEL_RADIUS,
    ADM_SAMPLE_PERCENT,
    DISTANCE_TO_ANGLE_COEFFICIENTS
)
_data_ = cam.util.cal_vza_vaa(cam.common.FPA_NX, cam.common.FPA_NY, cam.common.FPA_DX, cam.common.FPA_DY, coef_dist2ang=cam.common.COEF_DIST2ANG)


def mask_adm(
        mask=None,
        vza_shift=0, # shift of viewing zenith angle bin [deg]
        vaa_shift=0, # shift of viewing azimuth angle bin [deg]
        percent_samp=_percent_samp_,
        max_radius_of_pix=_max_radius_of_pix_,
        fpa_dx=cam.common.FPA_DX,
        fpa_dy=cam.common.FPA_DY,
        coef_ang2dist=cam.common.COEF_ANG2DIST,
        planet_radius=cam.common.PLANET_RADIUS,
        vza_bounds=np.arange(0.0,  90.1, 10.0), # equivalent to CERES-TRMM angular bins
        vaa_bounds=np.arange(0.0, 360.1, 20.0), # equivalent to CERES-TRMM angular bins
        alt=824000.0,
        n_vza_bins=9,
        n_vaa_bins=18,
        vza_limit=_vza_limit_,
        ):

    """
    Credits to original code:
        Created: 2 Feb 2024
        Latest update: 9 Feb 2024
        Pixel mask generator for Libera camera lock-on ADM samples, in barebones format
        @author: Jake Gristey
    """

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    x_center_pix = int(Nx//2)
    y_center_pix = int(Ny//2)

    # create array of angular bin numbers [VZA x VAA]
    n_bins = n_vza_bins * n_vaa_bins # total number of bins
    bin_nums_all = np.arange(n_bins).reshape((n_vza_bins, n_vaa_bins))

    # select the angular bins to sample, determined by "percent_samp"
    n_samples = int(percent_samp/100.0 * n_bins) # int always rounds down to be conservative for data rate
    selected_bin_nums = np.int_(n_bins/n_samples * bin_nums_all.ravel()) # evenly sample angular bins based on n_samples

    # loop over selected angular bins to identify corresponding camera pixels
    x_pix_selected = np.zeros(selected_bin_nums.size, dtype=np.int16)
    y_pix_selected = np.zeros(selected_bin_nums.size, dtype=np.int16)
    for i, bin_num in enumerate(selected_bin_nums):

        # select a VZA and VAA within each bin
        vza_i = bin_num // n_vaa_bins
        vaa_i = bin_num %  n_vaa_bins

        vza = vza_bounds[vza_i] + 0.5*(vza_bounds[vza_i+1]-vza_bounds[vza_i]) + vza_shift
        vaa = vaa_bounds[vaa_i] + 0.5*(vaa_bounds[vaa_i+1]-vaa_bounds[vaa_i]) + vaa_shift

        # if the shift moved the angle out of possible range, put back in range
        vza = vza % 90.0
        vaa = vaa % 360.0

        # convert VZA to satellite reference frame (spherical Earth assumption)
        vza_sat_ref = np.rad2deg(np.arcsin(np.sin(np.deg2rad(vza))*(planet_radius/(planet_radius+alt))))

        # find the closest corresponding pixel
        dist_from_center = np.polyval(coef_ang2dist,vza_sat_ref)

        x_dist_from_center = dist_from_center * np.sin(np.deg2rad(vaa))
        y_dist_from_center = dist_from_center * np.cos(np.deg2rad(vaa))

        x_pix_selected[i] = np.round(x_center_pix+x_dist_from_center/fpa_dx)
        y_pix_selected[i] = np.round(y_center_pix+y_dist_from_center/fpa_dy)

    # start by selecting a box around each central pixel
    for i in range(len(x_pix_selected)):

        x_pix_box=range(x_pix_selected[i]-max_radius_of_pix,x_pix_selected[i]+max_radius_of_pix+1)
        y_pix_box=range(y_pix_selected[i]-max_radius_of_pix,y_pix_selected[i]+max_radius_of_pix+1)

        # only keep pixels from the box that are within a circular radius of the central pixel
        for x in x_pix_box:
            for y in y_pix_box:
                radius_from_central_pix = np.sqrt((x-x_pix_selected[i])**2.0 + (y-y_pix_selected[i])**2.0)
                if (radius_from_central_pix <= (max_radius_of_pix+0.5)): # +0.5 buffer (closer to cam result)
                    if (x < x_center_pix*2.0) and (y < y_center_pix*2.0):
                        mask[x, y] = 0

    if vza_limit is not None:
        mask[_data_['vza']>vza_limit] = 1

    return mask



if __name__ == '__main__':

    mask = mask_adm()
    pass
