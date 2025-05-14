import sys

import numpy as np

import cam.common
import cam.util

__all__ = [
        'mask_dark',
        ]

_data_ = cam.util.cal_vza_vaa(cam.common.FPA_NX, cam.common.FPA_NY, cam.common.FPA_DX, cam.common.FPA_DY, coef_dist2ang=cam.common.COEF_DIST2ANG)



def mask_dark(
        mask=None,
        Nrow=4,
        ):

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    # mask dark
    #╭────────────────────────────────────────────────────────────────────────────╮#
    mask[:,  :Nrow] = 0
    mask[:, -Nrow:] = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    mask[_data_['vza']<=63.0] = 0

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    import matplotlib.pyplot as plt
    # import cartopy.crs as ccrs
    # mpl.use('Agg')

    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(mask.T, origin='lower', cmap='jet', interpolation='none', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
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

    return mask



if __name__ == '__main__':

    mask = mask_dark()
    pass
