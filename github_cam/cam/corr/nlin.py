import os
import sys
import glob
import tqdm
import datetime
import numpy as np


import cam.common
import cam.util


__all__ = [
        'get_nlin_factor',
        'get_nlin_factor_syn_r',
        'get_nlin_factor_syn',
        ]

def get_pix_dep(
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        ):

    # pixel dependency
    #╭────────────────────────────────────────────────────────────────────────────╮#
    x1d = np.arange(Nx)
    y1d = np.arange(Ny)
    x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')
    xc = Nx // 2
    yc = Ny // 2
    pix_dep = 0.9 + 0.2 * (1.0 - ((x2d-xc)**2.0 + (y2d-yc)**2.0)/(((xc+yc)/2.0)**2.0))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return pix_dep

def nlin_func(x,
        pix_dep=None,
        ):

    if pix_dep is None:
        pix_dep = np.ones_like(x, dtype=np.float32)

    y = 0.8*pix_dep*(2.0*x - np.tan(0.8*pix_dep*x))

    return y

def get_nlin_factor(
        cnt_true,
        ):

    scale_factor = get_nlin_factor_syn_r(cnt_true)

    return scale_factor

def get_nlin_factor_syn_r(
        cnt_true,
        Nbit=cam.common.NBIT,
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        ):

    true = cnt_true / (2.0**Nbit)

    pix_dep = get_pix_dep()

    nlin = nlin_func(true, pix_dep=pix_dep)

    scale_factor = true / nlin

    return scale_factor

def get_nlin_factor_syn(
        cnt_nlin,
        fname=None,
        Nbit=cam.common.NBIT,
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        ):

    # get nonlinearity factor from polynomial coefficients
    #╭──────────────────────────────────────────────────────────────╮#
    nlin = cnt_nlin / (2.0**Nbit)

    if fname is not None:
        cam.common.DATA_COEF['nlin'] = \
                cam.util.load_h5(fname)['coef']
    else:
        if 'nlin' not in cam.common.DATA_COEF.keys():
            cam.common.DATA_COEF['nlin'] = \
                    cam.util.load_h5(
                            '%s/data_nlin_poly_coef_syn.h5' % \
                            cam.common.FDIR_DATA)['coef']

    coef = cam.common.DATA_COEF['nlin']
    #╰──────────────────────────────────────────────────────────────╯#

    true = cam.util.polyval(coef, nlin)

    scale_factor = true / nlin

    return scale_factor

def cdata_nlin_poly_coef_syn(
        fname='%s/data_nlin_poly_coef_syn.h5' % cam.common.FDIR_DATA,
        Nbit=cam.common.NBIT,
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        Ndeg=5,
        ):

    pix_dep = get_pix_dep()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.path as mpl_path
        import matplotlib.image as mpl_img
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        from matplotlib import rcParams, ticker
        from matplotlib.ticker import FixedLocator
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # import cartopy.crs as ccrs
        # mpl.use('Agg')
        plt.close('all')
        fig = plt.figure(figsize=(4, 10))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        xx = np.linspace(0.0, 1.0)
        # yy = nlin_func(xx, pix_dep=None)

        ax1 = fig.add_subplot(211, aspect='equal')
        pix_deps = np.linspace(0.9, 1.1, 512)
        colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, pix_deps.size))
        for i in range(pix_deps.size):
            yy = nlin_func(xx, pix_deps[i])
            ax1.plot(xx, yy, lw=0.1, color=colors[i, ...])
        ax1.plot([0, 1], [0, 1], color='gray', ls='--')

        ax2 = fig.add_subplot(212)
        cs = ax2.imshow(pix_dep.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        ax1.set_xlim((0, 1))
        ax1.set_ylim((0, 1))
        ax1.set_xlabel('True (Normalized DN)')
        ax1.set_ylabel('Obs. (Normalized DN)')
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # polynomial fit
    #╭────────────────────────────────────────────────────────────────────────────╮#
    poly_coef = np.zeros((Nx, Ny, Ndeg+1), dtype=np.float32)
    xx = np.linspace(0.0, 1.0, 1000)
    for i in tqdm.tqdm(range(Nx), desc='Outer Loop'):
        for j in tqdm.tqdm(range(Ny), desc='Inner Loop', leave=False):
            yy = nlin_func(xx, pix_dep=pix_dep[i, j])
            poly_coef0 = np.polyfit(yy, xx, deg=Ndeg)
            poly_coef[i, j, :] = poly_coef0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # save coefficients
    #╭────────────────────────────────────────────────────────────────────────────╮#
    import h5py
    f = h5py.File(fname, 'w')
    f.create_dataset('coef', data=poly_coef, compression='gzip', compression_opts=9, chunks=True)
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return


if __name__ == '__main__':

    cdata_nlin_poly_coef_syn()

    pass
