from numba import cuda
import numpy as np
from src.brewster.brewster import layer_thickness_kernel


def test_layer_thickness(inputs, kernel_config):

    dp = cuda.device_array(inputs['nlayers'], dtype=np.float64)
    dz = cuda.device_array(inputs['nlayers'], dtype=np.float64)

    threads_per_block, blocks_per_grid = kernel_config
    layer_thickness_kernel[blocks_per_grid, threads_per_block](
        inputs['grav'], 
        inputs['tot_VMR_molmass'], 
        inputs['mu'], 
        inputs['fH'], 
        inputs['fHmin'], 
        inputs['fH2'], 
        inputs['fHe'], 
        inputs['nlayers'], 
        inputs['press'], 
        inputs['temp'], 
        dp, 
        dz)
        #grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp, dp, dz

    dp = dp.copy_to_host()
    assert dp.all() > 0
    assert np.isclose(dp.mean(), 3.4980)
    assert np.isclose(dp.max(), 46.044158)
    assert np.isclose(dp.min(), 2.3076757e-05)
    assert np.diff(dp).all()  # should be increasing 

    assert dz.all() > 0
    assert np.isclose(dz.mean(), 2187.457)
    assert np.isclose(dz.max(), 3866.671)
    assert np.isclose(dz.min(), 493.308)

# spectrum = forward(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp, linelist, inlinetemps, ngas, nwave, nlinetemps, VMR, ch4index, wavelen, pcover, clphot, othphot, do_cf, clphotspec, othphotspec, cf, inwavenum)
# print("spectrum : ", spectrum)
