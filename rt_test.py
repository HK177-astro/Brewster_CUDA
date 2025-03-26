import numpy as np
import pickle
import math
from numba import cuda, jit
import time


with open("/home/hkothari/harshil/brewster/rt_input.pic", 'rb') as f:
    inputs = pickle.load(f, encoding='bytes')

clphotspec, othphotspec, cf, clphot, othphot, do_cf, inwavenum, do_clouds, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover = inputs

maxwave = 100000
nlayers = press.size
ngas = 8
nwave = inwavenum.size
nlinetemps = 27
npress = press.size

@cuda.jit
def set_temp_levels_kernel(leveltemp, press, temp):

    nlayers = 64
    logP = cuda.local.array(nlayers, dtype=np.float64)  # Adjust size based on nlayers

    for ilayer in range(nlayers):
        logP[ilayer] = math.log10(press[ilayer])

    ilayer = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if ilayer < nlayers:


        if ilayer == 0:
            # Handle the first layer
            p1 = math.exp(1.5 * math.log(press[ilayer]) - 0.5 * math.log(press[ilayer + 1]))
            p2 = math.exp(0.5 * math.log(press[ilayer] * press[ilayer + 1]))

            leveltemp[ilayer - 1] = temp[ilayer] + ((
                (temp[ilayer + 1] - temp[ilayer]) / (logP[ilayer + 1] - logP[ilayer])
            ) * (math.log10(p1) - logP[ilayer]))

            leveltemp[ilayer] = temp[ilayer] + ((
                (temp[ilayer + 1] - temp[ilayer]) / (logP[ilayer + 1] - logP[ilayer])
            ) * (math.log10(p2) - logP[ilayer]))

        elif ilayer == nlayers - 1:
            # Handle the last layer
            p1 = math.exp(0.5 * math.log(press[ilayer - 1] * press[ilayer]))
            p2 = (press[ilayer] ** 2) / p1

            leveltemp[ilayer] = temp[ilayer] + ((
                (temp[ilayer] - temp[ilayer - 1]) / (logP[ilayer] - logP[ilayer - 1])
            ) * (math.log10(p2) - logP[ilayer]))

        else:
            # Handle intermediate layers
            p2 = math.exp(0.5 * math.log(press[ilayer] * press[ilayer + 1]))

            leveltemp[ilayer] = temp[ilayer] + (
                (temp[ilayer + 1] - temp[ilayer]) / (logP[ilayer + 1] - logP[ilayer])
            )  * (math.log10(p2) - logP[ilayer])



@cuda.jit(device=True)
def bbplk(waven, T):
    C = 299792458.0
    kb = 1.38064852e-23
    h = 6.62607004e-34
    wavelen = 1.0e-6 * (1.0e4 / waven)
    bb_value = 1.0e-6 * ((2.0 * h * C ** 2) / wavelen ** 5) / (math.exp(h * C / (wavelen * kb * T)) - 1.0)
    return bb_value


@cuda.jit
def gfluxi_device(TEMP, TAU, W0, COSBAR, wavenum, RSF, fup, fdown):
    # mean ubar
    UBARI = 0.5

    nlayers = 64
    nlevels = 65
    ngaussi = 8

    # arrays
    B0 = cuda.local.array(64, dtype=np.float64)
    B1 = cuda.local.array(64, dtype=np.float64)
    ALPHA = cuda.local.array(64, dtype=np.float64)
    LAMDA = cuda.local.array(64, dtype=np.float64)
    GAMA = cuda.local.array(64, dtype=np.float64)
    CP = cuda.local.array(64, dtype=np.float64)
    CM = cuda.local.array(64, dtype=np.float64)
    CPM1 = cuda.local.array(64, dtype=np.float64)
    CMM1 = cuda.local.array(64, dtype=np.float64)
    E1 = cuda.local.array(64, dtype=np.float64)
    E2 = cuda.local.array(64, dtype=np.float64)
    E3 = cuda.local.array(64, dtype=np.float64)
    E4 = cuda.local.array(64, dtype=np.float64)
    g = cuda.local.array(64, dtype=np.float64)
    xj = cuda.local.array(64, dtype=np.float64)
    h = cuda.local.array(64, dtype=np.float64)
    xk = cuda.local.array(64, dtype=np.float64)
    alpha1 = cuda.local.array(64, dtype=np.float64)
    alpha2 = cuda.local.array(64, dtype=np.float64)
    sigma1 = cuda.local.array(64, dtype=np.float64)
    sigma2 = cuda.local.array(64, dtype=np.float64)

    fpt = cuda.local.array(65, dtype=np.float64)
    fmt = cuda.local.array(65, dtype=np.float64)
    em = cuda.local.array(65, dtype=np.float64)
    em2 = cuda.local.array(65, dtype=np.float64)
    em3 = cuda.local.array(65, dtype=np.float64)
    epp = cuda.local.array(65, dtype=np.float64)

    x = (0.0446339553, 0.1443662570, 0.2868247571, 0.4548133152, 0.6280678354, 0.7856915206, 0.9086763921, 0.9822200849)
    w = (0.0032951914, 0.0178429027, 0.0454393195, 0.0791995995, 0.1060473594, 0.1125057995, 0.0911190236, 0.0445508044)

    iflag = 0

    for j in range(nlayers):
        ALPHA[j] = math.sqrt((1.0 - W0[j]) / (1.0 - W0[j] * COSBAR[j]))
        LAMDA[j] = ALPHA[j] * (1.0 - W0[j] * COSBAR[j]) / UBARI
        GAMA[j] = (1.0 - ALPHA[j]) / (1.0 + ALPHA[j])
        term = 0.5 / (1.0 - W0[j] * COSBAR[j])

        B1[j] = (bbplk(wavenum, TEMP[j + 1]) - bbplk(wavenum, TEMP[j])) / TAU[j]
        B0[j] = bbplk(wavenum, TEMP[j])

        if (TAU[j] < 1e-6):
            B1[j] = 0.0
            B0[j] = 0.5 * (bbplk(wavenum, TEMP[j]) + bbplk(wavenum, TEMP[j + 1]))

        CP[j] = B0[j] + B1[j] * TAU[j] + B1[j] * term
        CM[j] = B0[j] + B1[j] * TAU[j] - B1[j] * term
        CPM1[j] = B0[j] + B1[j] * term
        CMM1[j] = B0[j] - B1[j] * term

    for j in range(nlayers):
        EP = math.exp(35.0)
        if LAMDA[j] * TAU[j] < 35.0:
            EP = math.exp(LAMDA[j] * TAU[j])
        EMM = 1.0 / EP
        E1[j] = EP + GAMA[j] * EMM
        E2[j] = EP - GAMA[j] * EMM
        E3[j] = GAMA[j] * EP + EMM
        E4[j] = GAMA[j] * EP - EMM

    TAUTOP = TAU[0]
    BTOP = (1.0 - math.exp(-TAUTOP / UBARI)) * bbplk(wavenum, TEMP[0])
    BSURF = bbplk(wavenum, TEMP[nlevels - 1])
    BOTTOM = BSURF + B1[nlayers - 1] * UBARI

    L = 2 * nlayers

    # Declare variables
    AF = cuda.local.array(128, dtype=np.float64)
    BF = cuda.local.array(128, dtype=np.float64)
    CF = cuda.local.array(128, dtype=np.float64)
    DF = cuda.local.array(128, dtype=np.float64)
    XK = cuda.local.array(128, dtype=np.float64)
    XK1 = cuda.local.array(nlayers, dtype=np.float64)
    XK2 = cuda.local.array(nlayers, dtype=np.float64)

    # Solve for AF, BF, CF, and DF
    AF[0] = 0.0
    BF[0] = GAMA[0] + 1.0
    CF[0] = GAMA[0] - 1.0
    DF[0] = BTOP - CMM1[0]

    # Replace slicing with explicit loops
    for i in range(1, L, 2):
        AF[i] = (E1[(i - 1) // 2] + E3[(i - 1) // 2]) * (GAMA[i // 2] - 1.0)
        BF[i] = (E2[(i - 1) // 2] + E4[(i - 1) // 2]) * (GAMA[i // 2] - 1.0)
        CF[i] = 2.0 * (1.0 - GAMA[i // 2] ** 2)
        DF[i] = (GAMA[i // 2] - 1.0) * (CPM1[i // 2] - CP[(i - 1) // 2]) + (1.0 - GAMA[i // 2]) * (CM[(i - 1) // 2] - CMM1[i // 2])

    for i in range(2, L, 2):
        AF[i] = 2.0 * (1.0 - GAMA[(i - 2) // 2] ** 2)
        BF[i] = (E1[(i - 2) // 2] - E3[(i - 2) // 2]) * (GAMA[i // 2] + 1.0)
        CF[i] = (E1[(i - 2) // 2] + E3[(i - 2) // 2]) * (GAMA[i // 2] - 1.0)
        DF[i] = E3[(i - 2) // 2] * (CPM1[i // 2] - CP[(i - 2) // 2]) + E1[(i - 2) // 2] * (CM[(i - 2) // 2] - CMM1[i // 2])

    AF[-1] = E1[-1] - RSF * E3[-1]
    BF[-1] = E2[-1] - RSF * E4[-1]
    CF[-1] = 0.0
    DF[-1] = BOTTOM - CP[-1] + RSF * CM[-1]

    NMAX = 301

    AS, DS, XK = cuda.local.array(128, dtype=np.float64), cuda.local.array(128, dtype=np.float64), cuda.local.array(128, dtype=np.float64)

    AS[-1] = AF[-1] / BF[-1]
    DS[-1] = DF[-1] / BF[-1]

    for i in range(L - 2, -1, -1):
        X = 1.0 / (BF[i] - CF[i] * AS[i + 1])
        AS[i] = AF[i] * X
        DS[i] = (DF[i] - CF[i] * DS[i + 1]) * X

    XK[0] = DS[0]
    for i in range(1, L):
        XK[i] = DS[i] - AS[i] * XK[i - 1]

    # Unmix the coefficients
    for i in range(0, nlayers, 2):
        XK1[i] = XK[2 * i] + XK[2 * i + 1]
        XK2[i] = XK[2 * i] - XK[2 * i + 1]

    for ng in range(ngaussi):
        ugauss = x[ng]
        for j in range(nlayers):
            if W0[j] >= 0.01:
                alphax = ((1 - W0[j]) / (1 - W0[j] * COSBAR[j])) ** 0.5

                g[j] = 2 * math.pi * W0[j] * XK1[j] * (1 + COSBAR[j] * alphax) / (1 + alphax)
                h[j] = 2 * math.pi * W0[j] * XK2[j] * (1 - COSBAR[j] * alphax) / (1 + alphax)
                xj[j] = 2 * math.pi * W0[j] * XK1[j] * (1 - COSBAR[j] * alphax) / (1 + alphax)
                xk[j] = 2 * math.pi * W0[j] * XK2[j] * (1 + COSBAR[j] * alphax) / (1 + alphax)

                alpha1[j] = 2 * math.pi * (B0[j] + B1[j] * (UBARI * W0[j] * COSBAR[j] / (1 - W0[j] * COSBAR[j])))
                alpha2[j] = 2 * math.pi * B1[j]
                sigma1[j] = 2 * math.pi * (B0[j] - B1[j] * (UBARI * W0[j] * COSBAR[j] / (1 - W0[j] * COSBAR[j])))
                sigma2[j] = alpha2[j]
            else:
                g[j] = 0.0
                h[j] = 0.0
                xj[j] = 0.0
                xk[j] = 0.0
                alpha1[j] = 2 * math.pi * B0[j]
                alpha2[j] = 2 * math.pi * B1[j]
                sigma1[j] = alpha1[j]
                sigma2[j] = alpha2[j]

        fpt[nlevels - 1] = 2.0 * math.pi * (BSURF + B1[nlayers - 1] * ugauss)
        fmt[0] = 2.0 * math.pi * (1.0 - math.exp(-TAUTOP / ugauss)) * bbplk(wavenum, TEMP[0])

        for j in range(nlayers):
            em[j] = math.exp(-LAMDA[j] * TAU[j])
            em2[j] = math.exp(-TAU[j] / ugauss)
            em3[j] = em[j] * em2[j]
            epp[j] = math.exp(35.0)
            obj = LAMDA[j] * TAU[j]
            if obj < 35.0:
                epp[j] = math.exp(obj)

            fmt[j + 1] = fmt[j] * em2[j] + xj[j] / (LAMDA[j] * ugauss + 1.0) * (epp[j] - em2[j]) + xk[j] / (
                    LAMDA[j] * ugauss - 1.0) * (em2[j] - em[j]) + sigma1[j] * (1.0 - em2[j]) + sigma2[j] * (
                                 ugauss * em2[j] + TAU[j] - ugauss)

        for j in range(nlayers - 1, -1, -1):
            fpt[j] = fpt[j + 1] * em2[j] + (g[j] / (LAMDA[j] * ugauss - 1.0)) * (epp[j] * em2[j] - 1.0) + (
                    h[j] / (LAMDA[j] * ugauss + 1.0)) * (1.0 - em3[j]) + alpha1[j] * (1.0 - em2[j]) + alpha2[
                         j] * (ugauss - (TAU[j] + ugauss) * em2[j])

        fup[0] = fup[0] + w[ng] * fpt[0]

    return fup[0]

@cuda.jit
def compute_flux_kernel(temper, opd_ext, DTAUC, SSALB, COSBAR, inwavenum, ALBEDO, gflup, fdi, upflux, pcover, spectrum):
    iwave = cuda.blockIdx.x  # Each block handles one wavelength
    ilayer = cuda.threadIdx.x  # Each thread handles one layer

    # Ensure we don't go out of bounds
    if iwave < opd_ext.shape[1] and ilayer < opd_ext.shape[0]:
        # Put optical depth into the right variable for radtran
        DTAUC[ilayer] = opd_ext[ilayer, iwave]

    # Synchronize threads to ensure DTAUC is fully updated
    cuda.syncthreads()

    # Only the first thread in the block computes the flux
    if ilayer == 0:
        gflup[0] = gfluxi_device(temper, DTAUC, SSALB, COSBAR, inwavenum[iwave], ALBEDO, gflup, fdi)  # Call subroutine
        upflux[iwave] = gflup[0]
        spectrum[iwave] += upflux[iwave] #* pcover

@cuda.jit
def run_RT_kernel(clphotspec, othphotspec, cf, clphot, othphot, do_cf, inwavenum, cloudy, press, temp, temper, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover, spectrum, upflux, DTAUC, SSALB, COSBAR, gflup, fdi):
    # Thread indices
    iwave = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ilayer = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    nlayers = 64
    nlevel = nlayers + 1
    cldone = 0
    othdone = 0

    # DTAUC = cuda.local.array(64, dtype=np.float64)
    # SSALB = cuda.local.array(64, dtype=np.float64)
    # COSBAR = cuda.local.array(64, dtype=np.float64)
    # gflup = cuda.local.array(65, dtype=np.float64)
    # fdi = cuda.local.array(65, dtype=np.float64)

    if iwave < nwave and ilayer < nlayers:
        # Local arrays

        ALBEDO = 0.000000001

        # Compute optical depths
        tau_cloud = opd_ext[ilayer, iwave]
        tau_others = opd_lines[ilayer, iwave] + opd_CIA[ilayer, iwave] + opd_rayl[ilayer, iwave] + opd_hmbff[ilayer, iwave]

        opd_ext[ilayer, iwave] = opd_ext[ilayer, iwave] + opd_lines[ilayer, iwave] + opd_CIA[ilayer, iwave] + opd_rayl[ilayer, iwave] + opd_hmbff[ilayer, iwave]
        opd_scat[ilayer, iwave] = opd_scat[ilayer, iwave] + opd_rayl[ilayer, iwave]

        # Compute SSALB and COSBAR
        if cloudy != 0:
            SSALB[ilayer] = opd_scat[ilayer, iwave] / opd_ext[ilayer, iwave]
            COSBAR[ilayer] = gg[ilayer, iwave]
        else:
            SSALB[ilayer] = opd_scat[ilayer, iwave] / opd_ext[ilayer, iwave]
            COSBAR[ilayer] = 0.0

        # Set reference tau for cloud taup_cl and others taup_oth
        taup_cl = 1.0
        taup_oth = 1.0

        # Diagnostics for photospheres
        if clphot and not cldone:
            # Compute sum of tau_cloud up to ilayer
            total_tau_cloud = 0.0
            for k in range(ilayer):
                total_tau_cloud += opd_ext[k, iwave]  # Sum over all layers up to ilayer

            if total_tau_cloud > taup_cl:
                cldone = 1
                tau2 = total_tau_cloud
                tau1 = tau2 - opd_ext[ilayer - 1, iwave]

                if ilayer == nlayers:
                    p1 = math.exp(0.5 * math.log(press[ilayer - 1] * press[ilayer]))
                else:
                    p1 = math.exp(1.5 * math.log(press[ilayer]) - 0.5 * math.log(press[ilayer + 1]))

                clphotspec[iwave] = p1 + (taup_cl - tau1) * dp[ilayer] / opd_ext[ilayer, iwave]

        if othphot and not othdone:
            # Compute sum of tau_others up to ilayer
            total_tau_others = 0.0
            for k in range(ilayer):
                total_tau_others += (opd_lines[k, iwave] + opd_CIA[k, iwave] + opd_rayl[k, iwave] + opd_hmbff[k, iwave])  # Sum over all layers up to ilayer

            if total_tau_others > taup_oth:
                othdone = 1
                tau2 = total_tau_others
                tau1 = tau2 - (opd_lines[ilayer, iwave] + opd_CIA[ilayer, iwave] + opd_rayl[ilayer, iwave] + opd_hmbff[ilayer, iwave])

                if ilayer == nlayers:
                    p1 = math.exp(0.5 * math.log(press[ilayer - 1] * press[ilayer]))
                else:
                    p1 = math.exp(1.5 * math.log(press[ilayer]) - 0.5 * math.log(press[ilayer + 1]))

                othphotspec[iwave] = p1 + (taup_oth - tau1) * dp[ilayer] / (opd_lines[ilayer, iwave] + opd_CIA[ilayer, iwave] + opd_rayl[ilayer, iwave] + opd_hmbff[ilayer, iwave])


        DTAUC = opd_ext[:, iwave]
        # gflup[0] = gfluxi_device(temper, DTAUC, SSALB, COSBAR, inwavenum[iwave], ALBEDO, gflup, fdi)
        # upflux[iwave] = gflup[0]
        #
        #
        # spectrum[iwave] += upflux[iwave] * pcover  # Ensure pcover is indexed correctly


def rt(clphotspec, othphotspec, cf, clphot, othphot, do_cf, inwavenum, do_clouds, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover):
    # Allocate device memory
    clphotspec_d = cuda.to_device(clphotspec)
    othphotspec_d = cuda.to_device(othphotspec)
    cf_d = cuda.to_device(cf)
    inwavenum_d = fHe_d = cuda.to_device(inwavenum)
    press_d = cuda.to_device(press)
    temp_d = cuda.to_device(temp)
    gg_d = cuda.to_device(gg)
    dp_d = cuda.to_device(dp)
    dz_d = cuda.device_array(nlayers, dtype=np.float64)
    ndens_d = cuda.device_array(nlayers, dtype=np.float64)

    opd_ext_d = cuda.to_device(opd_ext)
    opd_CIA_d = cuda.to_device(opd_CIA)
    opd_rayl_d = cuda.to_device(opd_rayl)
    opd_lines_d = cuda.to_device(opd_lines)
    opd_scat_d = cuda.to_device(opd_scat)
    opd_hmbff_d = cuda.to_device(opd_hmbff)

    pcover_d = 1
    do_clouds_d = 0

    spectrum_d = cuda.device_array(nwave, dtype=np.float64)
    upflux_d = cuda.device_array(nwave, dtype=np.float64)

    leveltemp_d = cuda.device_array(65, dtype=np.float64)
    temper_d = cuda.device_array(65, dtype=np.float64)

    threads_per_block = 64
    blocks_per_grid = (nlayers + threads_per_block - 1) // threads_per_block

    set_temp_levels_kernel[blocks_per_grid, threads_per_block](leveltemp_d, press_d, temp_d)

    temper_d[0] = leveltemp_d[-1]
    temper_d[1:] =  leveltemp_d[0:-1]

    # Define thread and block dimensions
    threads_per_block = (16, 16)  # Example: 16x16 threads per block
    blocks_per_grid_x = (nwave + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (nlayers + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    DTAUC = cuda.device_array(64, dtype=np.float64)
    SSALB = cuda.device_array(64, dtype=np.float64)
    COSBAR = cuda.device_array(64, dtype=np.float64)
    gflup = cuda.device_array(65, dtype=np.float64)
    fdi = cuda.device_array(65, dtype=np.float64)

    # Launch the kernel
    run_RT_kernel[blocks_per_grid, threads_per_block](
        clphotspec_d, othphotspec_d, cf_d, clphot, othphot, do_cf, inwavenum_d, do_clouds_d, press, temp, temper_d, nwave, opd_ext_d, opd_lines_d,
        opd_CIA_d, opd_rayl_d, opd_scat_d, opd_hmbff_d, gg_d, dp_d, pcover_d, spectrum_d, upflux_d, DTAUC, SSALB, COSBAR, gflup, fdi)

    threads_per_block = nlayers
    blocks_per_grid = nwave

    ALBEDO = 0.000000001

    # compute_flux_kernel[blocks_per_grid, threads_per_block](temper_d, opd_ext_d, DTAUC, SSALB, COSBAR, inwavenum_d, ALBEDO, gflup, fdi, upflux_d, pcover_d, spectrum_d)

    #     DTAUC = opd_ext_d[:, iwave]
    #     gflup[0] = gfluxi_device(temper, DTAUC, SSALB, COSBAR, inwavenum[iwave], ALBEDO, gflup, fdi)
    #     upflux[iwave] = gflup[0]
    #
    #
    #     spectrum[iwave] += upflux[iwave] * pcover

    upflux = DTAUC.copy_to_host()
    spectrum = opd_ext_d.copy_to_host()

    return upflux, spectrum

upflux, spectrum = rt(clphotspec, othphotspec, cf, clphot, othphot, do_cf, inwavenum, do_clouds, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover)

print("DTAUC : ", upflux)
print("opd_ext_d : ", spectrum)
