import numpy as np
import pickle
import math
from numba import cuda, jit
import time

with open("/home/hkothari/harshil/brewster/forward_model_input.pic", 'rb') as f:
    inputs = pickle.load(f, encoding='bytes')

temp, logg, R2D2, gasnum, logVMR, pcover, do_clouds, cloudnum, cloudrad, cloudsig, cloudprof, inlinetemps, press, inwavenum, linelist, cia, ciatemps, clphot, ophot, make_cf, do_bff, bff = inputs

temp, logg, R2D2, gasnum, logVMR, pcover, do_clouds, cloudnum,cloudrad, cloudsig, cloudprof, inlinetemps, press, inwavenum, linelist, cia, ciatemps, clphot, othphot,do_cf, bfing, bff = temp, logg, R2D2, gasnum, logVMR, pcover, do_clouds, cloudnum, cloudrad, cloudsig, cloudprof, inlinetemps, press, inwavenum, linelist, cia, ciatemps, clphot, ophot, make_cf, do_bff, bff

maxwave = 100000
nlayers = press.size
ngas = gasnum.size
nwave = inwavenum.size
nlinetemps = inlinetemps.size
nclouds = cloudnum.size
npress = press.size
# npatches = do_clouds.size

# cloudname = np.empty((nclouds), dtype='U15')
# cl_phot_press = np.zeros((npatches, maxwave))
# oth_phot_press = np.zeros((npatches, maxwave))
out_spec = np.empty((2, nwave))
clphotspec = np.empty((maxwave))
othphotspec = np.empty((maxwave))
cf = np.empty((npress, nwave, nclouds))
# cfunc = np.zeros((npatches, maxwave, maxlayers))

ndens = np.zeros((nlayers))

opd_scat = np.zeros((nlayers, nwave))
gg = np.zeros((nlayers, nwave))
opd_CIA = np.zeros((nlayers, nwave))
opd_ext = np.zeros((nlayers, nwave))
opd_lines = np.zeros((nlayers, nwave))
opd_rayl = np.zeros((nlayers, nwave))
opd_hmbff = np.zeros((nlayers, nwave))

fe = np.zeros((nlayers))
fH = np.zeros((nlayers))
fHmin = np.zeros((nlayers))
fH2 = np.zeros((nlayers))
fHe = np.zeros((nlayers))
mu = np.zeros((nlayers))
tol_VMR = np.zeros((ngas))
tot_molmass = np.zeros((ngas))


wavelen = 1e4 / inwavenum
grav = 10. ** (logg) / 100.

with open("gaslist.dat", "r") as file:
    maxgas = int(file.readline())
    gaslist = []
    masslist = []
    for igas in range(maxgas):
        line = file.readline().split()
        idum1 = int(line[0])
        gas = line[1]
        mass = float(line[2])
        gaslist.append(gas)
        masslist.append(mass)

gasname = []
molmass = []

for igas in range(ngas):
    gasname.append(gaslist[gasnum[igas] - 1].strip())
    molmass.append(masslist[gasnum[igas] - 1])

# with open("cloudlist.dat", "r") as file:
#     maxcloud = int(file.readline())
#     cloudlist = []
#     for icloud in range(maxcloud):
#         line = file.readline().split()
#         idum2 = int(line[0])
#         cloud = line[1]
#         cloudlist.append(cloud)


# for icloud in range(nclouds):
#     if cloudnum[icloud] > 50:
#         cloudname[icloud] = "mixto"
#     else:
#         cloudname[icloud] = cloudlist[cloudnum[icloud] - 1].strip()


ch4index = 0

VMRname = np.empty((ngas, nlayers), dtype=list)
VMR = np.zeros((ngas, nlayers))
molmass_layered = np.zeros((ngas, nlayers))

for igas in range(ngas):
    for ilayer in range(nlayers):
        VMRname[igas, ilayer] = gasname[igas].strip()
        VMR[igas, ilayer] = 10. ** (logVMR[igas, ilayer])
        molmass_layered[igas, ilayer] = molmass[igas]

if (VMRname[igas, 0] == "ch4"):
    ch4index = igas


if (bfing):
    fe[0] = 10. ** bff[0, :]
    fH[0] = 10. ** bff[1, :]
    fHmin[0] = 10. ** bff[2, :]
else:
    fe[0] = 0.
    fH[0] = 0.
    fHmin[0] = 0.

tol_VMR = VMR[:, 0]
tot_molmass = molmass_layered[:, 0]

for ilayer in range(nlayers):
    allelse = (np.sum(tol_VMR) +
               fe[ilayer] +
               fH[ilayer] +
               fHmin[ilayer])

    fboth = 1.0 - allelse

    fratio = 0.84

    fH2[ilayer] = fratio * fboth
    fHe[ilayer] = (1.0 - fratio) * fboth

tot_VMR_molmass = np.sum(tol_VMR * tot_molmass)

# Compute temperature differences
# tdiff = np.zeros((nlayers, nlinetemps), dtype=float)
# Tlay1 = np.zeros((nlayers), dtype=float)
# for ilayer in range(nlayers):
#     for i in range(nlinetemps):
#         tdiff[ilayer, i] = abs(inlinetemps[i] - temp[ilayer])
#     Tlay1[ilayer] = np.argmin(tdiff)

@cuda.jit
def layer_thickness_kernel(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp, dp, dz):
    # Thread index
    ilayer = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if ilayer < nlayers:
        R_GAS = 8.3144621
        AVOGADRO = 6.02e23
        amu = 1.6605402e-27
        K_BOLTZ = R_GAS / AVOGADRO

        XH2 = 2.01588
        XHe = 4.002602
        XH = 1.008

        # Compute mean molecular weight
        mu[ilayer] = fH2[ilayer] * XH2 + fHe[ilayer] * XHe + fH[ilayer] * XH + fHmin[ilayer] * XH + tot_VMR_molmass

        # Compute specific gas constant
        R_spec = K_BOLTZ / (mu[ilayer] * amu)

        if ilayer == nlayers - 1:
            # Handle the last layer
            p1 = math.exp(0.5 * (math.log(press[ilayer - 1] * press[ilayer])))
            p2 = (press[ilayer] ** 2) / p1
            dp[ilayer] = p2 - p1
            dz[ilayer] = abs((R_spec * temp[ilayer] / grav) * math.log(p1 / p2))
        else:
            # Handle intermediate layers
            p1 = math.exp(1.5 * math.log(press[ilayer]) - 0.5 * math.log(press[ilayer + 1]))
            p2 = math.exp(0.5 * math.log(press[ilayer] * press[ilayer + 1]))

            dp[ilayer] = p2 - p1
            dz[ilayer] = abs((R_spec * temp[ilayer] / grav) * math.log(p1 / p2))

@cuda.jit
def compute_ndens_kernel(press, temp, ndens):
    K_BOLTZ = 1.380649e-23
    ilayer = cuda.grid(1)
    if ilayer < press.size:
        ndens[ilayer] = 1.0e+5 * press[ilayer] / (K_BOLTZ * temp[ilayer])

@jit(nopython=True)
def line_mixer(ilayer, opd_lines, linelist, linetemps, dz, ngas, nwave, ndens, nlinetemps, temp, VMR):

    logkap1 = np.zeros((ngas, nwave))
    logkap2 = np.zeros((ngas, nwave))
    logintkappa = np.zeros((ngas, nwave))
    tdiff = np.zeros(nlinetemps)

    for i in range(nlinetemps):
        tdiff[i] = np.abs(linetemps[i] - temp[ilayer])

    Tlay1 = np.argmin(tdiff)

    if linetemps[Tlay1] < temp[ilayer]:
        Tlay2 = Tlay1 + 1
    else:
        Tlay2 = Tlay1 - 1

    if temp[ilayer] < linetemps[0]:
        Tlay1 = 0
        Tlay2 = 1
    elif temp[ilayer] > linetemps[nlinetemps - 1]:
        Tlay1 = nlinetemps - 2
        Tlay2 = nlinetemps - 1

    if Tlay1 > Tlay2:
        torder = 1
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay2])) / (
                    np.log10(linetemps[Tlay1]) - np.log10(linetemps[Tlay2]))
    else:
        torder = 2
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay1])) / (
                    np.log10(linetemps[Tlay2]) - np.log10(linetemps[Tlay1]))

    if temp[ilayer] > linetemps[nlinetemps - 1]:
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay2])) / (
                    np.log10(linetemps[Tlay2]) - np.log10(linetemps[Tlay1]))

    logkap1 = linelist[:, ilayer, Tlay1, :]
    logkap2 = linelist[:, ilayer, Tlay2, :]

    if torder == 1:
        logintkappa = ((logkap1 - logkap2) * intfact) + logkap2
    else: #torder == 2:
        logintkappa = ((logkap2 - logkap1) * intfact) + logkap1
    #else:
    #    print("something wrong with interpolate order")

    if temp[ilayer] > linetemps[nlinetemps - 1]:
        logintkappa = ((logkap2 - logkap1) * intfact) + logkap2

    for igas in range(ngas):
        opd_lines[ilayer, :] = opd_lines[ilayer, :]+(VMR[igas, ilayer] * ndens[ilayer] * dz[ilayer] * 0.0001) * (10 ** logintkappa[igas, :])

    return opd_lines[ilayer, :]

@cuda.jit
def get_ray_kernel(ch4index, wavelen, nwave, ndens, nlayers, opd_rayl, VMR, fH2, fHe, dz):
    # Thread indices
    ilayer = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    iwave = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if ilayer < nlayers and iwave < nwave:
        XN0 = 2.687e19
        cfray = 32.0 * math.pi ** 3.0 * 1.0e21 / (3.0 * XN0)

        dpol = cuda.local.array(3, dtype=np.float64)
        dpol[0] = 1.022
        dpol[1] = 1.0
        dpol[2] = 1.0

        gnu = cuda.local.array((2, 3), dtype=np.float64)
        gnu[0, 0] = 1.355e-4
        gnu[0, 1] = 3.469e-5
        gnu[0, 2] = 4.318e-4
        gnu[1, 0] = 1.235e-6
        gnu[1, 1] = 8.139e-8
        gnu[1, 2] = 3.408e-6

        wa = wavelen[iwave]
        cold = ndens[ilayer] * dz[ilayer] * 1.0e-4
        taur = 0.0

        if ch4index != 0:
            ng = 3
            gasss = cuda.local.array(3, dtype=np.float64)
            gasss[0] = fH2[ilayer]
            gasss[1] = fHe[ilayer]
            gasss[2] = VMR[ch4index, ilayer]
        else:
            ng = 2
            gasss = cuda.local.array(2, dtype=np.float64)
            gasss[0] = fH2[ilayer]
            gasss[1] = fHe[ilayer]

        for nn in range(ng):
            tec = cfray * (dpol[nn] / wa ** 4) * (gnu[0, nn] + gnu[1, nn] / wa ** 2) ** 2
            taur += cold * gasss[nn] * tec * 1.0e-5 / XN0

        opd_rayl[ilayer, iwave] = taur

@jit(nopython=True)
def get_cia(cia, ciatemp, grav, ch4index, opd_cia, temp, press, VMR, fH2, fHe, fH, dz):

    nlayers = press.size
    nciatemps = 198

    ph2h2 = cia[0,:,:]
    ph2he = cia[1,:,:]
    ph2h = cia[2,:,:]
    ph2ch4 = cia[3,:,:]

    for ilayer in range(nlayers):
        tdiff = np.abs(ciatemp - temp[ilayer])
        tcia1 = np.argmin(tdiff)

        if ciatemp[tcia1] < temp[ilayer]:
            tcia2 = tcia1 + 1
        else:
            tcia2 = tcia1
            tcia1 = tcia2 - 1

        if temp[ilayer] < ciatemp[0]:
            tcia1 = 0
            tcia2 = 1
        elif temp[ilayer] > ciatemp[nciatemps - 1]:
            tcia1 = nciatemps - 2
            tcia2 = nciatemps - 1

        if tcia1 == 0:
            intfact = (np.log10(temp[ilayer]) - np.log10(ciatemp[0])) / (
                        np.log10(ciatemp[1]) - np.log10(ciatemp[0]))

            ciaH2H2 = 10 ** (((ph2h2[1, :] - ph2h2[0, :]) * intfact) + ph2h2[0, :])
            ciaH2He = 10 ** (((ph2he[1, :] - ph2he[0, :]) * intfact) + ph2he[0, :])
            ciaH2H = 10 ** (((ph2h[1, :] - ph2h[0, :]) * intfact) + ph2h[0, :])
            ciaH2CH4 = 10 ** (((ph2ch4[1, :] - ph2ch4[0, :]) * intfact) + ph2ch4[0, :])

        else:
            intfact = (np.log10(temp[ilayer]) - np.log10(ciatemp[tcia1])) / (
                        np.log10(ciatemp[tcia2]) - np.log10(ciatemp[tcia1]))

            ciaH2H2 = 10.0 ** (((ph2h2[tcia2, :] - ph2h2[tcia1, :]) * intfact) + ph2h2[tcia1, :])
            ciaH2He = 10.0 ** (((ph2he[tcia2, :] - ph2he[tcia1, :]) * intfact) + ph2he[tcia1, :])
            ciaH2H = 10.0 ** (((ph2h[tcia2, :] - ph2h[tcia1, :]) * intfact) + ph2h[tcia1, :])
            ciaH2CH4 = 10.0 ** (((ph2ch4[tcia2, :] - ph2ch4[tcia1, :]) * intfact) + ph2ch4[tcia1, :])

        n_amg = (press[ilayer] / 1.01325) * (273.15 / temp[ilayer])

        if ch4index != 0:
            opd_cia[ilayer, :] = (n_amg ** 2 * fH2[ilayer] * dz[ilayer] * 100) * (
                    (fH2[ilayer] * ciaH2H2) +
                    (fHe[ilayer] * ciaH2He) +
                    (fH[ilayer] * ciaH2H) +
                    (VMR[ch4index, ilayer] * ciaH2CH4))
        else:
           opd_cia[ilayer, :] = (n_amg ** 2 * fH2[ilayer] * dz[ilayer] * 100) * (
                    (fH2[ilayer] * ciaH2H2) +
                    (fH[ilayer] * ciaH2H) +
                    (fHe[ilayer] * ciaH2He))

    return opd_cia

import math

@cuda.jit(device=True)
def bbplk(waven, T):
    C = 299792458.0
    kb = 1.38064852e-23
    h = 6.62607004e-34
    wavelen = 1.0e-6 * (1.0e4 / waven)
    bb_value = 1.0e-6 * ((2.0 * h * C ** 2) / wavelen ** 5) / (math.exp(h * C / (wavelen * kb * T)) - 1.0)
    return bb_value


@cuda.jit(device=True)
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
def run_RT_kernel(clphotspec, othphotspec, cf, clphot, othphot, do_cf, inwavenum, cloudy, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover, spectrum, upflux):
    # Thread indices
    iwave = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ilayer = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    nlayers = 64
    nlevel = nlayers + 1
    cldone = 0
    othdone = 0

    if iwave < nwave and ilayer < nlayers:
        # Local arrays
        DTAUC = cuda.local.array(64, dtype=np.float64)
        SSALB = cuda.local.array(64, dtype=np.float64)
        COSBAR = cuda.local.array(64, dtype=np.float64)
        temper = cuda.local.array(65, dtype=np.float64)
        gflup = cuda.local.array(65, dtype=np.float64)
        fdi = cuda.local.array(65, dtype=np.float64)

        ALBEDO = 0.000000001

        # Set temperature levels
        for i in range(nlevel):
            temper[i] = temp[i]

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

        # Call gfluxi for radiative transfer
        DTAUC = opd_ext[:, iwave]
        if ilayer == 0:  # Only one thread per wavelength should call gfluxi_device
            gflup[0] = gfluxi_device(temper, DTAUC, SSALB, COSBAR, inwavenum[iwave], ALBEDO, gflup, fdi)
            upflux[iwave] = gflup[0]

            # Accumulate spectrum
        if ilayer == 0:  # Only one thread per wavelength should update the spectrum
            spectrum[iwave] += upflux[iwave] * pcover  # Ensure pcover is indexed correctly


def forward(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp, linelist, linetemps, ngas, nwave, nlinetemps, VMR, ch4index, wavelen, pcover, clphot, othphot, do_cf, clphotspec, othphotspec, cf, inwavenum):
    # Allocate device memory
    press_d = cuda.to_device(press)
    temp_d = cuda.to_device(temp)
    mu_d = cuda.to_device(mu)
    fH_d = cuda.to_device(fH)
    fHmin_d = cuda.to_device(fHmin)
    fH2_d = cuda.to_device(fH2)
    fHe_d = cuda.to_device(fHe)
    dp_d = cuda.device_array(nlayers, dtype=np.float64)
    dz_d = cuda.device_array(nlayers, dtype=np.float64)
    ndens_d = cuda.device_array(nlayers, dtype=np.float64)

    clphotspec_d = cuda.to_device(clphotspec)
    othphotspec_d = cuda.to_device(othphotspec)
    cf_d = cuda.to_device(cf)
    inwavenum_d = cuda.to_device(inwavenum)

    # opd_lines_d = cuda.device_array((nlayers, nwave), dtype=np.float64)
    # linelist_d = cuda.to_device(linelist)
    # linetemps_d = cuda.to_device(linetemps)
    # logVMR_d = cuda.to_device(VMR)

    opd_lines = np.zeros((nlayers, nwave))
    opd_CIA = np.zeros((nlayers, nwave))

    opd_rayl_d =  cuda.device_array(((nlayers, nwave)), dtype=np.float64)
    opd_ext_d = cuda.device_array(((nlayers, nwave)), dtype=np.float64)
    opd_scat_d = cuda.device_array(((nlayers, nwave)), dtype=np.float64)
    opd_hmbff_d = cuda.device_array(((nlayers, nwave)), dtype=np.float64)
    gg_d = cuda.device_array(((nlayers, nwave)), dtype=np.float64)
    spectrum_d = cuda.device_array(nwave, dtype=np.float64)
    upflux_d = cuda.device_array(nwave, dtype=np.float64)

    # Define block and grid dimensions
    threads_per_block = 64
    blocks_per_grid = (nlayers + threads_per_block - 1) // threads_per_block

    # # Create CUDA events for timing
    # start_event = cuda.event()
    # end_event = cuda.event()
    # # Record the start event
    # start_event.record()

    # Launch the kernel
    layer_thickness_kernel[blocks_per_grid, threads_per_block](
        grav, tot_VMR_molmass, mu_d, fH_d, fHmin_d, fH2_d, fHe_d, nlayers, press_d, temp_d, dp_d, dz_d)

    compute_ndens_kernel[blocks_per_grid, threads_per_block](press_d, temp_d, ndens_d)

    threads_per_block = (16, 16)  # Example: 16x16 threads per block

    # Calculate the number of blocks needed
    blocks_per_grid_x = (nlayers + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (nwave + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    get_ray_kernel[blocks_per_grid, threads_per_block](ch4index, wavelen, nwave, ndens_d, nlayers, opd_rayl_d, VMR, fH2, fHe, dz_d)

    # Calculate the elapsed time in milliseconds
    # elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    # print(f"GPU kernel time: {elapsed_time:.6f} milliseconds")

    # dp = dp_d.copy_to_host()
    dz = dz_d.copy_to_host()
    ndens = ndens_d.copy_to_host()
    # opd_rayl = opd_rayl_d.copy_to_host()

    for ilayer in range(nlayers):
        opd_lines[ilayer, :] = line_mixer(ilayer, opd_lines, linelist, inlinetemps, dz, ngas, nwave, ndens,
                                          nlinetemps, temp, VMR)
    opd_CIA = get_cia(cia, ciatemps, grav, ch4index, opd_CIA, temp, press, VMR, fH2, fHe, fH, dz)

    opd_lines_d = cuda.to_device(opd_lines)
    opd_CIA_d = cuda.to_device(opd_CIA)
    pcover_d = 0
    do_clouds_d = 0

    # Define thread and block dimensions
    threads_per_block = (16, 16)  # Example: 16x16 threads per block
    blocks_per_grid_x = (nwave + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (nlayers + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    run_RT_kernel[blocks_per_grid, threads_per_block](
        clphotspec_d, othphotspec_d, cf_d, clphot, othphot, do_cf, inwavenum_d, do_clouds_d, press, temp, nwave, opd_ext_d, opd_lines_d,
        opd_CIA_d, opd_rayl_d, opd_scat_d, opd_hmbff_d, gg_d, dp_d, pcover_d, spectrum_d, upflux_d)

    spectrum = spectrum_d.copy_to_host()

    # # Record the end event
    # end_event.record()
    # # Synchronize to ensure the events are completed
    # end_event.synchronize()

    return spectrum

spectrum = forward(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp, linelist, inlinetemps, ngas, nwave, nlinetemps, VMR, ch4index, wavelen, pcover, clphot, othphot, do_cf, clphotspec, othphotspec, cf, inwavenum)

print("spectrum : ", spectrum)
