# TODO make and test path to pickle file and gaslist.dat

import numpy as np


def setup(inputs, gaslistpath):

    (
        temp,
        logg,
        R2D2,
        gasnum,
        logVMR,
        pcover,
        do_clouds,
        cloudnum,
        cloudrad,
        cloudsig,
        cloudprof,
        inlinetemps,
        press,
        inwavenum,
        linelist,
        cia,
        ciatemps,
        clphot,
        ophot,
        make_cf,
        do_bff,
        bff,
    ) = inputs

    (
        temp,
        logg,
        R2D2,
        gasnum,
        logVMR,
        pcover,
        do_clouds,
        cloudnum,
        cloudrad,
        cloudsig,
        cloudprof,
        inlinetemps,
        press,
        inwavenum,
        linelist,
        cia,
        ciatemps,
        clphot,
        othphot,
        do_cf,
        bfing,
        bff,
    ) = (
        temp,
        logg,
        R2D2,
        gasnum,
        logVMR,
        pcover,
        do_clouds,
        cloudnum,
        cloudrad,
        cloudsig,
        cloudprof,
        inlinetemps,
        press,
        inwavenum,
        linelist,
        cia,
        ciatemps,
        clphot,
        ophot,
        make_cf,
        do_bff,
        bff,
    )

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
    grav = 10.0 ** (logg) / 100.0

    with open(gaslistpath, "r") as file:
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
            VMR[igas, ilayer] = 10.0 ** (logVMR[igas, ilayer])
            molmass_layered[igas, ilayer] = molmass[igas]

    if VMRname[igas, 0] == "ch4":
        ch4index = igas

    if bfing:
        fe[0] = 10.0 ** bff[0, :]
        fH[0] = 10.0 ** bff[1, :]
        fHmin[0] = 10.0 ** bff[2, :]
    else:
        fe[0] = 0.0
        fH[0] = 0.0
        fHmin[0] = 0.0

    tol_VMR = VMR[:, 0]
    tot_molmass = molmass_layered[:, 0]

    for ilayer in range(nlayers):
        allelse = np.sum(tol_VMR) + fe[ilayer] + fH[ilayer] + fHmin[ilayer]

        fboth = 1.0 - allelse

        fratio = 0.84

        fH2[ilayer] = fratio * fboth
        fHe[ilayer] = (1.0 - fratio) * fboth

    tot_VMR_molmass = np.sum(tol_VMR * tot_molmass)

    return grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp

# Compute temperature differences
# tdiff = np.zeros((nlayers, nlinetemps), dtype=float)
# Tlay1 = np.zeros((nlayers), dtype=float)
# for ilayer in range(nlayers):
#     for i in range(nlinetemps):
#         tdiff[ilayer, i] = abs(inlinetemps[i] - temp[ilayer])
#     Tlay1[ilayer] = np.argmin(tdiff)
