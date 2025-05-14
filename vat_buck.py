"""
Variable angle tow buckling and weight function with geometric imperfection

 ___________________________
|                           |
|                           |
|                           |
|___________________________|
T0            T1            T0


INPUTS:
T0: VAT angle at initial and final(x-axis)
T1: VAT angle at mid-point(x-axis)
thick: 0 or 1 for optimization i.e. switching on and off a laminate
"""

import os

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu, spsolve
from composites import laminated_plate

from bfsccylinder import DOF, DOUBLE, INT
from bfsccylinder.bfsccylinder_sanders import (BFSCCylinderSanders, update_KC0, update_KG, KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights


def va_2nd_order(x, T0, T1, T2, phi, L):
    x0 = 0
    x1 = L/2
    x2 = 3*L/3
    phi *= pi / (180)
    T0 *= pi / (180)
    T1 *= pi / (180)
    T2 *= pi / (180)

    tht_x = phi+ T0*(x-x1)*(x-x2)/((x0-x1)*(x0-x2))+ T1*(x-x0)*(x-x2)/((x1-x0)*(x1-x2)) +\
            T2*(x-x0)*(x-x1)/((x2-x0)*(x2-x1))
    tht_xd = tht_x * 180 / pi
    return tht_xd


def theta_VAT_P_x(x, L, theta_ctrl):
    theta_VP_1, theta_VP_2, theta_VP_3 = theta_ctrl
    x1 = 0
    x2 = L/4
    x3 = L/2
    x4 = 3*L/4
    x5 = L
    check = (x <= L/2)
    N1 = (x - x2)*(x - x3)/((x1 - x2)*(x1 - x3))
    N2 = (x - x1)*(x - x3)/((x2 - x1)*(x2 - x3))
    N3L = (x - x1)*(x - x2)/((x3 - x1)*(x3 - x2))
    N3R = (x - x4)*(x - x5)/((x3 - x4)*(x3 - x5))
    N4 = (x - x3)*(x - x5)/((x4 - x3)*(x4 - x5))
    N5 = (x - x3)*(x - x4)/((x5 - x3)*(x5 - x4))

    out = np.zeros_like(x)
    out[check] = (N1*theta_VP_1 + N2*theta_VP_2 + N3L*theta_VP_3)[check]
    out[~check] = (N3R*theta_VP_3 + N4*theta_VP_2 + N5*theta_VP_1)[~check]

    return out


def optim_test(desvars, geo_prop=None, mat_prop=None, ny=60, vol_only=False,
               out_mesh=False, balanced=True, theta_func=theta_VAT_P_x, cg_x0=None, lobpcg_X=None):

    if geo_prop is None:
        # Geometric Parameters(m)
        geo_prop = dict(
            L=0.300,
            R=0.15
        )
    if mat_prop is None:
        # Material Properties
        mat_prop = dict(
            E11=90e9,  # CS-Z wang
            E22=7e9,
            nu12=0.32,
            G12=4.4e9,
            G23=1.8e9,
            plyt=0.4e-3
        )

    out = {}
    out['lobpcg_X'] = None
    out['cg_x0'] = None
    out['Pcr'] = 1e-6
    out['volume'] = 1e15
    out['rel_vol'] = 1e15
    out['nid_pos'] = None
    out['n1s'] = None
    out['n2s'] = None
    out['n3s'] = None
    out['n4s'] = None
    out['xlin'] = None
    out['ylin'] = None
    out['ncoords'] = None

    L = geo_prop['L']
    R = geo_prop['R']
    b = 2*pi*R

    E11 = mat_prop['E11']
    E22 = mat_prop['E22']
    nu12 = mat_prop['nu12']
    G12 = mat_prop['G12']
    G23 = mat_prop['G23']
    plyt = mat_prop['plyt']
    laminaprop = (E11, E22, nu12, G12, G12, G23)

    # number of nodes
    nx = int(ny*L/b)
    if nx%2 == 0:
        nx += 1

    vol_00 = 2 * pi * R * L * 2 * plyt

    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    xlin = np.linspace(0, L, nx)
    ytmp = np.linspace(0, b, ny+1)
    ylin = np.linspace(0, b-(ytmp[-1] - ytmp[-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T
    zmesh = np.zeros_like(ymesh)

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten(), zmesh.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()
    if out_mesh:
        out['nid_pos'] = nid_pos
        out['n1s'] = n1s
        out['n2s'] = n2s
        out['n3s'] = n3s
        out['n4s'] = n4s
        out['xlin'] = xlin
        out['ylin'] = ylin
        out['ncoords'] = ncoords

    nint = 4
    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)

    elements = []
    N = DOF*nx*ny #u, dux, dut, v, dvx, dvt, w, dwx, dwt, d2wxt
    init_k_KC0 = 0
    init_k_KG = 0
    volume = 0
    first_element =True

    #TODO calculate steering radius constraint
    x_space = np.linspace(0, L, 1000)
    theta_min = []
    theta_max = []
    for desvar in desvars:
        theta_space = theta_func(x_space, L, desvar)
        theta_min.append(theta_space.min())
        theta_max.append(theta_space.max())

    # angle constraints
    # NOTE values taken from Wang et al.
    # https://doi.org/10.1007/s00158-022-03227-8#Sec3
    for k, desvar in enumerate(desvars):
        # minimum angle constraint
        if abs(theta_min[k]) < 3.3:
            return out
        # maximum angle constraint
        if abs(theta_max[k]) > 87.7:
            return out

    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinderSanders(nint)
        shell.n1 = n1
        shell.n2 = n2
        shell.n3 = n3
        shell.n4 = n4
        shell.c1 = DOF*nid_pos[n1]
        shell.c2 = DOF*nid_pos[n2]
        shell.c3 = DOF*nid_pos[n3]
        shell.c4 = DOF*nid_pos[n4]
        shell.R = R
        shell.lex = L/(nx-1)
        shell.ley = b/ny

        x1, y1, z1 = ncoords[nid_pos[shell.n1]]
        x2, y2, z2 = ncoords[nid_pos[shell.n2]]
        x4, y4, z4 = ncoords[nid_pos[shell.n4]]

        for i in range(nint):
            xi = points[i]
            wi = weights[i]
            x_local = x1 + (x2 - x1)*(xi + 1)/2
            stack = []
            plyts = []
            for k, desvar in enumerate(desvars):
                theta = theta_func(x_local, L, desvar)
                steering_angle = theta - theta_min[k]
                plyt_loc = plyt / np.cos(np.deg2rad(steering_angle))

                stack.append(theta)
                plyts.append(plyt_loc)
                if balanced:
                    stack.append(-theta)    # NOTE for balanced plyt_loc*thick, uncomment
                    plyts.append(plyt_loc)  # NOTE for balanced plyt_loc*thick, uncomment

            ABD = laminated_plate(stack=stack, plyts=plyts, laminaprop=laminaprop).ABD
            for j in range(nint):
                eta = points[j]
                wj = weights[j]
                y_local = y1 + (y4 - y1)*(eta + 1)/2
                volume += wi*wj*shell.lex*shell.ley/4*sum(plyts)
                shell.A11[i, j] = ABD[0, 0]
                shell.A12[i, j] = ABD[0, 1]
                shell.A16[i, j] = ABD[0, 2]
                shell.A22[i, j] = ABD[1, 1]
                shell.A26[i, j] = ABD[1, 2]
                shell.A66[i, j] = ABD[2, 2]
                shell.B11[i, j] = ABD[0, 0+3]
                shell.B12[i, j] = ABD[0, 1+3]
                shell.B16[i, j] = ABD[0, 2+3]
                shell.B22[i, j] = ABD[1, 1+3]
                shell.B26[i, j] = ABD[1, 2+3]
                shell.B66[i, j] = ABD[2, 2+3]
                shell.D11[i, j] = ABD[0+3, 0+3]
                shell.D12[i, j] = ABD[0+3, 1+3]
                shell.D16[i, j] = ABD[0+3, 2+3]
                shell.D22[i, j] = ABD[1+3, 1+3]
                shell.D26[i, j] = ABD[1+3, 2+3]
                shell.D66[i, j] = ABD[2+3, 2+3]
        shell.init_k_KC0 = init_k_KC0
        shell.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(shell)

    if vol_only:
        out['volume'] = volume
        out['rel_vol'] = volume/vol_00
        return out

    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KC0(shell, points, weights, Kr, Kc, Kv)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    # simply supported [DOFs: u, u_x,u_y,v,v_x,v_y, w, w_x, w_y, w_xy]
    #                         0  1    2  3  4   6   6  7     8   9
    checkBC = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkBC    #Clamped and SS
    bk[3::DOF] = checkBC    #Clamped and SS
    bk[6::DOF] = checkBC    #Clamped and SS
    bk[7::DOF] = checkBC    #Clamped

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # axial compression applied at x=L
    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.001
    checkTopEdge = isclose(x, L)
    u[0::DOF] += checkTopEdge*compression
    uk = u[bk]

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    Kuk = KC0[bu, :][:, bk]
    Kkk = KC0[bk, :][:, bk]
    fu = -Kuk*uk

    # solving
    PREC = 1/Kuu.diagonal().max()
    if cg_x0 is None:
        uu, info = cg(PREC*Kuu, PREC*fu, atol=0)
    else:
        uu, info = cg(PREC*Kuu, PREC*fu, x0=cg_x0[bu], atol=0)
    if info != 0:
        uu = spsolve(Kuu, fu)
    u[bu] = uu
    out['cg_x0'] = u.copy()

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # A * x[i] = lambda[i] * M * x[i]
    num_eigvals = 2
    Nu = N - bk.sum()
    try:
        #NOTE this works and seems to be the fastest option

        #NOTE incomplete SPLU with reduced drop_tol and increased fill_factor
        PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-10, fill_factor=100)

        def matvec(x):
            return PREC2.solve(x)
        Ainv = LinearOperator(matvec=matvec, shape=(Nu, Nu))

        maxiter = 1000
        if lobpcg_X is None:
            lobpcg_X = np.random.rand(Nu, num_eigvals) - 0.5  # For optimization use previous solver value
            lobpcg_X /= np.linalg.norm(lobpcg_X, axis=0)
        else:
            lobpcg_X = np.asarray(lobpcg_X)[bu, :]
        #NOTE default tolerance is too large
        tol = 1e-5
        eigvals, eigvecsu, hist = lobpcg(A=PREC*Kuu, B=-PREC*KGuu, X=lobpcg_X, M=Ainv, largest=False,
                maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
        load_mult = eigvals

    except:
        #NOTE works, but slower than lobpcg
        eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                tol=1e-9, sigma=1., mode='buckling')
        load_mult = -eigvals

    f = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
    f[bk] = fk
    Pcr = (load_mult[0]*f[0::DOF][checkTopEdge]).sum()

    eigvecs = np.zeros((N, num_eigvals), dtype=float)
    eigvecs[bu, :] = eigvecsu[:, :]
    out['Pcr'] = Pcr
    out['volume'] = volume
    out['rel_vol'] = volume/vol_00
    out['eigvecs'] = eigvecs
    out['bu'] = bu
    out['lobpcg_X'] = eigvecs

    return out


def objective_function(design_load, out):
    lbd = 0.95*abs(out['Pcr'])/abs(design_load)
    factor = max(1, 1/lbd**2)
    objective = factor*out['rel_vol']
    assert objective > 0
    return objective


if __name__ == '__main__':
    # import addcopyfighandler


    import time
    x_theta = np.linspace(0, 1, 50)
    y_del = np.zeros([len(x_theta)])
    cg00 = []
    DESIGN_LOAD = 550e3

    if True:
        print('started')
        # Geometric Parameters(m)
        geo_dict = dict(
            L= 0.3,
            R= 0.15
        )


        # Material Properties

        mat_dict = dict(
            E11 = 90e9,  # CS-Z wang
            E22 = 7e9,
            nu12 = 0.32,
            G12 = 4.4e9,
            G23 = 1.8e9,
            plyt = 0.4e-3)

        start_time = time.time()
        i_thk = 1
        desvars2 = [[10.5, 56.5, 10.6],[9.6, 80.7, 58.3]]
        if i_thk != 0:
            desvars2 = np.around(desvars2, 3)

            output = optim_test(desvars2, geo_dict, mat_dict, ny=55, vol_only=False,
                                balanced=True)
            Pcr, volume, rel_vol = output['Pcr'], output['volume'], output['rel_vol']
            print('Pcr:', -Pcr * 0.001, 'kN,', 'Vol: {:e}, rel_vol = '.format(volume), rel_vol)
            lamda_2 = (0.95 * Pcr / (DESIGN_LOAD)) ** 2
            factor = max(1, (1 / lamda_2))
            # y_del[i_x] = rel_vol*factor
            print('objective ', rel_vol * factor)
            print('weight:', volume * 1600, 'kg')
        else:
            print('No thickness')
        print("--- %s seconds ---" % (time.time() - start_time))

