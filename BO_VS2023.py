import math
from warnings import catch_warnings, simplefilter

import numpy as np
from numpy import arange, around, vstack, asarray, argmin
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from sklearn.model_selection import KFold
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score


from vat_buck import optim_test, objective_function


global lobpcg_X, cg_x0, out

lobpcg_X = {'0': [],
          '1': [],
          '2': [],
          '3': [],
          '4': []}
cg_x0 = {'0': [],
           '1': [],
           '2': [],
           '3': [],
           '4': []}
test_min = dict(objective=10,
                desvars=[],
                iter=0,
                final_desvars=[])

class OptParam(object):
    __slots__ = ['n_samples', 'del_theta_set',
                 'theta_space', 'total_iter', 'space']
    def __init__(self):
        self.n_samples = None
        self.del_theta_set = None
        self.theta_space = None
        self.total_iter = None
        self.space = None



def sort_desvar(desvars):
    desvars2 = []
    for i in range(MAX_LAYERS):
        if i == 0:
            T1, T2, T3 = desvars[0:3] * 89
            desvars2.append([abs(T1), T2, abs(T3)])
        elif desvars[4 * i + 2] == 1:
            T1, T2, T3 = desvars[3 + 4 * (i - 1):6 + 4 * (i - 1)] * 89
            desvars2.append([abs(T1), T2, abs(T3)])
    return around(desvars2, 2)


# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        mean, std = model.predict(X, return_std=True)

        return mean


# probability of improvement acquisition function
def acq_MPI(X, Xsamples, model):
    # calculate the best surrogate score found so far
    # print('acq Xsample:', Xsamples)
    yhat = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = model.predict(Xsamples, return_std=True)
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


def acq_EI(X, Xsamples, model, xi=0.01):
    '''
    from:http://krasserm.github.io/2018/03/21/bayesian-optimization/
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.

    '''
    mu, sigma = model.predict(X, return_std=True)
    Xsamples = Xsamples.reshape(len(Xsamples), INPUT_VARS - 1)
    # Ysamples,_ = surrogate(model, Xsamples)
    mu_sample, sigma_sam = model.predict(Xsamples, return_std=True)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_opt = np.min(mu)

    with np.errstate(divide='warn'):
        imp = mu_sample - mu_opt - xi
        Z = imp / sigma_sam
        ei = imp * norm.cdf(Z) + sigma_sam * norm.pdf(Z)
        ei[np.isclose(sigma_sam, 0.0)] = 0.0

    return ei


def acq_LCB(xsample, model, exploration_weight=0.3):
    """
    Computes the GP-Lower Confidence Bound
    """

    m, s = model.predict(xsample, return_std=True)
    m = asarray(m).reshape(len(m), 1)
    s = asarray(s).reshape(len(s), 1)

    f_acqu = (1 - exploration_weight) * m - exploration_weight * s
    return f_acqu


# optimize the acquisition function
def opt_acquisition(X, y, op, model, Acq_fun="LCB", weight=-1, seed_val=6):
    # random search, generate random samples
    # TODO check whether the starting points should be random
    Xsamples = asarray(random_desvar(op, seed_val=seed_val))

    gs = np.zeros_like(Xsamples)
    rel_vol_list = []
    # NOTE this for loop is just to check the feasibility of the Xsamples
    ny_check_fisibility = 25
    for desx in Xsamples:
        xvol = sort_desvar(desx)
        tmp_out = optim_test(xvol, geo_prop=geo_dict, mat_prop=mat_dict, ny=ny_check_fisibility)
        rel_vol_list.append(tmp_out['rel_vol'])
    rel_vol = asarray(rel_vol_list)
    gs = (rel_vol != -100)
    rel_vol = rel_vol[gs]

    Xsamples = Xsamples.reshape(len(Xsamples), INPUT_VARS - 1)
    Xsamples = Xsamples[gs]

    # calculate the acquisition function for each sample
    if Acq_fun == 'MPI':
        scores = acq_MPI(X, Xsamples, model)
    if Acq_fun == 'EI':
        if weight == -1:
            weight = 0.02
        scores = acq_EI(X, Xsamples, model, xi=weight)
    if Acq_fun == 'LCB':
        if weight == -1:
            weight = 0.3
        scores = acq_LCB(Xsamples, model, weight)
    # locate the index of the min scores
    ix = argmin(scores)
    return Xsamples[ix]


def random_desvar(op, seed_val=6):
    lhs = Lhs(lhs_type="classic", criterion=None)
    X = lhs.generate(op.space.dimensions, n_samples=op.n_samples, random_state=seed_val)
    des_sample = around(X, 5)
    ds = asarray(des_sample)

    des_sample = ds.reshape(len(ds), INPUT_VARS - 1)

    return around(des_sample, 5)


def k_fold_chk(IP_times):
    kf = KFold(n_splits=5, shuffle=True)
    train_pop = IP_times

    tot_pop = int(train_pop / 0.8)

    print('training size:', train_pop)

    r2 = []
    mse = []
    lml = []
    for i in range(3):
        idx = np.random.choice(len(X) - 1, tot_pop)
        for train_idx, test_idx in kf.split(X[idx]):
            xtrain, xtest, ytrain, ytest = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
            model.fit(xtrain, ytrain)
            y_pred = model.predict(xtest)
            lml.append(model.log_marginal_likelihood_value_)
            r2.append(r2_score(ytest, y_pred))
            mse.append(mean_squared_error(ytest, y_pred))
    print('___Avg LML___', np.average(lml), '___Var LML___', np.var(lml))
    print('___Avg R2___', np.average(r2), '___Var R2___', np.var(r2))
    print('___Avg MSE___', np.average(mse), '___Var MSE___', np.var(mse))


if __name__ == '__main__':
    MAX_LAYERS = 3
    ny_init_sampling = 30
    ny_optimization = 30
    ny_verification = 30

    layers_loads = (#(1,  50e3),
                    (2, 100e3),
                    (2, 200e3),
                    (2, 500e3),
                    (3, 200e3),
                    (3, 500e3),
                    (3, 1000e3),
                    (4, 500e3),
                    (4, 1000e3))
    for MAX_LAYERS, design_load in layers_loads:
        # ___________________________________________________________
        ## INPUTS
        global INPUT_VARS, X, Y, model, geo_dict, mat_dict, y_norm

        # Geometric Parameters(m)
        geo_dict = dict(
            L=0.300,
            R=0.15
        )

        # Material Properties
        mat_dict = dict(
            E11=90e9,  # CS-Z wang
            E22=7e9,
            nu12=0.32,
            G12=4.4e9,
            G23=1.8e9,
            plyt=0.4e-3
        )

        # Objective Parameters
        del_theta = 70  # in degrees; Max diffrence in angles of T1, T2 and T3
        theta_increment = 10  # in degrees; differnce between neighbouring angles

        # Optimizer parameters

        INPUT_VARS = MAX_LAYERS * 4
        ini_times = 10  # No of times of Input variables initial sample is
        ini_pop_size = ini_times * INPUT_VARS  # No of Initial points for GP fit
        n_samples = int(ini_pop_size/5)  # int(ini_pop_size/5)          	# Population size when optimizing
        total_iter = 50  # Total iterations
        tol = 1e-3  # Tolerance

        Acquisition_func = 'LCB'  # Choose Between "MPI", "EI" or "LCB" (default LCB)
        EE_weight = 0.4
        Sampling = 'LHS'  # Sampler type for Acquisition: 'random'/'LHS'(Latin Hypercube)

        print('Max_layer:{}, Design Load:{} N'.format(MAX_LAYERS, design_load))
        print('Acquisition func:', Acquisition_func, 'weight:', EE_weight)
        print('Initial pop: {}, \tTotal iterations:{}'.format(ini_pop_size, total_iter))
        # ___________________________________________________________
        # DESIGN SPACE

        # keeping theta increment constant at 5 deg for initial sampling
        theta_space = around(arange(0.0, 1, 15 / 89), 5)
        del_theta_space = around(arange(-1, 1, 15 / del_theta), 5)

        print('Begin Initial sample')
        X = []
        des_space = []
        for i in range(MAX_LAYERS):
            des_space.append(theta_space)
            des_space.append(theta_space)
            des_space.append(theta_space)
            if i != 0:
                des_space.append([0, 1])

        # ___________________________________________________________
        #  CALCULATING INITIAL SAMPLE POINTS
        op = OptParam()
        op.n_samples = ini_pop_size
        op.del_theta_set = del_theta_space
        op.theta_space = theta_space
        op.total_iter = total_iter
        op.space = Space(des_space)

        ii = 0
        Y = []
        lhs = Lhs(lhs_type="classic", criterion=None)
        Y = []
        X = random_desvar(op)
        Y_Pcr = []
        Y_vol = []
        for des_i in X:
            ii += 1
            if len(des_i) > 3 and np.sum(des_i[3::4]) == 0:
                Y_obj, Pcr, Vol = -100, -100, -100
            else:
                print('DEBUG', des_i)
                tmp_out = optim_test([des_i], geo_prop=geo_dict, mat_prop=mat_dict, ny=ny_init_sampling)
                Y_obj = objective_function(design_load, tmp_out)
                Pcr = tmp_out['Pcr']
                Vol = tmp_out['volume']
            Y.append(Y_obj)
            Y_Pcr.append(Pcr)
            Y_vol.append(Vol)

            if ii % 10 == 0:
                print(ii, des_i, Y[-1])

        print('sample_Space done')
        Yx = asarray(Y)
        Xx = asarray(X)

        load_dir = ""
        np.savetxt(load_dir + "finalx_{}kN_{}iter.csv".format(int(design_load / 1000), total_iter), Xx, delimiter=',')
        np.savetxt(load_dir + "finaly_{}kN_{}iter.csv".format(int(design_load / 1000), total_iter), Yx, delimiter=',')

        theta_space = around(arange(0.0, 1, 2 * theta_increment / 90), 5)
        del_theta_space = around(arange(-1, 1, theta_increment / del_theta), 5)
        des_space = []
        for i in range(MAX_LAYERS):
            des_space.append(theta_space)
            des_space.append(theta_space)
            des_space.append(theta_space)
            if i != 0:
                des_space.append([0, 1])
        # ___________________________________________________________
        ## Defining Dict for convenience
        op2 = OptParam()
        op2.n_samples = int(n_samples / 2)
        op2.del_theta_set = del_theta_space
        op2.theta_space = theta_space
        op2.total_iter = total_iter
        op2.space = Space(des_space)

        # ___________________________________________________________
        # Total Design Space calculation
        total_des_space = 1
        ncr = math.comb(len(theta_space) ** 2 + 1, 2)
        print('tot', total_des_space, 'ncr', ncr)
        for i_el in range(MAX_LAYERS - 1):
            total_des_space += ncr
            ncr = math.comb(ncr ** 2 + 1, 2) - math.comb(ncr ** 2 + 1 - len(theta_space) ** 2, 2)
        print('Total Design space = {:3e}'.format(total_des_space))

        # ___________________________________________________________
        # NORMALIZING OUTPUT FOR OPTIMIZATION
        X = asarray(Xx)
        Y = asarray(Yx).reshape(len(Yx))
        Y_Pcr = asarray(Y_Pcr)
        Y_vol = asarray(Y_vol)

        good_set = Y != -100
        Y[~good_set] = 2
        y_norm = max(Y[good_set])
        Y = Y[good_set] / y_norm
        Y_Pcr = Y_Pcr[good_set]
        Y_vol = Y_vol[good_set]
        X = asarray(X[good_set])
        ## reshape into rows and cols
        X = X.reshape(len(X), INPUT_VARS - 1)
        Y = Y.reshape(len(Y), 1)

        # ___________________________________________________________
        # MODEL FITTING

        ## defining the model using scikit-learn

        len_scale = 1.06
        se = kernels.RBF(length_scale=len_scale, length_scale_bounds=(1e-5, 1e5))
        prdc = kernels.ExpSineSquared(len_scale)
        ln1 = kernels.ConstantKernel()
        mat32 = kernels.Matern(length_scale=len_scale, nu=1.5)
        kernel = mat32  # + ln1 * prdc
        # Normalizing False as its done already
        model2 = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer='fmin_l_bfgs_b',
                                          n_restarts_optimizer=50)
        model2.fit(X, Y)
        opti_kernel = model2.kernel_
        print(opti_kernel)
        model = GaussianProcessRegressor(kernel=opti_kernel, normalize_y=False, optimizer=None,
                                         n_restarts_optimizer=1)
        k_fold_chk(len(Y) * 0.8)
        model.fit(X, Y)

        print('model_fit done')
        Y_best = []
        Y_log = []

        print('Optimization Progress:')
        # ___________________________________________________________
        ## Optimization
        for i in range(total_iter):
            if i == int(1 * total_iter / 5):
                op2.n_samples = n_samples
                op2.theta_space = around(arange(0.0, 1, theta_increment / (90)), 5)
            if i % 3 == 0:
                EE_weight = 0.3
                Acquisition_func = 'LCB'
            elif i % 3 == 1:
                Acquisition_func = 'MPI'
            else:
                EE_weight = 0.025
                Acquisition_func = 'EI'

            # select the next point to sample
            x_new = opt_acquisition(X, Y, op2, model, Acq_fun=Acquisition_func,
                                    weight=EE_weight, seed_val=(i + 1))

            # sample the point
            tmp_out = optim_test([x_new], geo_prop=geo_dict, mat_prop=mat_dict, ny=ny_optimization)
            actual = objective_function(design_load, tmp_out)
            ypcr = tmp_out['Pcr']
            yvol = tmp_out['volume']

            # summarize the finding
            est = surrogate(model, x_new.reshape(1, -1))

            # add the data to the dataset
            X = vstack((X, [x_new]))
            Y = vstack((Y, [actual]))
            Y_Pcr = np.hstack((Y_Pcr, [ypcr]))
            Y_vol = np.hstack((Y_vol, [yvol]))

            if i % int(total_iter / 2) == 0:
                ix = argmin(Y)
                print('=> iter {} of {} , f()=%3.4f, actual=%.4f'.format(i + 1, total_iter) % (est, actual))
                print('\n Best Result yet: x={}, y=%.5f'.format(X[ix]) % (Y[ix]))

            # Fit the model for next iteration
            model.fit(X, Y)

            Y_best.append(min(Y))
            Y_log.append(Y[-1])
            # Tolerance
            delta = [abs(Y[-1:-5] - Y[-2:-6])]
            if np.average(delta) <= tol:
                print('Tolerance Achieved')
                break

        ##best result
        ix = argmin(Y)
        print('Max_layer:{}, Design Load:{} N'.format(MAX_LAYERS, design_load))
        xopt = sort_desvar(X[ix])

        print('Acquisition func:', Acquisition_func, 'w:', EE_weight)
        print('Best Result: x={}, Objective=%.5f'.format(xopt) % (Y[ix]))
        out = optim_test(xopt, geo_prop=geo_dict, mat_prop=mat_dict, ny=ny_verification)
        print('Volume:', out['volume'], 'Pcr:', abs(out['Pcr']) * 0.001, 'kN,', ' rel_vol =', out['rel_vol'])

        np.savetxt("xopt{}t_iter{}_{}kN_{}.csv".format(ini_times, total_iter, int(design_load / 1000),
                                                                  Acquisition_func), X[ini_pop_size:], delimiter=',')
        np.savetxt("yopt{}t_iter{}_{}kN_{}.csv".format(ini_times, total_iter, int(design_load / 1000),
                                                                  Acquisition_func), Y[ini_pop_size:], delimiter=',')
        np.savetxt("ypcr{}t_iter{}_{}kN_{}.csv".format(ini_times, total_iter, int(design_load / 1000),
                                                                  Acquisition_func), Y_Pcr[ini_pop_size:], delimiter=',')
        np.savetxt("yvol{}t_iter{}_{}kN_{}.csv".format(ini_times, total_iter, int(design_load / 1000),
                                                                  Acquisition_func), Y_vol[ini_pop_size:], delimiter=',')

