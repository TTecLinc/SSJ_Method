# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 01:43:55 2021

@author: Peilin Yang
"""

# Python packages
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
# Sequence jacobian package
from toolkit import jacobian as jac
from toolkit.solved_block import solved
from toolkit import estimation as ssj_est
from toolkit import nonlinear
sns.set_palette("cubehelix", 4)

# Models
import scipy.optimize as opt
from models import smets_wouters as sw
from models import krusell_smith as ks
from models import hank_1a
from models import hank_2a

# Auxiliary functions
from toolkit import aux_fn as aux
from toolkit import aux_speed as aux_jac


# ------------------------------------------------------------------------
# Estimate

def objective(x):

    # x is parameter vector

    T_irf = T - 20
    n_se, n_sh = len(outputs), len(shock_series)
    meas_error = np.zeros(n_se)  # set measurement error to zero
        
    # extract shock parameters from x; order: always sigma first, then AR coefs, then MA coefs
    # Step 0: Get ARMA
    ix, ishock = 0, 0
    sigmas, arcoefs, macoefs = np.zeros((3, len(shock_series)))
    # in our model, order=2
    for shock_name, order in shock_series:
        sigmas[ishock] = x[ix]
        ix += 1
        if order >= 1:
            arcoefs[ishock] = x[ix]
            ix += 1
        if order >= 2:
            macoefs[ishock] = x[ix]
            ix += 1
        ishock += 1
    # Step 1
    """Compute the MA representation given G"""
    # Compute MA representation of outcomes in As
    As = np.empty((T_irf, n_se, n_sh))
    for i_sh in range(n_sh):

        # Generates shock IRF for any ARMA process 

        ar_coeff = np.array([arcoefs[i_sh]])
        ma_coeff = np.array([macoefs[i_sh]])
        
        arma_shock = np.empty((T,))
        n_ar = ar_coeff.size
        n_ma = ma_coeff.size
        sign_ma = -1  # this means all MA coefficients are multiplied by -1 (this is what SW etc all have)
        for t in range(T):
            if t == 0:
                arma_shock[t] = 1
            else:
                ar_sum = 0
                for i in range(min(n_ar, t)):
                    ar_sum += ar_coeff[i] * arma_shock[t - 1 - i]
                ma_term = 0
                if 0 < t <= n_ma:
                    ma_term = ma_coeff[t - 1]
                arma_shock[t] = ar_sum + ma_term * sign_ma


        if np.abs(arma_shock[-1]) > 1e20:
            raise Warning('ARMA shock misspecified, leading to explosive shock path!')

        # store for each series
        shockname = shock_series[i_sh][0]
        for i_se in range(n_se):
            As[:, i_se, i_sh] = (G[outputs[i_se]][shockname] @ arma_shock)[:T_irf]


    # Step 2
    Sigma = ssj_est.all_covariances(As, sigmas)
    # Step 3
    sigma_o=meas_error
    To, O = data.shape
    llh = (ssj_est.log_likelihood(data, Sigma, sigma_o) - (To * O * np.log(2 * np.pi)) / 2)

    # compute the posterior by adding the log prior
    log_posterior = llh + log_priors(x, priors_list)
    return - log_posterior
# ------------------------------------------------------------------------


def log_priors(x, priors_list):
    """This function computes a sum of log prior distributions that are stored in priors_list.
    Example: priors_list = {('Normal', 0, 1), ('Invgamma', 1, 2)}
    and x = np.array([1, 2])"""
    assert len(x) == len(priors_list)
    sum_log_priors = 0
    for n in range(len(x)):
        dist = priors_list[n][0]
        mu = priors_list[n][1]
        sig = priors_list[n][2]
        if dist == 'Normal':
            sum_log_priors += - 0.5 * ((x[n] - mu) / sig) ** 2
        elif dist == 'Uniform':
            lb = mu
            ub = sig
            sum_log_priors += - np.log(ub - lb)
        elif dist == 'Invgamma':
            alpha = (mu / sig) ** 2 + 2
            beta = mu * (alpha - 1)
            sum_log_priors += (-alpha - 1) * np.log(x[n]) - beta / x[n]
        elif dist == 'Invgamma_hs':
            s = mu
            v = sig
            sum_log_priors += (-v - 1) * np.log(x[n]) - v * s ** 2 / (2 * x[n] ** 2)
        elif dist == 'Gamma':
            theta = sig ** 2 / mu
            k = mu / theta
            sum_log_priors += (k - 1) * np.log(x[n]) - x[n] / theta
        elif dist == 'Beta':
            alpha = (mu * (1 - mu) - sig ** 2) / (sig ** 2 / mu)
            beta = alpha / mu - alpha
            sum_log_priors += (alpha - 1) * np.log(x[n]) + (beta - 1) * np.log(1 - x[n])
        else:
            raise ValueError('Distribution provided is not implemented in log_priors!')

    if np.isinf(sum_log_priors) or np.isnan(sum_log_priors):
        print(x)
        raise ValueError('Need tighter bounds to prevent prior value = 0')
    return sum_log_priors



# MCMC chains
Nsim = 100_000  # number of simulations (200_000 for the paper)
Nburn = 50_000  # number of initial periods thrown away (50_000 for the paper)


ss = ks.ks_ss()

# Compute model jacobian G
T = 300
G = jac.get_G(block_list=[ks.firm, ks.mkt_clearing, ks.household], exogenous=['Z'], unknowns=['K'],
              targets=['asset_mkt'], T=T, ss=ss)



sigma = 0.02
rho = 0.9
T = 300
lag = 60

## Compute simulations
T_simul = 200
G['Z'] = {};
G['Z']['Z'] = np.identity(T)    # add shock itself to G matrix to compute simulated path
outputs = ['Z', 'Y', 'C', 'K']  # define outputs to compute
np.random.seed(2)

# Simulation 
# T is the length of the impulse responses 
# T_simul is simulation length

inputs = ['Z']
sigmas = {'Z': sigma}
rhos = {'Z': rho}

epsilons = {i: np.random.randn(T_simul + T - 1) for i in inputs}
simulations = {}
for o in outputs:
    dXs = {i: sigmas[i] * (G[o][i] @ (rhos[i] ** np.arange(T))) for i in inputs}

    i = 'Z'
    dX_flipped = dXs[i][::-1].copy()  # flip so we don't need to flip epsilon
    T = len(dXs[i])
    T_simul = len(epsilons[i])
    Y = np.empty(T_simul - T + 1)
    for t in range(T_simul - T + 1):
        Y[t] = np.vdot(dX_flipped, epsilons[i][t:t + T])

    simulations[o] = Y

simul = simulations

# Compute impulse response
dZ = rho ** (np.arange(T))
dY, dC, dK = G['Y']['Z'] @ dZ, G['C']['Z'] @ dZ, G['K']['Z'] @ dZ
dX = np.stack([dZ, dY, dC, dK], axis=1)


# Compute covariance
def all_covariances_oneshock(dX, sigma, T):
    dft = np.fft.rfftn(dX, s=(2 * T - 2,), axes=(0,))
    total = sigma ** 2 * (dft.conjugate()[:, :, np.newaxis] * dft[:, np.newaxis, :])
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]

def all_covariances(M, sigmas):
    """Use Fast Fourier Transform to compute covariance function between O vars up to T-1 lags.

    See equation (108) in appendix B.5 of paper for details.

    Parameters
    ----------
    M      : array (T*O*Z), stacked impulse responses of nO variables to nZ shocks (MA(T-1) representation) 
    sigmas : array (Z), standard deviations of shocks

    Returns
    ----------
    Sigma : array (T*O*O), covariance function between O variables for 0, ..., T-1 lags
    """
    T = M.shape[0]
    dft = np.fft.rfftn(M, s=(2 * T - 2,), axes=(0,))
    total = (dft.conjugate() * sigmas**2) @ dft.swapaxes(1, 2)
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]

Sigma = all_covariances_oneshock(dX, sigma, T)

# get sd of each series and correlation
sd = np.sqrt(np.diag(Sigma[0, ...]))
correl = (Sigma / sd) / (sd[:, np.newaxis])

figure_show = 0
if figure_show == 1:
    ## Figures
    T_simul = 200
    # format
    ls = np.arange(-lag, lag + 1)
    corrs_l_positive = correl[:lag + 1, 0, :]
    corrs_l_negative = correl[lag:0:-1, :, 0]
    corrs_combined = np.concatenate([corrs_l_negative, corrs_l_positive])
    
    # Plot simulations
    plt.figure(figsize=(6, 4.5))
    plt.plot(100 * simul['Z'] / ss['Z'], linewidth=2, label=r'$dZ/Z$')
    plt.plot(100 * simul['Y'] / ss['Y'], linewidth=2, label=r'$dY/Y$')
    plt.plot(100 * simul['C'] / ss['C'], linewidth=2, label=r'$dC/C$')
    plt.plot(100 * simul['K'] / ss['K'], linewidth=2, label=r'$dK/K$')
    plt.legend(framealpha=0, loc='upper right')
    plt.ylabel(r'\% deviation from ss')
    plt.xlim(-0, T_simul)
    plt.xlabel(r'Time $t$')
    plt.tight_layout()
    
    
    # Plot moments
    plt.figure(figsize=(6, 4.5))
    plt.plot(ls, corrs_combined[:, 0], linewidth=2, label=r'$dZ$')
    plt.plot(ls, corrs_combined[:, 1], linewidth=2, label=r'$dY$')
    plt.plot(ls, corrs_combined[:, 2], linewidth=2, label=r'$dC$')
    plt.plot(ls, corrs_combined[:, 3], linewidth=2, label=r'$dK$')
    plt.legend(framealpha=0, loc='upper right')
    plt.xlim(-lag, lag)
    plt.xlabel(r'Lag $l$')
    plt.tight_layout()
    
def hessian(f, x0, nfev=0, f_x0=None, dx=1e-4):
    """Compute Hessian of generic function."""
    n = x0.shape[0]
    Im = np.eye(n)

    # check if function value is given
    if f_x0 is None:
        f_x0 = f(x0)
        nfev += 1

    # compute Jacobian
    J = np.empty(n)
    for i in range(n):
        J[i] = (f_x0 - f(x0 - dx * Im[i, :])) / dx
        nfev += 1

    # compute the Hessian
    H = np.empty((n, n))
    for i in range(n):
        f_xi = f(x0 + dx * Im[i, :])
        nfev += 1
        H[i, i] = ((f_xi - f_x0) / dx - J[i]) / dx
        for j in range(i):
            jac_j_at_xi = (f(x0 + dx * Im[i, :] + dx * Im[j, :]) - f_xi) / dx
            nfev += 1
            H[i, j] = (jac_j_at_xi - J[j]) / dx - H[j, j]
            H[j, i] = H[i, j]

    return H, nfev


# Estimate

#--------------------------------------------------------------------------
# Define series to estimate
series = ['y']
data = aux.get_normalized_data(ss, 'import_export/data/data_bayes.csv', series)

# Define option for estimation
shock_series = [('Z', 2)]  # specifies shock to hit economy and whether it's an AR-1 or ARMA 1-1
x_guess = [0.4, 0.5, 0.4]
priors_list = [('Invgamma', 0.4, 4), ('Beta', 0.5, 0.2), ('Beta', 0.5, 0.2)]
bounds = [(1e-2, 4), (1e-2, 0.98), (1e-2, 0.98)]
outputs = ['Y']

x = x_guess

# x is parameter vector
T_irf = T - 20
n_se, n_sh = len(outputs), len(shock_series)
meas_error = np.zeros(n_se)  # set measurement error to zero
    
# extract shock parameters from x; order: always sigma first, then AR coefs, then MA coefs
# Step 0: Get ARMA
ix, ishock = 0, 0
sigmas, arcoefs, macoefs = np.zeros((3, len(shock_series)))
# in our model, order=2
for shock_name, order in shock_series:
    sigmas[ishock] = x[ix]
    ix += 1
    if order >= 1:
        arcoefs[ishock] = x[ix]
        ix += 1
    if order >= 2:
        macoefs[ishock] = x[ix]
        ix += 1
    ishock += 1
# Step 1
"""Compute the MA representation given G"""
# Compute MA representation of outcomes in As
As = np.empty((T_irf, n_se, n_sh))
for i_sh in range(n_sh):
    # Generates shock IRF for any ARMA process 
    ar_coeff = np.array([arcoefs[i_sh]])
    ma_coeff = np.array([macoefs[i_sh]])
    
    arma_shock = np.empty((T,))
    n_ar = ar_coeff.size
    n_ma = ma_coeff.size
    sign_ma = -1  # this means all MA coefficients are multiplied by -1 (this is what SW etc all have)
    for t in range(T):
        if t == 0:
            arma_shock[t] = 1
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += ar_coeff[i] * arma_shock[t - 1 - i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = ma_coeff[t - 1]
            arma_shock[t] = ar_sum + ma_term * sign_ma
    if np.abs(arma_shock[-1]) > 1e20:
        raise Warning('ARMA shock misspecified, leading to explosive shock path!')
    # store for each series
    shockname = shock_series[i_sh][0]
    for i_se in range(n_se):
        As[:, i_se, i_sh] = (G[outputs[i_se]][shockname] @ arma_shock)[:T_irf]
# Step 2
Sigma = all_covariances(As, sigmas)
# Step 3
# measurement error
sigma_o=meas_error
To, O = data.shape
llh = (ssj_est.log_likelihood(data, Sigma, sigma_o) - (To * O * np.log(2 * np.pi)) / 2)
# compute the posterior by adding the log prior
log_posterior = llh + log_priors(x, priors_list)


#-------------------------------------------------------------------------
if figure_show == 1:    
    # minimize objective
    result = opt.minimize(objective, x_guess, bounds = bounds)
    # Compute standard deviation if required
    
    H, nfev_total = hessian(objective, result.x, nfev=result.nfev, f_x0=result.fun)
    Hinv = np.linalg.inv(H)
    x_sd = np.sqrt(np.diagonal(Hinv))
    x = result.x
    
    #sim = aux.simulate(G, ['Z'], outputs, {'Z': x[0]}, {'Z': x[1]}, T, T_simul)  # Note, just simulate AR(1) for now
        
