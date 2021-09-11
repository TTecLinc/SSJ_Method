# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 02:22:29 2021

@author: Peilin Yang
"""


# Python packages
import numpy as np
import matplotlib.pyplot as plt

# Sequence jacobian package
from toolkit import jacobian as jac
from toolkit import utils

# Models
#from models import herbst_schorfheide as hs
#from models import smets_wouters as sw
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
plt.rc('font', size=10)


import scipy.optimize as opt
from toolkit import het_block as het
from toolkit.simple_block import simple

'''Part 1: HA block'''
def backward_iterate(Va_p, Pi_p, a_grid, e_grid, r, w, beta, eis):  
    
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    
    a = utils.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    
    utils.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)

    return Va, a, c


household = het.HetBlock(backward_iterate, exogenous='Pi', policy='a', backward='Va')

'''Part 2: Simple Blocks'''


@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A):
    asset_mkt = K - A
    return asset_mkt


@simple
def ks_nodag(Y, r, w, K, A, L, Z, alpha, delta):
    res_Y = Z * K(-1) ** alpha * L ** (1 - alpha) - Y
    res_r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta - r
    res_w = (1 - alpha) * Z * (K(-1) / L) ** alpha - w
    asset_mkt = K - A
    return res_Y, res_r, res_w, asset_mkt


'''Part 3: Steady state'''


def ks_ss_endo(lb=0.98, ub=0.999, r=0.01, eis=1, delta=0.025, alpha=0.11, rho=0.966, sigma=0.5, nS=7, nA=150, amax=200,method_name='Endo'):
    
    global coh
    global a_grid
    """Solve steady state of full GE model. Calibrate beta to hit target for interest rate."""
    # set up grid
    a_grid = utils.agrid(amax=amax, n=nA)
    e_grid, _, Pi = utils.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)

    # solve for aggregates analytically
    rk = r + delta
    Z = (rk / alpha) ** alpha  # normalize so that Y=1
    K = (alpha * Z / rk) ** (1 / (1 - alpha))
    Y = Z * K ** alpha
    w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))

    # figure out initializer
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # solve for beta consistent with this
    beta_min = lb / (1 + r)
    beta_max = ub / (1 + r)
    
    beta, sol = opt.brentq(lambda bet: household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=bet, eis=eis,
                                                    Va=Va, method=method_name)['A'] - K, beta_min, beta_max, full_output=True)
    if not sol.converged:
        raise ValueError('Steady-state solver did not converge.')

    # extra evaluation to report variables
    ss = household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=beta, eis=eis, Va=Va, method=method_name)
    ss.update({'Z': Z, 'K': K, 'L': 1, 'Y': Y, 'alpha': alpha, 'delta': delta, 'goods_mkt': Y - ss['C'] - delta * K})

    return ss

ss = ks_ss_endo()
ss_ks_endo = ss

# Parameters for irf
T = 25
cols = np.array(list(range(0,T)))

# Compute jacobian with fake news algorithm
J_endo = jac.get_G([household], ['r', 'w'], [], [], T, ss_ks_endo)

household, inputs, outputs, ss, T, cols=household, ['r', 'w'], ['A','C'], ss_ks_endo, T, cols
 
h=1e-5

# make J as a nested dict where J[o][i] is initially-empty T*T Jacobian
J_direct = {o: {i: np.empty((T, T))*0 for i in inputs} for o in outputs}
# run td once without any shocks to get paths to subtract against (better than subtracting by ss since ss not exact)

td_noshock = {'A': ss['A'], 'C': ss['C']}

for i in inputs:
    # simulate with respect to a shock at each date up to T
    for s in cols:
       
        kwargs = {i: ss[i] + h * (np.arange(T) == s)}


        # infer T from kwargs, check that all shocks have same length
        shock_lengths = [x.shape[0] for x in kwargs.values()]
        T = shock_lengths[0]
        # copy from ss info
        Pi_T = ss[household.exogenous].T.copy()
        
        D = ss['D']
        # allocate empty arrays to store result, assume all like D
        individual_paths = {k: np.empty((T,) + D.shape) for k in household.non_back_outputs | set(household.backward)}
        
        # backward iteration
        backdict = ss.copy()
        for t in reversed(range(T)):
            
            for k, v in kwargs.items():
                backdict.update({k: v[t]})
                
            Va_p = backdict['Va']
            r = backdict['r']
            w = backdict['w']
            
            Pi_p = backdict['Pi']
            a_grid = backdict['a_grid']
            e_grid = backdict['e_grid']
            beta = backdict['beta']
            eis = backdict['eis']

            # Input dictionary
            
            Va, a, c = backward_iterate(Va_p=Va_p, Pi_p=Pi_p, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=beta, eis=eis)
            
            individual = {'Va': Va, 'a':a, 'c': c}
            
            backdict.update({k: individual[k] for k in household.backward})
            
            for k in household.non_back_outputs | set(household.backward):
                individual_paths[k][t, ...] = individual[k]
        
        
        D_path = np.empty((T,) + D.shape)
        D_path[0, ...] = D
        pol = 'a'
        
        for t in range(T-1):
            # have to interpolate policy separately for each t to get sparse transition matrices
            
            sspol_i, sspol_pi = utils.interpolate_coord_robust(a_grid, individual_paths[pol][t, ...])

            D_local = D_path[t, ...]
            nZ, nX = D_local.shape
            Dnew = np.zeros_like(D_local)
            for iz in range(nZ):
                for ix in range(nX):
                    index = sspol_i[iz, ix]
                    pi = sspol_pi[iz, ix]
                    d = D_local[iz, ix]
                    Dnew[iz, index] += d * pi
                    Dnew[iz, index+1] += d * (1 - pi)

            D_path[t+1, ...] = Pi_T @ Dnew

        aggregates = {}
        # obtain aggregates of all outputs, made uppercase
        for k in household.non_back_outputs:
            T = D_path.shape[0]
            Xnew = D_path.reshape(T, -1)
            Ynew = individual_paths[k].reshape(T, -1)
            Z = np.empty(T)
            for t in range(T):
                Z[t] = Xnew[t, :] @ Ynew[t, :]
            aggregates[k.upper()] = Z

        td_out = aggregates

        # store results as column t of J[o][i] for each outcome o
        for o in outputs:
            J_direct[o][i][:, s] = (td_out[o] - td_noshock[o]) / h

J_direct_endo = J_direct


rho=0.8
dr = ss_ks_endo['r'] * rho**np.arange(T)

shock_Cr_endo=J_endo['C']['r']@dr
shock_Cr_D_endo=J_direct_endo['C']['r']@dr
plt.plot(shock_Cr_endo,'g-.',label="Fake Endogenous")
plt.plot(shock_Cr_D_endo,'bs',label="Direct Endogenous")

plt.legend()

