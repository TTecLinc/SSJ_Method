# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 21:24:38 2021

@author: Peilin Yang
"""

import numpy as np
import td
import matplotlib.pyplot as plt
import scipy.optimize as opt

from utils import set_parameters, make_path, pack_jacobians, ineq
from production import ss_production
from household import household_ss_olg

from td import td_olg, get_Jacobian, td_GE_olg

def pop_stationary(n, phi, T=100):
    """Computes the stationary population distribution for a given growth rate and vector of survival probabilities."""
    phi_lag = np.append(1, phi[:-1])
    Phi = np.cumprod(phi_lag)
    n_cum = (1 + n) ** np.arange(0, T + 1)
    pi0 = 1 / np.sum(Phi / n_cum)
    pi = Phi / n_cum * pi0

    return pi


def trans_pop(pi0, n, phi, Ttrans=200):
    """ Compute the population transition dynamics for fixed n and phi."""
    pitrans = np.zeros((Ttrans, pi0.shape[0]))
    pitrans[0,:] = pi0
    ntrans = np.zeros((Ttrans, pi0.shape[0]))
    ntrans[0,:] = pi0 / pi0[0]
    for t in np.arange(1, Ttrans):
        ntrans[t,0] = (1 + n) ** t
        ntrans[t,1:] = phi[1:] * ntrans[t-1,:-1]
        pitrans[t,:] = ntrans[t,:] / np.sum(ntrans[t,:])

    return pitrans

params = set_parameters()

show_para=0
if show_para==1:
    plt.figure()
    plt.plot(params['phi'])
    plt.title(r'Survival probabilities $\phi_{i,i+1}$')
    plt.show()
    
    plt.figure()
    plt.plot(params['pi'])
    plt.title(r'Stationnary population distribution $\pi_i$')
    plt.show()
    
    plt.figure()
    plt.plot(params['h'])
    plt.title(r'Age-labor supply profile $\bar{h}_i$')
    plt.show()
    
    plt.figure()
    plt.scatter(params['a'], np.ones_like(params['a']), facecolors='none', edgecolors='k')
    plt.title('Asset grid')
    plt.yticks([])
    plt.show()
    
    plt.figure()
    plt.bar(params['y_eps'], params['pi_eps'], width=0.1)
    plt.title('Idiosyncratic productivity')
    plt.ylabel('Stationary probaility')
    plt.ylabel('Idiosyncratic states')
    plt.xticks(np.round(params['y_eps'],1))
    plt.show()


w, K_L = ss_production(params['r'], params['alpha'], params['delta'])
K = K_L * params['workers']
print(f'Real wage: {w}')
print(f'K/L ratio: {K_L}')

def government(w, retirees, tau=0.2):
    """Computes the level of benefits consistent with budget balance for a given mass of retirees and real wage."""
    d = tau * w / retirees
    return tau, d

tau, d = government(w, params['retirees'])
print(f'Payroll tax: {100*tau: .0f}%')
print(f'Benefits: {d: .2f}')


def error(beta, K, params, w, tau, d):
    params['beta'] = beta
    return household_ss_olg(params, w, tau, d)['A'] - K

params['beta'] = opt.newton(error, x0=0.95, args=(K, params, w, tau, d))
print(f'Calibrated subjective discount factor : {params["beta"]: .2f}')

ss = household_ss_olg(params, w, tau, d)

if show_para==1:
    plt.plot()
    plt.title('Assets')
    plt.plot(ss['A_j'])
    plt.show()
    
    plt.plot()
    plt.title('Consumption')
    plt.plot(ss['C_j'])
    plt.show()
    
    lorenz_a_pctl, lorenz_a = ineq(params, ss)
    
    plt.plot()
    plt.title('Assets Lorenz curve')
    plt.plot(lorenz_a_pctl, lorenz_a)
    plt.plot([0,1],[0,1])
    plt.show()
    
    
n_new = 0.02
pi_old = pop_stationary(params['n'], params['phi'], T=params['T'])
pi_new = pop_stationary(n_new, params['phi'], T=params['T'])

if show_para==1:
    plt.plot(pi_old, label=f'n = {params["n"]}')
    plt.plot(pi_new, label=f'n = {n_new}')
    plt.legend()
    plt.xlabel('Age')
    plt.title('Stationnary population distribution')
    plt.show()
    

Ttrans = 100
Ttrans_full = Ttrans + params['T'] - params['Tw']
pi_trans = trans_pop(params['pi'], n_new, params['phi'], Ttrans=Ttrans_full)

if show_para==1:
    for t in [0, 25, 50, 100]:
        plt.plot(pi_trans[t,:], label=f'Years after shock: {t}')
    plt.plot(pi_new, label='New stationnary dist.', ls='--')
    plt.legend()
    plt.xlabel('Age')
    plt.title('Population distribution')
    plt.show()

paths_trans = {'r':make_path(params['r'], Ttrans)}
D0 = ss['D']

td_PE = td_olg(paths_trans, params, pi_trans, D0)

# Household income
#y_tjs=td.y_tjs
#w_path=td.w

inputs, outcomes = ['r'], ['nad']

get_tran=1
if get_tran==1:
    J = get_Jacobian(paths_trans, params, pi_trans, D0, inputs, outcomes)
    
    # Pack into one matrix
    H_X = pack_jacobians(J, inputs, outcomes, Ttrans)
    
    if show_para==1:
        plt.plot()
        columns = [0, 20, 40]
        for c in columns:
            plt.plot(J['nad']['r'][:, c], label=f'Column {c}: $\partial(A_t-K_t)/r_{{{c}}}$')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel(f'$\partial(A_t-K_t)/\partial r_s$')
        plt.title(f'Columns of the Jacobian')
        plt.show()
        
    td_GE = td_GE_olg(H_X, paths_trans, params, pi_trans, D0, outcomes, inputs, xtol=1E-8, disp=True)
    
    tau = make_path(tau, Ttrans_full)
    tau[20:,0]=0.36

    if show_para==1:
        # Figure fonts
        import seaborn as sns
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
        plt.rc('font', size=10)
        
        # Color palette for optimal black-and-white print
        fig, a = plt.subplots(2,3, squeeze=True)
        cpalette = sns.color_palette("hls", 5)
        sns.palplot(cpalette)
        sns.set_palette("hls", 5)
        #a[0][0].plot(np.arange(0,Ttrans), params['n']*np.ones(Ttrans) + (n_new-params['n']) * (np.arange(0,Ttrans)!=0))
        #a[0][0].set_ylim([0,0.03])
        #a[0][0].set_title('Population growth rate')
        #a[0][0].set_xlabel('Year')
        a[0][0].plot(td_GE['r_full'])
        a[0][0].set_title('Real full rate r')
        a[0][0].set_xlabel('Year')
        
        a[0][1].plot(np.sum(pi_trans[:Ttrans,50:],axis=1))
        a[0][1].set_title('Population over 50+')
        a[0][1].set_xlabel('Year')
        a[0][2].plot(td_GE['r'])
        a[0][2].set_title('Interest rate')
        a[0][2].set_xlabel('Year')
        a[1][0].plot(td_GE['A'], label='A')
        a[1][0].plot(td_GE['K'][:Ttrans], '--', label='K')
        a[1][0].set_title('Asset Market')
        a[1][0].legend()
        a[1][0].set_xlabel('Year')
        a[1][1].plot(tau, label='t')
        a[1][1].set_title('Tax Rate')
        a[1][1].legend()
        a[1][2].plot(td_GE['C'], label='C')
        a[1][2].set_title('Consumption')
        a[1][2].legend()
        
        fig.tight_layout(pad=1.0)
        plt.savefig('td_ge')
        plt.show()
