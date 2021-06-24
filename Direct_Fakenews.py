# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:59:04 2021

@author: Peilin Yang
"""

# Python packages
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Sequence jacobian package
from toolkit import jacobian as jac
from toolkit.solved_block import solved
from toolkit import estimation as ssj_est
from toolkit import nonlinear

# Models
from models import herbst_schorfheide as hs
from models import smets_wouters as sw
from models import krusell_smith as ks
from models import hank_1a
from models import hank_2a

# Auxiliary functions
from toolkit import aux_fn as aux
from toolkit import aux_speed as aux_jac

# Save output of models?
savedata = True

# MCMC chains
Nsim = 100_000  # number of simulations (200_000 for the paper)
Nburn = 50_000  # number of initial periods thrown away (50_000 for the paper)


# Steady state
ss_ks = ks.ks_ss()
ss_ha = hank_1a.hank_ss()
ss_ha2 = hank_2a.hank_ss(noisy=False)

# Parameters for irf
#cols = np.array([5, 25, 50, 100])
T = 66
cols = np.array(list(range(0,T)))


# Compute jacobian with fake news algorithm
J_ks = jac.get_G([ks.household], ['r', 'w'], [], [], T, ss_ks)
J_ha = jac.get_G([hank_1a.household_trans], ['r', 'w', 'Div', 'Tax'], [], [], T, ss_ha)
J_ha2 = jac.get_G([hank_2a.household_inc], ['ra', 'rb', 'beta', 'w', 'N', 'tax'], [], [], T, ss_ha2)

# Compute jacobian directly
J_ks_direct = aux_jac.get_J_direct(ks.household, ['r', 'w'], ['A','C'], ss_ks, T, cols)

J_ha_direct = aux_jac.get_J_direct(hank_1a.household_trans, ['r', 'w', 'Div', 'Tax'], ['C'], ss_ha, T, cols)
J_ha2_direct = aux_jac.get_J_direct(hank_2a.household_inc, ['ra', 'rb', 'beta', 'w', 'N', 'tax'], ['C'], ss_ha2, T, cols)
rho=0.8
dr = ss_ks['r'] * rho**np.arange(T)

plt.title("C to r")
shock_Cr=J_ks['C']['r']@dr
shock_Cr_D=J_ks_direct['C']['r']@dr
plt.plot(shock_Cr,'r-.',label="Fake")
plt.plot(shock_Cr_D,'bs',label="Direct")
plt.legend()

plt.figure()
plt.title("A to r")
shock_Ar=J_ks['A']['r']@dr
shock_Ar_D=J_ks_direct['A']['r']@dr
plt.plot(shock_Ar,'r-.',label="Fake")
plt.plot(shock_Ar_D,'bs',label="Direct")
plt.legend()
