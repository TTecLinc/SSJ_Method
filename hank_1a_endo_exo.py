"""
One-asset HANK model

Peilin Yang, Finite Difference Compare 

Endogenous and Exogenous Method

Model DAG:
blocks     = [household, firm, monetary, fiscal, nkpc, mkt_clearing]
unknowns   = [K]
targets    = [asset_mkt]
exogenous  = [Z]
"""

import numpy as np
from numba import vectorize, njit
from toolkit import utils
from toolkit import het_block as het
from toolkit.simple_block import simple

'''Part 1: HA block'''


def backward_iterate(Va_p, Pi_p, a_grid, e_grid, T, w, r, beta, eis, frisch, vphi, c_const, n_const,method, ssflag=False):
    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""
    
    if method=="Endo":
        # this one is useful to do internally
        ws = w * e_grid
    
        # uc(z_t, a_t)
        uc_nextgrid = (beta * Pi_p) @ Va_p
    
        # c(z_t, a_t) and n(z_t, a_t)
        
        # C_{t+1}, N_{t+1}
        c_nextgrid, n_nextgrid = cn(uc_nextgrid, ws[:, np.newaxis], eis, frisch, vphi)
        
        
        # c(z_t, a_{t-1}) and n(z_t, a_{t-1})
        
        # M_{t+1} = C_{t+1} - w*N_{t+1} + A_{t+1} - T
        # M_{t+1} ---> A_{t+1}
        lhs = c_nextgrid - ws[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
        
        # M_{t+2} = (1 + r) * A_{t+1}
        rhs = (1 + r) * a_grid
        
        # M_{t+1} ---> C_{t+1}, M_{t+1} ---> N_{t+1}
        # M_{t+2} ---> C_{t+2}, M_{t+2} ---> N_{t+2}        
        c = utils.interpolate_y(lhs, rhs, c_nextgrid)
        n = utils.interpolate_y(lhs, rhs, n_nextgrid)
        
        # M_{t+2} + w * N_{t+2} + T - C_{t+2} = A_{t+2}
        # test constraints, replace if needed
        a = rhs + ws[:, np.newaxis] * n + T[:, np.newaxis] - c
        iconst = np.nonzero(a < a_grid[0])
        a[iconst] = a_grid[0]
    
        if ssflag:
            # use precomputed values
            c[iconst] = c_const[iconst]
            n[iconst] = n_const[iconst]
        else:
            # have to solve again if in transition
            
            uc_seed = c_const[iconst] ** (-1 / eis)
            c[iconst], n[iconst] = solve_cn(ws[iconst[0]], rhs[iconst[1]] + T[iconst[0]] - a_grid[0], eis, frisch, vphi, uc_seed)
    
        # calculate marginal utility to go backward
        Va = (1 + r) * c ** (-1 / eis)
    
        # efficiency units of labor which is what really matters
        ns = e_grid[:, np.newaxis] * n
    
    if method=="Exo":
        # this one is useful to do internally
        ws = w * e_grid
    
        # uc(z_t, a_t)
        uc_nextgrid = (beta * Pi_p) @ Va_p
        rhs = (1 + r) * a_grid
    
        # c(z_t, a_t) and n(z_t, a_t)
        c_nextgrid, n_nextgrid = cn(uc_nextgrid, ws[:, np.newaxis], eis, frisch, vphi)
       
        # A_{t}
        a_past=(c_nextgrid+ a_grid[np.newaxis, :]- ws[:, np.newaxis] * n_nextgrid - T[:, np.newaxis])/(1+r)
        
        # A_{t} C_{t+1} N_{T+1}--->A_{t+1} C_{t+2} N_{T+2}
        c = utils.interpolate_y(a_past, a_grid, c_nextgrid)
        n = utils.interpolate_y(a_past, a_grid, n_nextgrid)
    
        a = (1 + r) * a_grid+ws[:, np.newaxis] * n + T[:, np.newaxis] - c
        
        iconst = np.nonzero(a < a_grid[0])
        a[iconst] = a_grid[0]
        
        
        
        if ssflag:
            # use precomputed values
            c[iconst] = c_const[iconst]
            n[iconst] = n_const[iconst]
        else:
            # have to solve again if in transition
            uc_seed = c_const[iconst] ** (-1 / eis)
            c[iconst], n[iconst] = solve_cn(ws[iconst[0]], rhs[iconst[1]] + T[iconst[0]] - a_grid[0], eis, frisch, vphi, uc_seed)
    
        # calculate marginal utility to go backward
        Va = (1 + r) * c ** (-1 / eis)
    
        # efficiency units of labor which is what really matters
        ns = e_grid[:, np.newaxis] * n   
        
        
    return Va, a, c, n, ns


# Do not use the decorator here because of the speed test
household = het.HetBlock(backward_iterate, exogenous='Pi', policy='a', backward='Va')


@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


@vectorize
def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc = np.log(uc_seed)
    for i in range(30):
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        if abs(ne) < 1E-11:
            break
        else:
            log_uc -= ne / ne_p
    else:
        raise ValueError("Cannot solve constrained household's problem: No convergence after 30 iterations!")

    return np.exp(log_uc)


@njit
def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc


'''Part 2: Simple blocks and hetinput'''


@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi, phi_y, Y):
    Y_target = 1
    i = rstar + phi * pi
    r = (1 + rstar(-1) + phi * pi(-1) + phi_y * (Y(-1) - Y_target)) / (1 + pi) - 1
    return r, i


@simple
def fiscal(r, B, G):
    Tax = r * B + G
    return Tax


@simple
def nkpc(pi, w, Z, Y, r, mu, kappa, markup):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) + markup - np.log(1 + pi)
    return nkpc_res


@simple
def mkt_clearing(A, NS, C, G, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NS - L
    goods_mkt = Y - C - G - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y
    return asset_mkt, labor_mkt, goods_mkt


def transfers(pi_e, Div, Tax, div_rule, tax_rule):
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


household_trans = household.attach_hetinput(transfers)


@simple
def ha1_nodag(A, NS, C, Y, L, w, r, pi, Div, Tax, Z, rstar, B, phi, mu, kappa):
    res_L = Y / Z - L
    res_Div = Y - w * L - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y - Div
    res_r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1 - r
    res_Tax = r * B - Tax
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) - np.log(1 + pi)
    asset_mkt = A - B
    labor_mkt = NS - L
    goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y
    return res_L, res_Div, res_r, res_Tax, nkpc_res, asset_mkt, labor_mkt, goods_mkt


'''Part 3: Steady state'''
nA_num=16

def hank_ss_exo(beta_guess=0.986, vphi_guess=0.8, r=0.005, eis=0.5, frisch=0.5, mu=1.2, B_Y=5.6, rho_s=0.966, sigma_s=0.5,
            kappa=0.1, phi=1.5, nS=7, amax=150, nA=nA_num,method_name='Exo', tax_rule=None, div_rule=None):
    
    print("method", method_name)
    """Solve steady state of full GE model. Calibrate (beta, vphi) to hit target for interest rate and Y."""

    # set up grid
    a_grid = utils.agrid(amax=amax, n=nA)
    e_grid, pi_e, Pi = utils.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)

    # default incidence rules are proportional to skill
    if tax_rule is None:
        tax_rule = e_grid  # scale does not matter, will be normalized anyway
    if div_rule is None:
        div_rule = e_grid
    assert len(tax_rule) == len(div_rule) == len(e_grid), 'Incidence rules are inconsistent with income grid.'

    # solve analytically what we can
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B
    T = transfers(pi_e, Div, Tax, div_rule, tax_rule)

    # initialize guess for policy function iteration
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # residual function
    def res(x):
        beta_loc, vphi_loc = x
        # precompute constrained c and n which don't depend on Va
        c_const_loc, n_const_loc = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi_loc, Va)
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        out = household_trans.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta_loc,
                                 eis=eis, Div=Div, Tax=Tax, frisch=frisch, vphi=vphi_loc,
                                 c_const=c_const_loc, n_const=n_const_loc, tax_rule=tax_rule, div_rule=div_rule,
                                 ssflag=True, method=method_name)
        return np.array([out['A'] - B, out['NS'] - 1])

    # solve for beta, vphi
    (beta, vphi), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess]), noisy=False)

    # extra evaluation for reporting
    c_const, n_const = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi, Va)
    ss = household_trans.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta, eis=eis,
                            Div=Div, Tax=Tax, frisch=frisch, vphi=vphi, c_const=c_const, n_const=n_const,
                            tax_rule=tax_rule, div_rule=div_rule, method=method_name, ssflag=True)

    # check Walras's law
    walras = 1 - ss['C']
    assert np.abs(walras) < 1E-8

    # add aggregate variables
    ss.update({'B': B, 'phi': phi, 'kappa': kappa, 'Y': 1, 'rstar': r, 'Z': 1, 'mu': mu, 'L': 1, 'pi': 0,
               'walras': walras, 'ssflag': False, 'markup': 0,
               'pi_e': pi_e, 'phi_y': 0, 'Div': Div, 'Tax': Tax, 'div_rule': div_rule, 'tax_rule': tax_rule, 'G': 0})

    return ss






def hank_ss_endo(beta_guess=0.986, vphi_guess=0.8, r=0.005, eis=0.5, frisch=0.5, mu=1.2, B_Y=5.6, rho_s=0.966, sigma_s=0.5,
            kappa=0.1, phi=1.5, nS=7, amax=150, nA=nA_num,method_name='Endo', tax_rule=None, div_rule=None):
    
    print("method", method_name)
    """Solve steady state of full GE model. Calibrate (beta, vphi) to hit target for interest rate and Y."""

    # set up grid
    a_grid = utils.agrid(amax=amax, n=nA)
    e_grid, pi_e, Pi = utils.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)

    # default incidence rules are proportional to skill
    if tax_rule is None:
        tax_rule = e_grid  # scale does not matter, will be normalized anyway
    if div_rule is None:
        div_rule = e_grid
    assert len(tax_rule) == len(div_rule) == len(e_grid), 'Incidence rules are inconsistent with income grid.'

    # solve analytically what we can
    B = B_Y
    w = 1 / mu
    Div = (1 - w)
    Tax = r * B
    T = transfers(pi_e, Div, Tax, div_rule, tax_rule)

    # initialize guess for policy function iteration
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)

    # residual function
    def res(x):
        beta_loc, vphi_loc = x
        # precompute constrained c and n which don't depend on Va
        c_const_loc, n_const_loc = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi_loc, Va)
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        out = household_trans.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta_loc,
                                 eis=eis, Div=Div, Tax=Tax, frisch=frisch, vphi=vphi_loc,
                                 c_const=c_const_loc, n_const=n_const_loc, tax_rule=tax_rule, div_rule=div_rule,
                                 ssflag=True, method=method_name)
        return np.array([out['A'] - B, out['NS'] - 1])

    # solve for beta, vphi
    (beta, vphi), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess]), noisy=False)

    # extra evaluation for reporting
    c_const, n_const = solve_cn(w * e_grid[:, np.newaxis], fininc, eis, frisch, vphi, Va)
    ss = household_trans.ss(Va=Va, Pi=Pi, a_grid=a_grid, e_grid=e_grid, pi_e=pi_e, w=w, r=r, beta=beta, eis=eis,
                            Div=Div, Tax=Tax, frisch=frisch, vphi=vphi, c_const=c_const, n_const=n_const,
                            tax_rule=tax_rule, div_rule=div_rule, method=method_name, ssflag=True)

    # check Walras's law
    walras = 1 - ss['C']
    assert np.abs(walras) < 1E-8

    # add aggregate variables
    ss.update({'B': B, 'phi': phi, 'kappa': kappa, 'Y': 1, 'rstar': r, 'Z': 1, 'mu': mu, 'L': 1, 'pi': 0,
               'walras': walras, 'ssflag': False, 'markup': 0,
               'pi_e': pi_e, 'phi_y': 0, 'Div': Div, 'Tax': Tax, 'div_rule': div_rule, 'tax_rule': tax_rule, 'G': 0})

    return ss
