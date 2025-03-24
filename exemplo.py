# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:53:42 2024

@author: Rodrigo Piccinini (rbpiccinini at gmail)
"""

import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import seaborn as sns

from libesmda import classESMDA

# Defining random seed
#------------------------------------------------------------------------------
rng = np.random.default_rng(318253716843412397083108031079509180189)

# Ensemble size
#------------------------------------------------------------------------------
Ne = 1000

# Number of ES-MDA iterations
#------------------------------------------------------------------------------
Na = 4

# Create prior ensemble
#------------------------------------------------------------------------------
# g(t) = c*exp(-lam*t)
lams = rng.normal(loc=1.0, scale=0.1, size=Ne)
cs = rng.normal(loc=1.0, scale=0.1, size=Ne)

m_prior = np.vstack((lams, cs))

t = np.linspace(0,2,200)
gm_prior = np.exp(-np.einsum('t,i->ti', t, lams))*cs

# Define model
#------------------------------------------------------------------------------
def g(m):
    return m[1]*np.exp(-m[0]*tt)

# Create observed data
#------------------------------------------------------------------------------
tt = np.linspace(0,2,20)
sd = 0.025
d_obs = np.exp(-tt) + rng.normal(loc=0.0, scale=sd, size=len(tt))

# Create data error matrix
#------------------------------------------------------------------------------
Ce = sd**2*np.eye(len(tt))


# Data assimilation
#------------------------------------------------------------------------------
alpha = Na*np.ones(Na)
bounds=np.array([[0,0], [2,2]])
esmda = classESMDA(m_prior, Na, d_obs, Ce, g, bounds=[False], alpha=alpha, sing_val_cutoff=0.99, rng=rng)
esmda.run(run_post=True, inversion='tsvd')

# Create posterior ensemble
#------------------------------------------------------------------------------
lams_post = esmda.m[-1][0,:]
cs_post = esmda.m[-1][1,:]

gm_post = np.exp(-np.einsum('t,i->ti', t, lams_post))*cs_post

# Plot results
#------------------------------------------------------------------------------
# Defining plot parameters
#------------------------------------------------------------------------------

plt.style.use('default')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['figure.titlesize'] = 'small'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['grid.linewidth'] = 0.25
plt.rcParams['grid.linestyle'] = ':'

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True

#------------------------------------------------------------------------------
fig, ax = plt.subplots(2,2)
fig.set_figwidth(140./25.4)
fig.set_figheight(120./25.4)
ax[0,0].plot(t, gm_prior, '-', alpha=0.5, color='grey', lw=0.25)
ax[0,0].plot(t, gm_prior[:,0], '-', alpha=0.5, color='grey', label='prior', lw=0.5)
ax[0,0].plot(t, gm_post, '-', alpha=0.5, color='b', lw=0.25)
ax[0,0].plot(t, gm_post[:,0], '-', alpha=0.5, color='b', label='post', lw=0.5)
ax[0,0].errorbar(tt, d_obs, fmt='ok', yerr=sd, ms=4, mew=1.0, mfc='r', elinewidth=1.0, label='data', alpha=0.75)
ax[0,0].legend()
ax[0,0].set_xlabel('t')
ax[0,0].set_ylabel('$g(t)=c e^{-\lambda t}$')

ax[1,0].boxplot([np.log10(esmda.obj[i]) for i in range(esmda.obj.shape[0])], showfliers=False)
ax[1,0].set_ylabel('Log10 of $O_d(m)$')
ax[1,0].set_xlabel('ES-MDA iterations')


ax[0,1].hist(cs, color='grey', bins=20, alpha=0.5, label='prior')
ax[0,1].hist(cs_post, color='b', bins=20, alpha=0.5, label='post')
ax[0,1].axvline(1.0, color='r', label='true')
ax[0,1].set_xlabel('Coefficient $c$')
ax[0,1].legend()

ax[1,1].hist(lams, color='grey', bins=20, alpha=0.5, label='prior')
ax[1,1].hist(lams_post, color='b', bins=20, alpha=0.5, label='post')
ax[1,1].axvline(1.0, color='r', label='true')
ax[1,1].set_xlabel('Exponent $\lambda$')
ax[1,1].legend()

fig.tight_layout()
fig.savefig('exemplo.png', dpi=200)

