# -*- coding: utf-8 -*-

"""
Small library for applications of the ES-MDA method in reservoir engineering.

Reference:

[1] Lacerda, J. M., Emerick, A. A., & Pires, A. P. (2019). Methods to mitigate loss of variance due to sampling errors in ensemble data assimilation with non-local model parameters.
    Journal of Petroleum Science and Engineering, 172, 690-706.
"""

import numpy as np
import scipy.linalg as spl
from tqdm import tqdm, trange

# Sets global seed

class classESMDA:
    """Class to run ES-MDA method.

            Dimensions
            ----------
            Na : int
                Number of data assimilations.

            Nd : int
                Number of observations.

            Ne : int
                Number of models in the ensemble (ensemble size).

            Nm : int
                size of the vector with model parameters.

            Arrays
            ------

            Cmd : numpy.array (Nm, Nd)
                Covariance matrix between model parameters and observed data with shape (Nm, Nd).

            Cdd : numpy.array (Nd, Nd)
                Covariance matrix of observed data with shape (Nd, Nd).

            e : numpy.array (Nd,)
                Random perturbations vector ~ Normal(0, \alpha Ce) with shape (Nd,).

            m : numpy.array (Ne, Nm)
                Ensemble of models (each row is a vector of model parameters) with shape (Ne, Nm).

            gm : numpy.array (Ne, Nd)
                Matrix of predicted data from the ensemble of models "m" with shape (Ne, Nd).

            d_obs :  numpy.array (Nd,).
                Vector with observed data.

            Ce :  numpy.array (Nd, Nd).
                Covariance matrix of observed-data error.

           Functions
           ---------

            g : takes argument "mj" numpy.array (Nm,) and returns "d" numpy.array (Nd,).
    """

    def __init__(self, m, Na, d_obs, Ce, g, alpha, bounds=[False], sing_val_cutoff=0.99,
                 localization=0., rng=np.random.default_rng(None),
                 update=1.0):
        """Prepares ES-MDA loop.
        """

        # save the random number generator
        self.rng = rng
        
        # read observed data and exclude np.nan
        self.d_obs_not_null = ~np.isnan(d_obs)
        self.d_obs = d_obs[self.d_obs_not_null]
        self.Nd = self.d_obs.shape[0]
        self.d_obs = self.d_obs.reshape([self.Nd, 1])
        
        self.Na = Na
        self.Nm = m.shape[0]
        self.Ne = m.shape[1]
        print('> Nd = {:4d}'.format(self.Nd))
        print('> Nm = {:4d}'.format(self.Nm))
        print('> Ne = {:4d}'.format(self.Ne))
        

        self.m = np.zeros([self.Na+1, self.Nm, self.Ne])
        self.m[0,:,:] = m
        self.g = g
        
        self.alpha = alpha

        self.gm = np.zeros([self.Na+1, self.Nd, self.Ne])
        self.dd = np.zeros([self.Nd, self.Ne])
        self.dm = np.zeros([self.Nm, self.Ne])
        self.obj = np.zeros([self.Na+1, self.Ne])
        self.nv = np.zeros(self.Na+1)
        self.Ce = Ce[self.d_obs_not_null,:][:, self.d_obs_not_null]
        
        # Subspace inversion
        self.S = np.diag(np.diag(self.Ce)**0.5)
        self.Sinv = np.diag(1.0/np.diag(self.S))
        self.Ce_tilda = self.Sinv @ self.Ce @ self.Sinv
        
        # C matrix singular values
        self.sing_vals = []
        self.sing_val_cutoff = sing_val_cutoff
        
        self.bounds = bounds
        
        # Localization matrix
        self.localization = localization
        self.R = np.ones([self.Nm, self.Nd])
        
        # Update parameters
        self.update = update
        

    def run(self, run_post=True, inversion='tsvd', anomalies=False):
        print('> Starting ES-MDA.')
        # loop for multiple data assimilations
        for k in range(self.Na):
            print('> Assimilation {:03d}/{:03d}'.format(k+1, self.Na))
            
            # computes ensemble of predicted data
            for j in trange(self.Ne):
                self.gm[k,:,j] = self.g(self.m[k,:,j])[self.d_obs_not_null]
            print('>\t Done running models.')
            
            # check if some model failed
            idx = ~np.isnan(self.gm[k]).any(axis=-2)
            self.gm = self.gm[:,:,idx]
            self.m = self.m[:,:,idx]
            self.Ne = self.m.shape[2]
            print('> Ensemble size is {:3d}'.format(self.Ne))
            
            # Create matrices
            print('>\t Computing delta_m and delta_d.')
            self.dm =  self.m[k]  @ (np.eye(self.Ne) - 1.0/self.Ne * np.ones([self.Ne, self.Ne]))
            self.dd =  self.gm[k] @ (np.eye(self.Ne) - 1.0/self.Ne * np.ones([self.Ne, self.Ne]))           
            self.Cmd = self.dm @ self.dd.T
            

            
            if (k == 0) and (anomalies == True):
                self.dm0 = self.dm.copy()
                self.dd0 = self.dd.copy()
            
            # Nm >= Ne -1
            # self.Cdd = self.dd @ self.dd.T
            
            # if k == 0:
            #     sd_inv_m = np.diag(1.0/np.sqrt(np.diag(self.dm @ self.dm.T)))
            #     sd_inv_d = np.diag(1.0/np.sqrt(np.diag(self.dd @ self.dd.T)))
            #     self.corr_prior =  sd_inv_m @ self.Cmd @ sd_inv_d
            #     np.savetxt('corr_prior_Cmd.txt', self.corr_prior)
            #     np.savetxt('m0.txt', self.m[0])
            #     np.savetxt('d_obs.txt', self.d_obs)
            
            if k == 0 and self.localization > 0:
                print('>\t Computing localization matrix.')
                cii = np.std(self.m[0,:,:], axis=1)
                cjj = np.std(self.gm[0,:,:], axis=1)
                self.cij =  np.abs(np.outer(cii, cjj))
                self.R = np.where(self.Cmd > self.localization*self.cij, 1.0, 0.0)
                

            if inversion == 'subspace':
                print('>\t Computing pseudo inverse matrix (subspace inversion).')

                # Apply TSVD to Sinv * dd
                u, s, vh = np.linalg.svd(self.Sinv @ self.dd, full_matrices=False)
                energy = s.cumsum()/s.sum()
                svd = len(s)
                
                # Apply energy cutoff
                energy = s.cumsum()/s.sum()
                
                s_tsvd = []
                for i in range(len(energy)):
                    s_tsvd.append(s[i])
                    if energy[i] > self.sing_val_cutoff:
                        break
                      
                s = np.array(s_tsvd)
                self.sing_vals.append(s.copy())
                u = u[:,:len(s)]
                vh = vh[:len(s),:]
                tsvd = len(s)
                print('>\t tsvd / svd = {:d} / {:d}'.format(tsvd, svd))
                
                sigma = np.diag(s)
                sigma_inv = np.diag(1.0/s)
                
                B = self.alpha[k]*(self.Ne-1)*sigma_inv@u.T@self.Ce_tilda@u@sigma_inv
                self.B = B
                self.Css = self.S @ u @ sigma @ (np.eye(len(B))+B)@(self.S @ u @ sigma).T
                print('>\t C rank = {:4d}'.format(np.linalg.matrix_rank(self.Css)))
    
                a, lam, ah = np.linalg.svd(B, full_matrices=False)
                self.pinv = (self.Sinv @ u @ sigma_inv @ a) @ np.diag(1.0/(1.0+lam)) @ (self.Sinv @ u @ sigma_inv @ a).T
            
               
            elif inversion=='tsvd':        
                
                # Apply SVD to dm
                # Nm < Ne -1
                u, s, vh = np.linalg.svd(self.dm, full_matrices=False)            
                self.Cdd = self.dd @ self.dd.T
                
                # self.C_tilda = self.Sinv @ self.dd @ self.dd.T @ self.Sinv + self.alpha[k]*(self.Ne-1)*self.Ce_tilda
                self.C_tilda = self.Sinv @ self.Cdd @ self.Sinv + self.alpha[k]*(self.Ne-1)*self.Ce_tilda
                print('>\t C rank = {:4d}'.format(np.linalg.matrix_rank(self.C_tilda)))
                
                print('>\t Computing pseudo inverse matrix (C-matrix TSVD).')
                s, u = np.linalg.eigh(self.C_tilda)
                # sort in descending order
                s = s[::-1]
                u = u[:,::-1]
                svd = len(s)

                # Apply energy cutoff
                energy = s.cumsum()/s.sum()
                
                s_tsvd = []
                for i in range(len(energy)):
                    s_tsvd.append(s[i])
                    if energy[i] > self.sing_val_cutoff:
                        break
                      
                s = np.array(s_tsvd)
                self.sing_vals.append(s.copy())
                u = u[:,:len(s)]
                vh = vh[:len(s),:]
                tsvd = len(s)
                print('>\t tsvd / svd = {:d} / {:d}'.format(tsvd, svd))


                print('>\t pinv cond = {:4.3e}'.format((1.0/s).max()/(1.0/s).min()))
                self.pinv = self.Sinv @ (u @ np.diag(1.0/s) @ u.T ) @ self.Sinv

            
                
            elif inversion=='numpy.linalg.pinv':
                self.C_tilda = self.Sinv @ self.dd @ self.dd.T @ self.Sinv + self.alpha[k]*(self.Ne-1)*self.Ce_tilda
                print('>\t C rank = {:4d}'.format(np.linalg.matrix_rank(self.C_tilda)))
                print('>\t Computing pseudo inverse matrix (numpy.linalg.pinv).')
                pinv = np.linalg.pinv(self.C_tilda, hermitian=True, rcond=self.sing_val_cutoff)
                s, u = np.linalg.eigh(pinv)
                # sort in descending order
                s = s[::-1]
                u = u[:,::-1]
                svd = len(s)
                self.sing_vals.append(s)
                print('>\t pinv cond = {:4.3e}'.format(s.max()/s.min()))
                self.pinv = self.Sinv @ pinv @ self.Sinv
             
                
               
            # calcula função objetivo (Eq. 14 da ref)
            delta_obs = self.d_obs - self.gm[k] # (Nd, Ne)
            Ceinv = spl.inv(self.Ce)

            self.obj[k, :] = 1.0/(2*self.Nd)*np.array([delta_obs[:,i] @ Ceinv @ delta_obs[:,i].T for i in range(self.Ne)])
            self.nv[k] = self.m[k].var(axis=1).mean()
            print('>\t Mean objective function = {:1.4e}'.format(np.mean(self.obj[k,:])))
            print('>\t Normalized variance = {:1.4e}'.format(self.nv[k]/self.nv[0]))
            print('>\t Current inflation factor = {:1.4e}'.format(self.alpha[k]))
            
            # check whether obj function is finite
            if np.isnan(self.obj[k]).any():
                print('>\t ERROR: Objective function is nan.')
                break
  
            print('>\t Computing Kalman gain and updating parameters.')
            # e = self.rng.normal(loc=0, scale=np.sqrt(np.diag(self.alpha[k]*self.Ce))).reshape([self.Nd,1])
            # e = self.rng.multivariate_normal(np.zeros(self.Nd), self.Ce, size=self.Ne).T
            e = self.rng.multivariate_normal(np.zeros(self.Nd), self.alpha[k]*self.Ce, size=self.Ne).T
            self.KG = (self.Cmd * self.R) @ self.pinv
            self.m[k+1] = self.m[k] + ( (self.Cmd * self.R) @ self.pinv @ (self.d_obs + e - self.gm[k]) ) * self.update
            
            # check matrices health
            print('>\t Cmd rank = {:4d}'.format(np.linalg.matrix_rank(self.Cmd)))
            
            # apply limits to parameters
            if len(self.bounds) > 1:
                print('>\t Applying parameter bounds.')
                self.m[k+1,:,:] = np.clip(self.m[k+1].T, self.bounds[:,0], self.bounds[:,1]).T
            
        # run posterior ensemble
        if run_post:
            print('> Running posterior ensemble.')
            for j in trange(self.Ne):
                # print('>\t Running model {:3d}/{:3d}'.format(j+1, self.Ne))
                self.gm[k+1,:,j] = self.g(self.m[k+1,:,j])[self.d_obs_not_null]

            print('> Done running posterior ensemble. ESMDA ended.')

            # computes objective function (Eq. 14)
            delta_obs = self.d_obs - self.gm[k+1] # (Ne, Nd)
            Ceinv = spl.inv(self.Ce)
            # self.obj[k+1] = (0.5*np.dot( np.dot(delta_obs, Ceinv).T, delta_obs) / self.Nd).flatten().sum()/self.Ne
            self.obj[k+1, :] = 1.0/(2*self.Nd)*np.array([delta_obs[:,i]@Ceinv@delta_obs[:,i].T for i in range(self.Ne)])
            self.nv[k+1] = self.m[k+1].var(axis=1).mean()
            print('>\t Final mean objective function = {:1.4e}'.format(np.mean(self.obj[k+1,:])))
            print('>\t Final normalized variance = {:1.4e}'.format(self.nv[k+1]/self.nv[0]))
        else:
            print('> Running of posterior model was disabled. ESMDA ended.')

    def run_prior(self):
        print('> Runnning prior ensemble.')
        for j in range(self.Ne):
            print('>\t Running model {:3d}/{:3d}'.format(j+1, self.Ne))
            self.gm[0,:,j] = self.g(self.m[0,:,j])

        print('>\t Done running prior ensemble.')
