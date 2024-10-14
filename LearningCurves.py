#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:57:28 2024

@author: vvinod
"""

import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML
from sklearn.utils import shuffle
import qml.kernels as k
from qml.math import cho_solve


def KRR(X_train:np.ndarray, X_test:np.ndarray, 
        y_train:np.ndarray, y_test:np.ndarray, 
        k_type:str='laplacian', sigma:float=30.0, reg:float=1e-10):
    '''
    Function to perform KRR (for single fidelity case)

    Parameters
    ----------
    X_train : np.ndarray
        Training reps.
    X_test : np.ndarray
        Test reps.
    y_train : np.ndarray
        training energies.
    y_test : np.ndarray
        test energies.
    k_type : str, optional
        The kernel type to be used. The default is 'laplacian'.
    sigma : float, optional
        Kernel width. The default is 30.0.
    reg : float, optional
        Lavrentiev regularizer. The default is 1e-10.

    Returns
    -------
    mae : float
        Mean absolute error calculated over y_test.
    preds : np.ndarray
        Array of predictions of size X_test along axis 0.

    '''
    if k_type=='matern':
        K_train = k.matern_kernel(X_train,X_train,sigma, order=1, metric='l2')
        K_test = k.matern_kernel(X_train,X_test,sigma, order=1, metric='l2')
    elif k_type=='laplacian':
        K_train = k.laplacian_kernel(X_train,X_train,sigma)
        K_test = k.laplacian_kernel(X_train,X_test,sigma)
    elif k_type=='gaussian':
        K_train = k.gaussian_kernel(X_train,X_train,sigma)
        K_test = k.gaussian_kernel(X_train,X_test,sigma)
    elif k_type=='linear':
        K_train = k.linear_kernel(X_train,X_train)
        K_test = k.linear_kernel(X_train,X_test)
    elif k_type=='sargan':
        K_train = k.sargan_kernel(X_train,X_train,sigma,gammas=None)
        K_test = k.sargan_kernel(X_train,X_test,sigma,gammas=None)
    
    #regularize 
    K_train[np.diag_indices_from(K_train)] += reg
    #train
    alphas = cho_solve(K_train,y_train)
    #predict
    preds = np.dot(alphas, K_test)
    #MAE calculation
    mae = np.mean(np.abs(preds-y_test))
    
    return mae, preds

def SF_learning_curve(X_train:np.ndarray, X_test:np.ndarray, 
                      y_train:np.ndarray, y_test:np.ndarray, 
                      k_type:str='laplacian',sigma:float=30, 
                      reg:float=1e-10, navg:int=10):
    '''
    Function to generate averaged single fidelity learning curve

    Parameters
    ----------
    X_train : np.ndarray
        Train reps.
    X_test : np.ndarray
        Test reps.
    y_train : np.ndarray
        training energies.
    y_test : np.ndarray
        Test energies.
    k_type : str, optional
        Type of kernel. The default is 'laplacian'.
    sigma : float, optional
        Kernel width. The default is 30.
    reg : float, optional
        Lavrentiev regularizer. The default is 1e-10.
    navg : int, optional
        Number of times to CV across trianing set. The default is 10.

    Returns
    -------
    full_maes : np.ndarray
        Learning Curve MAEs.
    preds : np.ndarray
        Predictions of size X_test across axis 0.

    '''
    full_maes = np.zeros((11),dtype=float)
    for n in tqdm(range(navg),desc='average LC generation'):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in range(1,12):
            temp, preds = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg,k_type=k_type)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes, preds

def LC_routine(y_trains:np.ndarray, indexes:np.ndarray, X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray, 
               y_test:np.ndarray, y_val:np.ndarray, k_type:str='laplacian',sigma:float=200.0, reg:float=1e-10, navg:int=10,
               factor:int=2,nmax:int=10):
    nfids = y_trains.shape[0]
    
    MAEs_OLS = np.zeros((nmax-1),dtype=float) #for OLS MFML
    MAEs_def = np.zeros((nmax-1),dtype=float) # for default MFML
    
    for i in tqdm(range(navg),desc='avg run',leave=False):
        mae_ntr_OLS = []
        mae_ntr_def = []
        for j in range(1,nmax):
            n_trains = (2**j)*np.asarray([factor**(4),factor**(3),factor**2,factor,1])[5-nfids:]
            ###TRAINING######
            model = MFML(reg=reg, kernel=k_type, 
                         order=1, metric='l2', #only used for matern kernel
                         sigma=sigma, p_bar=False)
            
            model.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=True, n_trains=n_trains, seed=i)
            ######default#########
            _ = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='default')
            mae_ntr_def.append(model.mae)
            ##########OLS##########
            _ = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='OLS', copy_X= True, 
                                     fit_intercept= False)
            mae_ntr_OLS.append(model.mae)
            
            
        #store each avg run MAE  
        mae_ntr_OLS = np.asarray(mae_ntr_OLS)
        mae_ntr_def = np.asarray(mae_ntr_def)
        
        MAEs_OLS += mae_ntr_OLS
        MAEs_def += mae_ntr_def
        
    #return averaged MAE
    MAEs_OLS = MAEs_OLS/navg
    MAEs_def = MAEs_def/navg
    return MAEs_OLS, MAEs_def


def varying_baselines(X_train:np.ndarray, X_val:np.ndarray, 
                      X_test:np.ndarray, y_train:np.ndarray, 
                      y_val:np.ndarray, y_test:np.ndarray, indexes:np.ndarray,
                      ker:str='laplacian', sig:float=200.0, reg:float=1-10, 
                      navg:int=1, factor:int=2, nmax:int=10):
    '''
    Function to generate the learnign curves of MFML and o-MFML for different baseline fidelities


    Parameters
    ----------
    X_train : np.ndarray
        Training reps.
    X_val : np.ndarray
        validation reps.
    X_test : np.ndarray
        test reps.
    y_train : np.ndarray
        training energies across all fidelities.
    y_val : np.ndarray
        validaiton energies at target fidelity.
    y_test : np.ndarray
        Test energies at target fidelity.
    indexes : np.ndarray
        Indexes of training reps and corresponding energy locations across fidelities.
    ker : str, optional
        Type of kernel to be used. The default is 'laplacian'.
    sig : float, optional
        Kernel width. The default is 200.0.
    reg : float, optional
        Lavrentiev regularizer. The default is 1-10.
    navg : int, optional
        Number of times to avg across the training set. The default is 1.
    factor : int, optional
        Scaling factor between fidelities. The default is 2.
    nmax : int, optional
        Log2 of maximum number of training samples to be used at the target fidelity keeping in mind the scaling factor. The default is 10.

    Returns
    -------
    None.
    All outputs are saved locally.

    '''
    
    maeols = np.zeros((4),dtype=object)
    maedef = np.zeros((4),dtype=object)
    
    
    for fb in range(1,2):#tqdm(range(4),desc='Baseline loop...'):
        maeols[fb],maedef[fb] = LC_routine(y_trains=y_train[fb:], indexes=indexes[fb:], 
                                           X_train=X_train, X_test=X_test, 
                                           X_val=X_val, y_test=y_test, y_val=y_val, k_type=ker,
                                           sigma=sig, reg=reg, navg=navg,
                                           factor=factor, nmax=nmax)
    
    np.save(f'outs/def_mae_{prop}_{rep}_{factor}.npy',maedef)
    np.save(f'outs/ols_mae_{prop}_{rep}_{factor}.npy',maeols)
    
def main():
    X_train = np.load(f'Data/{rep}_train.npy')
    X_test = np.load(f'Data/{rep}_test.npy')
    X_val = np.load(f'Data/{rep}_val.npy')
    
    y_train = np.load(f'Data/{prop}_train.npy',allow_pickle=True)
    y_test = np.load(f'Data/{prop}_test.npy')
    y_val = np.load(f'Data/{prop}_val.npy')
    
    indexes = np.load('Data/indexes.npy',allow_pickle=True)
    
    #run single fidelity KRR for given molecule
    '''
    sf_maes,_ = SF_learning_curve(X_train=X_train, X_test=X_test, 
                                  y_train=y_train[-1], y_test=y_test, 
                                  k_type=ker,
                                  sigma=sig, reg=reg,
                                  navg=navg) 
    
    np.save(f'outs/sf_mae_{prop}_{rep}.npy',sf_maes)
    '''
    n_list = [12,10,8,7,6,2,2,2,2]
    
    for i in range(2,11): #loop over factors
        if i>6:
            varying_baselines(X_train, X_val, X_test, y_train, y_val, y_test, 
                              indexes=indexes,
                              ker=ker, sig=sig, reg=reg, 
                              navg=navg, factor=i, nmax=n_list[i-2])
    

if __name__=='__main__':
    prop='EV'
    rep='CM'
    ker='matern' #matern usually; gaussian for SLATM SCF
    reg=1e-10
    sig=200.0 #200 for EV(CM) 2200 for SCF(CM) 650 for SCF(SLATM)
    navg=10
    
    main()