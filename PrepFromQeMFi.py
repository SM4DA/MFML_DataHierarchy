#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:06:02 2024

@author: vvinod
"""

import numpy as np
from sklearn.utils import shuffle



def prep_data(prop='EV',rep='CM'):
    '''
    Function to extract desired data from the QeMFi dataset.

    Parameters
    ----------
    prop : str, optional
        Property of interest to be extracted from QeMFi. The default is 'EV'.
    rep : str, optional
        The type of representation to be used. The default is 'CM'.

    Returns
    -------
    X_train : np.ndarray
        training reps.
    X_val : np.ndarray
        validaiton reps. used only for o-MFML.
    X_test : np.ndarray
        test reps.
    y_train : np.ndarray
        training energie across all fidelities.
    y_val : np.ndarray
        validaiton energies of TZVP.
    y_test : np.ndarray
        test energies of TZVP.
    idx_names : np.ndarray
        ID of which molecule is being chosen. 0-urea, 8-ohbdi.

    '''
    
    molnames = ['urea','acrolein','alanine','sma','nitrophenol',
                'urocanic','dmabn','thymine','o-hbdi']
    idx = np.arange(0,15000)
    idx = shuffle(idx,random_state=42)
    X = np.load(f'../QeMFi/MFML/Reps/o-hbdi_{rep}.npy')
    #largest SLATM is for o-hbdi with 6438 features.
    X=np.zeros((135000,X.shape[1]),dtype=float)  
    #Rest will be padded to this size.
    y_all = np.zeros((135000,5),dtype=float)
    
    start=0
    end=15000
    idx_names = np.zeros((135000),dtype=float)
    for i,m in enumerate(molnames):
        names = np.full(15000,i)
        idx_names[start:end] = np.copy(names)
        temp_X = np.load(f'../QeMFi/MFML/Reps/{m}_{rep}.npy')
        X[start:end,:temp_X.shape[-1]] = temp_X[idx,:]
        if prop=='EV':
            temp_data=np.load(f'../QeMFi/dataset/QeMFi_{m}.npz')['EV'][:,:,0]
        elif prop=='SCF':
            temp_data=np.load(f'../QeMFi/dataset/QeMFi_{m}.npz')['SCF']
        y_all[start:end,:] = temp_data[idx,:]
        
        #increment for next molecule
        start+= 15000
        end += 15000
    y_new = np.zeros((5),dtype=object)
    
    X,idx_names = shuffle(X,idx_names,random_state=42)
    y_train = np.zeros((5),dtype=object)
    for i in range(5):
        y_all[:,i] = shuffle(y_all[:,i],random_state=42)
        y_new[i] = y_all[:,i]
        y_train[i] = y_new[i][:120000]
    
    X_train = X[:120000,:]
    #y_train = y_new[:11000,:]
    X_val = X[120000:122000,:]
    y_val = y_new[-1][120000:122000]
    X_test = X[122000:,:]
    y_test = y_new[-1][122000:]
    
    
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_names
    
def main_SCF_SLATM():
    X_train, X_val, X_test, y_train, y_val, y_test, idx_names = prep_data(prop='SCF',rep='SLATM')
    np.save('Data/SLATM_train.npy',X_train)
    np.save('Data/SLATM_val.npy',X_val)
    np.save('Data/SLATM_test.npy',X_test)
    np.save('Data/SLATM_train.npy',X_train)
    np.save('Data/SCF_train.npy',y_train)
    np.save('Data/SCF_val.npy',y_val)
    np.save('Data/SCF_test.npy',y_test)
    np.save('Data/idx_names.npy',idx_names)
    
def main_EV_CM():
    X_train, X_val, X_test, y_train, y_val, y_test, idx_names = prep_data(prop='EV',rep='CM')
    np.save('Data/CM_train.npy',X_train)
    np.save('Data/CM_val.npy',X_val)
    np.save('Data/CM_test.npy',X_test)
    np.save('Data/CM_train.npy',X_train)
    np.save('Data/EV_train.npy',y_train)
    np.save('Data/EV_val.npy',y_val)
    np.save('Data/EV_test.npy',y_test)
    #np.save('Data/idx_names.npy',idx_names)
    

main_EV_CM()