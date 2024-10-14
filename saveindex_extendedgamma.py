#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:48:11 2024

@author: vvinod
"""

import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML


def save_inds_for_time(X_train:np.ndarray, X_val:np.ndarray, X_test:np.ndarray, 
                       y_train:np.ndarray, y_val:np.ndarray, y_test:np.ndarray, 
                       indexes:np.ndarray, factor:int, nmax:int):
    '''
    Function to save the indexes of the training samples across fidelities. 
    This will be used to calcualte the training set generation time.

    arameters
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
    factor : int, optional
        Scaling factor between fidelities. The default is 2.
    nmax : int, optional
        Log2 of maximum number of training samples to be used at the target fidelity keeping in mind the scaling factor. The default is 10.

    Returns
    -------
    None.

    '''
    n_trains = (2**(nmax-1))*np.asarray([factor**(4),factor**(3),factor**2,factor,1])
    ###TRAINING######
    model = MFML(reg=reg, kernel=ker, 
                 order=1, metric='l2', 
                 sigma=sig, p_bar=False)

    model.train(X_train_parent=X_train, 
                y_trains=y_train, indexes=indexes, 
                shuffle=True, n_trains=n_trains, seed=42)
    
    np.save(f'outs/train_indexes_{factor}.npy',model.indexes)
    
    
    
def main():
    X_train = np.load(f'Data/{rep}_train.npy')
    X_test = np.load(f'Data/{rep}_test.npy')
    X_val = np.load(f'Data/{rep}_val.npy')
    
    y_train = np.load(f'Data/{prop}_train.npy',allow_pickle=True)
    y_test = np.load(f'Data/{prop}_test.npy')
    y_val = np.load(f'Data/{prop}_val.npy')
    
    indexes = np.load('Data/indexes.npy',allow_pickle=True)
    
    n_list = [12,10,8,7,6]
    
    for i in tqdm(range(2,7),desc='loop over scaling factors'):
        save_inds_for_time(X_train, X_val, X_test, y_train, y_val, y_test, 
                           indexes, factor=i, nmax=n_list[i-2])
        
    

        
        
if __name__=='__main__':
    rep='CM'
    prop='EV'
    ker='matern' #matern usually; gaussian for SLATM SCF
    reg=1e-10
    sig=200.0 #200 for EV(CM) 2200 for SCF(CM) 650 for SCF(SLATM)
    main()