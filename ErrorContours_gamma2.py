import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML
from sklearn.utils import shuffle

def TZVP_countour_generation(X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray,
                              y_trains:np.ndarray,y_test:np.ndarray, y_val:np.ndarray, 
                              indexes:np.ndarray, k_type:str='laplacian', 
                              sigma:float=200.0, reg:float=1e-10, navg:int=10):
    
    nmax=9 #maximum of 512 samples at TZVP
    full_MAEs_OLS = np.zeros((nmax,nmax),dtype=float) #for OLS MFML
    
    for n in tqdm(range(navg),desc='Average run',leave=True):
        MAEs_OLS = np.zeros((nmax,nmax),dtype=float)
        for i in tqdm(range(1,nmax+1),desc='upper fidelity loop',leave = False):
            for j in tqdm(range(i+1,nmax+2),desc='lower fidelity loop',leave=False):
                n_trains = np.asarray([2**(j+3),2**(j+2),2**(j+1),2**j,2**i])
                model = MFML(reg=reg, kernel=k_type, 
                             order=1, metric='l2', #only used for matern kernel
                             sigma=sigma, p_bar=False)
                model.train(X_train_parent=X_train, 
                            y_trains=y_trains, indexes=indexes, 
                            shuffle=True, n_trains=n_trains, seed=n)
                ##########OLS##########
                _ = model.predict(X_test = X_test, y_test = y_test, 
                                         X_val = X_val, y_val = y_val, 
                                         optimiser='OLS', copy_X= True, 
                                         fit_intercept= False)
                MAEs_OLS[i-1,j-2] = np.copy(model.mae)
        full_MAEs_OLS[:,:] += MAEs_OLS
    
    return full_MAEs_OLS/navg


def SVP_countour_generation(X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray,
                              y_trains:np.ndarray,y_test:np.ndarray, y_val:np.ndarray, 
                              indexes:np.ndarray, k_type:str='laplacian', 
                              sigma:float=200.0, reg:float=1e-10, navg:int=10):
    
    nmax=10 #maximum of 512 samples at TZVP
    full_MAEs_OLS = np.zeros((nmax-1,nmax-1),dtype=float) #for OLS MFML
    
    for n in tqdm(range(navg),desc='Average run',leave=True):
        MAEs_OLS = np.zeros((nmax-1,nmax-1),dtype=float)
        for i in tqdm(range(1,nmax+1),desc='upper fidelity loop',leave = False):
            for j in tqdm(range(i+1,nmax+1),desc='lower fidelity loop',leave=False):
                n_trains = np.asarray([2**(j+3),2**(j+2),2**(j+1),2**(i+1),2])
                model = MFML(reg=reg, kernel=k_type, 
                             order=1, metric='l2', #only used for matern kernel
                             sigma=sigma, p_bar=False)
                model.train(X_train_parent=X_train, 
                            y_trains=y_trains, indexes=indexes, 
                            shuffle=True, n_trains=n_trains, seed=n)
                ##########OLS##########
                _ = model.predict(X_test = X_test, y_test = y_test, 
                                         X_val = X_val, y_val = y_val, 
                                         optimiser='OLS', copy_X= True, 
                                         fit_intercept= False)
                MAEs_OLS[i-1,j-2] = np.copy(model.mae)
        full_MAEs_OLS[:,:] += MAEs_OLS
    
    return full_MAEs_OLS/navg

def G631_countour_generation(X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray,
                              y_trains:np.ndarray,y_test:np.ndarray, y_val:np.ndarray, 
                              indexes:np.ndarray, k_type:str='laplacian', 
                              sigma:float=200.0, reg:float=1e-10, navg:int=10):
    
    nmax = 11 #maximum of 512 samples at TZVP
    full_MAEs_OLS = np.zeros((nmax-2,nmax-2),dtype=float) #for OLS MFML
    
    for n in tqdm(range(navg),desc='Average run',leave=True):
        MAEs_OLS = np.zeros((nmax-2,nmax-2),dtype=float)
        for i in tqdm(range(1,nmax+1),desc='upper fidelity loop',leave = False):
            for j in tqdm(range(i+1,nmax),desc='lower fidelity loop',leave=False):
                n_trains = np.asarray([2**(j+3),2**(j+2),2**(i+2),4,2])
                model = MFML(reg=reg, kernel=k_type, 
                             order=1, metric='l2', #only used for matern kernel
                             sigma=sigma, p_bar=False)
                model.train(X_train_parent=X_train, 
                            y_trains=y_trains, indexes=indexes, 
                            shuffle=True, n_trains=n_trains, seed=n)
                ##########OLS##########
                _ = model.predict(X_test = X_test, y_test = y_test, 
                                         X_val = X_val, y_val = y_val, 
                                         optimiser='OLS', copy_X= True, 
                                         fit_intercept= False)
                MAEs_OLS[i-1,j-2] = np.copy(model.mae)
        full_MAEs_OLS[:,:] += MAEs_OLS
    
    return full_MAEs_OLS/navg


def G321_countour_generation(X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray,
                              y_trains:np.ndarray,y_test:np.ndarray, y_val:np.ndarray, 
                              indexes:np.ndarray, k_type:str='laplacian', 
                              sigma:float=200.0, reg:float=1e-10, navg:int=10):
    
    nmax = 12 #maximum of 512 samples at TZVP
    full_MAEs_OLS = np.zeros((nmax-3,nmax-3),dtype=float) #for OLS MFML
    
    for n in tqdm(range(navg),desc='Average run',leave=True):
        MAEs_OLS = np.zeros((nmax-3,nmax-3),dtype=float)
        for i in tqdm(range(1,nmax+1),desc='upper fidelity loop',leave = False):
            for j in tqdm(range(i+1,nmax-1),desc='lower fidelity loop',leave=False):
                n_trains = np.asarray([2**(j+3),2**(i+3),8,4,2])
                model = MFML(reg=reg, kernel=k_type, 
                             order=1, metric='l2', #only used for matern kernel
                             sigma=sigma, p_bar=False)
                model.train(X_train_parent=X_train, 
                            y_trains=y_trains, indexes=indexes, 
                            shuffle=True, n_trains=n_trains, seed=n)
                ##########OLS##########
                _ = model.predict(X_test = X_test, y_test = y_test, 
                                         X_val = X_val, y_val = y_val, 
                                         optimiser='OLS', copy_X= True, 
                                         fit_intercept= False)
                MAEs_OLS[i-1,j-2] = np.copy(model.mae)
        full_MAEs_OLS[:,:] += MAEs_OLS
    
    return full_MAEs_OLS/navg


def main():
    X_train = np.load(f'Data/{rep}_train.npy')
    X_test = np.load(f'Data/{rep}_test.npy')
    X_val = np.load(f'Data/{rep}_val.npy')
    
    y_train = np.load(f'Data/{prop}_train.npy',allow_pickle=True)
    y_test = np.load(f'Data/{prop}_test.npy')
    y_val = np.load(f'Data/{prop}_val.npy')
    
    indexes = np.load('Data/indexes.npy',allow_pickle=True)
    
    contour = TZVP_countour_generation(X_train=X_train, X_test=X_test, X_val=X_val,
                                        y_trains=y_train, y_test=y_test, y_val=y_val, 
                                        indexes=indexes, k_type=ker, 
                                        sigma=sig, reg=reg, navg=10)
    np.save(f'outs/{factor}_contour_TZVP_SVP.npy',contour)
    
    contour = SVP_countour_generation(X_train=X_train, X_test=X_test, X_val=X_val,
                                        y_trains=y_train, y_test=y_test, y_val=y_val, 
                                        indexes=indexes, k_type=ker, 
                                        sigma=sig, reg=reg, navg=10)
    np.save(f'outs/{factor}_contour_SVP_631G.npy',contour)
    
    contour = G631_countour_generation(X_train=X_train, X_test=X_test, X_val=X_val,
                                        y_trains=y_train, y_test=y_test, y_val=y_val, 
                                        indexes=indexes, k_type=ker, 
                                        sigma=sig, reg=reg, navg=10)
    np.save(f'outs/{factor}_contour_631G_321G.npy',contour)
    
    
    contour = G321_countour_generation(X_train=X_train, X_test=X_test, X_val=X_val,
                                        y_trains=y_train, y_test=y_test, y_val=y_val, 
                                        indexes=indexes, k_type=ker, 
                                        sigma=sig, reg=reg, navg=10)
    np.save(f'outs/{factor}_contour_321G_STO3G.npy',contour)
    
    
    
if __name__=='__main__':
    prop='EV'
    rep='CM'
    ker='matern' #matern usually; gaussian for SLATM SCF
    reg=1e-10
    sig=200.0 #200 for EV(CM) 2200 for SCF(CM) 650 for SCF(SLATM)
    navg=10
    
    factor=2 #3,4,5,6
    
    main()
    