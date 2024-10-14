import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML

def additionalfactors_LC_routine(X_train, X_test, y_trains, y_test, 
                                 indexes, factor, X_val, y_val, navg=10):
    reg = 1e-10
    sigma=200.0
    k_type='matern'
    maes = 0
    n_trains = ntop*np.asarray([factor**3,factor**2,factor,1])
    #print(n_trains)
    for i in tqdm(range(navg),desc='Avg loop',leave=False):
        model = MFML(reg=reg, kernel=k_type, 
                     order=1, metric='l2', #only used for matern kernel
                     sigma=sigma, p_bar=False)
        model.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=True, n_trains=n_trains, seed=i)
        
        _ = model.predict(X_test = X_test, y_test = y_test, 
                         X_val = X_val, y_val = y_val, 
                         optimiser='OLS', copy_X= True, 
                         fit_intercept= False)
        maes += model.mae
    return maes/navg


def additional_factors(rep='CM',prop='EV'):
    
    X_train = np.load(f'Data/{rep}_train.npy')
    X_test = np.load(f'Data/{rep}_test.npy')
    X_val = np.load(f'Data/{rep}_val.npy')
    
    y_train = np.load(f'Data/{prop}_train.npy',allow_pickle=True)
    y_test = np.load(f'Data/{prop}_test.npy')
    y_val = np.load(f'Data/{prop}_val.npy')
    
    indexes = np.load('Data/indexes.npy',allow_pickle=True)
    maes = []
    factors=np.asarray([2,3,4,5,6,7,8,9,10])
    for factor in tqdm(factors,desc='Looping over factors'):
        m_temp = additionalfactors_LC_routine(X_train =X_train, X_test = X_test,
                                              y_trains = y_train[1:], y_test = y_test,
                                              indexes = indexes[1:], factor=factor, 
                                              X_val = X_val, y_val = y_val, navg=10)
        maes.append(m_temp)
    #print(maes)
    maes = np.asarray(maes)
    np.save(f'outs/{ntop}_gamma_curve_maes.npy',maes)
    
if __name__=='__main__':
    ntop=64
    additional_factors()