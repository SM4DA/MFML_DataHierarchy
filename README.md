# Investigating Data Hierarchies in Multifidelity Machine Learning for Excitation Energies
Scripts for multifidelity models used to assess data hierarchy scaling for the prediction of excitation energies of molecules. This also includes scripts to generate the newly introduced Gamma curve. The scripts given here can be used to generate the figures from the manuscript of the same title hosted at [TBA]. The `requirements.txt` file contains all required packages to run the scripts given herein. The dataset used in this work is hosted freely at [this ZENODO repository](https://zenodo.org/records/13925688).

* The python file `Model_MFML.py` is the module that was developed in [this previous work](https://iopscience.iop.org/article/10.1088/2632-2153/ad2cef) and contains both both MFML and o-MFML implementations that are used in this work.
* `PrepFromQeMFi.py` separates the data from the QeMFi dataset into train, test, and validation datasets.
* The jupyter notebook `Plots.ipynb.` offers the different functions to reproduce the plots from the manuscript.
* `LearningCurve.py` generates the data needed to assess the different fixed scaling factors ($\gamma$) 
* The script `RatioTimeBasedScalingFactor.py` produces the learning curves for the scaling factors defined as $\theta_{f-1}^f$.
* `TargetFidelityTimeRatioScalingFactor.py` generates the data for scaling factors defined as $\theta_f^F$ in the manuscript.
* The script `ErrorContours_gamma2.py` will generate the data needed to plot the error contours of MFML (Fig 6 of manuscript).
* `GammaCurve.py` creates all the data points needed to assess the different $\Gamma(N_{train}^{TZVP})$ from the manuscript. The value of `ntop` can be changed based on $N_{train}^{TZVP}$.
* The scripts `saveindexfortimevsmae.py` and `saveindex_extendedgamma.py` are used to get the indices of the training samples used so they can be used to generate the time-cost plots.

