from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from proj_one_func import R2score, MSEscore, surface_plot, predict, OLS


# Make data.
steps = 0.05
x = np.arange(0, 1, steps)
y = np.arange(0, 1, steps)
rows = len(x)
cols = len(y)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)


# Plot the surface.
surface_plot(z, 'Raw data of Franke Function')


z = np.ravel(z)
x_ = np.ravel(x)
y_ = np.ravel(y)

# Fifth order design matrix with y and x terms
design = np.c_[np.ones(rows*cols), x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3, \
                                x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4, \
                                x_**5, x_**4*y_, x_**3*y_**2, x_**2*y_**3,x_*y_**4,y_**5]
    
print(design.shape)
betapredict = OLS(z,design,0)
zpredict = predict(rows, cols, betapredict)

surface_plot(zpredict, '$Z_{pred}$ of the Franke Function')
