from em import EM

import numpy as np
from scipy.stats import multivariate_normal
import math
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
import os

if __name__ == "__main__":

  if not os.path.exists('plots'):
    os.makedirs('plots')

  plt.rcParams['figure.figsize'] = [15, 15]
  X,y = make_blobs(n_samples=200, centers=[[2.5,3.5],[3.5,5.5],[4,10]])

  em = EM(X,num_clusters=3)
  means,sigmas = em.cluster_means,em.cluster_cov_mtrxs
  iters = 16
  color_vals = ['red','blue','green']
  for i in np.arange(iters):
      print(f'Performing iteration {i}/{iters-1}..')

      plt.clf()
      plt.axis([-5, 30, -5, 30])
      plt.scatter(X[:,0],X[:,1],marker='o', c=y,
                      s=35, edgecolor='k')
      j = 0
      for mu,cov in zip(means,sigmas):
          x_plt, y_plt = np.mgrid[mu[0] - 3*cov[0,0]:mu[0] +3*cov[0,0]:.01, mu[1] - 3*cov[1,1]:mu[1] + 3*cov[1,1]:.01]
          pos = np.empty(x_plt.shape + (2,))
          pos[:, :, 0] = x_plt 
          pos[:, :, 1] = y_plt
          rv = multivariate_normal(mu, cov)
          plt.contour(x_plt, y_plt, rv.pdf(pos),colors = color_vals[j])
          j+=1
      plt.savefig(f'plots/frame{i}.png')
      means,sigmas = em.iterate()
