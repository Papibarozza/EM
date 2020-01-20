from scipy.stats import multivariate_normal
import scipy.stats as stats
import numpy as np
from numpy.linalg.linalg import LinAlgError

from sklearn.datasets import make_blobs
import math
import time



class EM:

  def __init__(self,X,num_clusters=2, cov_mtrx_diag = 1.0 ):
    self.X = X
    self.num_clusters = num_clusters
    self.min_belonging_ratio = 0.01
    self.min_cov_val = 0.001
    self.max_cov_val = 5
    self.min_mean = 0
    self.max_mean = 10
    self.cluster_means = np.random.randint(self.min_mean,self.max_mean, size = (num_clusters,len(self.X[0]))).astype("float")
    self.cluster_cov_mtrxs = np.array([np.identity(len(X[0])).astype("float") for i in np.arange(num_clusters)])*cov_mtrx_diag
    self.priors = np.ones(num_clusters)/num_clusters
 
  def __str__(self):
    return f"{self.cluster_means},{self.cluster_cov_mtrxs}"
  

  def iterate(self):
    probs = self._expectation()
    return self._maximization(probs)


  def _expectation(self):
    pdf_vals = list()

    for i in np.arange(self.num_clusters):
      try:
        pdf_vals.append(multivariate_normal.pdf(self.X,self.cluster_means[i],self.cluster_cov_mtrxs[i,...]))
      except (LinAlgError, ValueError) as err:
        raise Exception("LinAlgError or ValueError resulting from bad initialization when evaluating pdf. Try to run again with new initialization.") from err
    
    nominator = np.array(pdf_vals)*np.array([self.priors]).T
    denom = np.sum(np.array(pdf_vals)*np.array([self.priors]).T,axis=0)
    probs = nominator/denom
    return probs


  def _maximization(self,probs):

    m = np.sum(probs)

    for i,prob in enumerate(probs):
      m_c = np.sum(prob)
      new_prior = m_c/m
      if(new_prior < self.min_belonging_ratio):
        raise RuntimeWarning('One cluster contains too few points which may be the result of bad initialization.')

      mean_c = np.sum(self.X*np.array([prob]).T,axis = 0)/m_c
      cov_mtrx_c = np.clip(np.dot((np.array(prob).reshape(len(self.X),1)*(self.X-mean_c)).T,(self.X-mean_c))/m_c,self.min_cov_val,self.max_cov_val)
      self.cluster_means[i] = mean_c
      self.cluster_cov_mtrxs[i,:,:] = cov_mtrx_c
      self.priors[i] = new_prior
      
    return self.cluster_means,self.cluster_cov_mtrxs


    

if __name__ == "__main__":
 
  X,Y = make_blobs(n_samples=100,centers=[[3,3], [10,5]])
  em = EM(X)

  for i in np.arange(20):
    em.iterate()
  print(em)
