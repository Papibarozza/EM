# EM algorithm implementation

This code is accompanying a video that I made which demonstrate the intuitive way of thinking about the EM algorithm.
If you look at the video, the steps as implemented in the code shouldn't be a problem to follow.
The `plot.py` function can help you to plot the result of the algorithm and save figures of the state of the gaussians after each iteration.
It can be useful to study these plots if the program crashes to see what has happened to the gaussians.

*_Note_* : This code lacks serious features that a real implementation would have. The implementation throws an error when it runs into problems stemming from a bad initialization of the parameters so we can observe this in the plots. You will have to keep re-running it until it finds a good starting point for the algorithm. If you are only interesting in finding the cluster means and covariance matrices after a successful run you will need to alter the code accordingly! A quick solution could then be to do:

```python
X,y = make_blobs(n_samples=200, centers=[[2.5,3.5],[3.5,5.5],[4,10]])
em = EM(X,num_clusters=3)
num_iters = 15
i=0
while i<=num_iters:
  print(f'Current iteration {i}/{(num_iters-1)}')
  try:
    means,covariance_mtrxs = em.iterate()
  except:
    print('Bad state reached. Resetting the means and covariances and iteration count..')
    i = 0
    em = EM(X,num_clusters=3)
  i+=1

```
