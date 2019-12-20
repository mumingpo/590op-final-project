import os

import numpy as np

from src import data_utils
from src import sgd

M = 943
N = 1682
OFFSET = 3

def main():
  data_path = os.path.join("data", "u.data")

  x, omega = data_utils.read_data(
    data_path,
    M=M,
    N=N,
    offset=OFFSET
  )
  
  try:
    p = np.load("p.npy")
    q = np.load("q.npy")
    trained_pq = (p, q)
  except FileNotFoundError:
    trained_pq = None
  
  
  ## changing k would require starting with different p and q
  ## delete those files before continuing
  p, q = sgd.stochastic_gd(
    x=x,
    omega=omega,
    lam=0.,
    k=10,
    trained_pq=trained_pq,
    batch_size=943,
    learning_rate=1.)
    
  p.dump("p.npy")
  q.dump("q.npy")
  

if __name__ == "__main__":
  main()