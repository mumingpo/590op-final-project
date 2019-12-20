import numpy as np

DEBUG = True
np.random.seed(0)

def debug(s):
  try:
    if DEBUG:
      print(s, flush=True)
  except NameError:
    pass

def objective(x, omega, p, q, lam):
  obj = lam * (np.linalg.norm(p, ord=2)**2 + np.linalg.norm(q, ord=2)**2)

  x_hat = x @ p @ q
  resid = x[omega] - x_hat[omega]
  obj += np.sum(resid**2)
  
  return obj

def gradient_p(x, omega, p, q, lam):
  gradient = 2 * lam * p

  x_hat = x @ p @ q
  
  for user, movie in zip(*np.where(omega)):
    u = x[user]
    v = q[:, movie]
    uvT = u.reshape([-1, 1]) @ v.reshape([1, -1])
    
    gradient += 2 * (x_hat[user][movie] - x[user][movie]) * uvT

  return gradient

def gradient_q(x, omega, p, q, lam):
  gradient = 2 * lam * q
  
  xp = x @ p
  x_hat = xp @ q

  for user, movie in zip(*np.where(omega)):
    u = xp[user]
    v = q[:, movie]

    gradient[:, movie] += 2 * (x_hat[user][movie] - x[user][movie]) * u
  
  return gradient

def line_search_p(x, omega, p, q, lam, maxit=100, tol=1e-3, in_place=True):

  if in_place is False:
    p = p.copy()

  prev_obj = objective(x, omega, p, q, lam)

  for _ in range(maxit):
    ss = 1.
    gradient = gradient_p(x, omega, p, q, lam)
    
    p_guess = p - ss * gradient
    obj = objective(x, omega, p_guess, q, lam)
    while obj > prev_obj - 0.5 * ss * np.linalg.norm(gradient, ord=2) ** 2:
      ss /= 2
      assert(ss != 0.)
      p_guess = p - ss * gradient
      obj = objective(x, omega, p_guess, q, lam)
    
    p = p_guess
    
    if (abs(prev_obj - obj) / (abs(prev_obj) + 1e-15)) < tol:
      debug("line search for p converged.")
      return p
    
    prev_obj = obj

  debug("line search for p maxit reached.")
  return p

def line_search_q(x, omega, p, q, lam, maxit=100, tol=1e-3, in_place=True):

  if in_place is False:
    q = q.copy()

  prev_obj = objective(x, omega, p, q, lam)

  for _ in range(maxit):
    ss = 1.
    gradient = gradient_q(x, omega, p, q, lam)
    
    q_guess = q - ss * gradient
    obj = objective(x, omega, p, q_guess, lam)
    while obj > prev_obj - 0.5 * ss * np.linalg.norm(gradient, ord=2) ** 2:
      ss /= 2
      assert(ss != 0.)
      q_guess = q - ss * gradient
      obj = objective(x, omega, p, q_guess, lam)
    
    q = q_guess
    
    if (abs(prev_obj - obj) / (abs(prev_obj) + 1e-15)) < tol:
      debug("line search for q converged.")
      return q
    
    prev_obj = obj

  debug("line search for q maxit reached.")
  return q

def stochastic_gd(x, omega, lam, k, trained_pq = None, batch_size=20, learning_rate=1.):
  """
  Perform stochastic gradient descent to find low rank p @ q such that
  x_hat = x @ p @ q is a prediction of all the ratings that a user would
  give to all movies
  
  Params:
  x: sparse matrix of movie ratings of shape [# users, # movies]
  omega: boolean matrix with the same shape as x indicating where x has data
  lam: regularization term limiting the maximum singular value of p, q
  k: rank of the low-rank approximation
  batch_size: number of samples trained for one iteration
  learning_rate: how much p, q should move in respect to the target step
  
  Return:
  p, q: the low rank matrices for which x_hat = x @ p @ q
  """
  
  if not (trained_pq is None):
    p = trained_pq[0]
    q = trained_pq[1]
    #p = np.zeros([x.shape[1], k])
    #q = np.zeros([k, x.shape[1]])
  else:
    p = 0.01 * np.random.normal(size=[x.shape[1], k])
    q = 0.01 * np.random.normal(size=[k, x.shape[1]])
  
  debug("initial objective: {}".format(objective(x, omega, p, q, lam)))
  
  try:
    #for i in range(int(1 / learning_rate) + 1):
    for i in range(10000):
      training_sample_index = np.random.choice(x.shape[0], batch_size, replace=False)
      training_sample_x = x[training_sample_index]
      training_sample_omega = omega[training_sample_index]
      
      rate = learning_rate #(1 / (i + 1))
      
      new_p = line_search_p(
        x=training_sample_x,
        omega=training_sample_omega,
        p=p,
        q=q,
        lam=lam,
        in_place=False
      )
      
      debug("p change: {}".format(np.linalg.norm(new_p - p, ord=2)))
      
      p = (1 - rate) * p + rate * new_p
      #p = (1 - learning_rate) * p + learning_rate * new_p
      #p = new_p
      
      new_q = line_search_q(
        x=training_sample_x,
        omega=training_sample_omega,
        p=p,
        q=q,
        lam=lam,
        in_place=False
      )
      
      debug("q change: {}".format(np.linalg.norm(new_q - q, ord=2)))
  
      q = (1 - rate) * q + rate * new_q
      #q = (1 - learning_rate) * q + learning_rate * new_q
      #q = new_q
      
      debug("objective after iteration #{}: {}".format(i, objective(x, omega, p, q, lam)))
  except KeyboardInterrupt:
    debug("premature exit, p q saved.")
    pass
  finally:
    return p, q