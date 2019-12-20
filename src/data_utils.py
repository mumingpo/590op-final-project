import numpy as np

def read_data(path, M, N, offset=0):
  """
  Read u.data in the default format.
  The memory cost associated with a [943, 1682] matrix of floats are not big,
  so we can still do this.
  "Might" run into trouble for larger datasets,
  where we will need to handle things in batches.
  
  Params:
  M: number of users
  N: number of movies
  offset: center of ratings (to assist in regularization.
  
  Return:
  arr: [M, N] matrix of user ratings of movies
  omega: [M, N] matrix indicating where user rating is valid
  """

  arr = np.zeros([M, N], dtype=np.float)
  omega = np.full([M, N], False, dtype=np.bool)

  with open(path, "rt") as f:
    for line in f:
      if line == "":
        continue
      # fields are "user", "movie", "rating", and "timestamp" respectively in order,
      # delimited by '\t'
      fields = line.split('\t')
      if len(fields) != 4:
        raise ValueError("Data corruption: line contains {}".format(fields))

      user, movie = [int(field) - 1 for field in fields[:2]]
      rating = int(fields[2])
      arr[user][movie] = rating - offset
      omega[user][movie] = True
  
  return arr, omega