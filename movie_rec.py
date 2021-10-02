import numpy as np # Library for matrix stuff
import csv # Library for loading csv files

movie_data = np.array(
  [
    [3, 2, 5, 3, 5, 2],
    [4, 5, 4, 4, 5, 4],
    [4, 4, 5, 3, 5, 4],
    [4, 4, 5, 3, 2, 3]
  ]
)

def average(user, data):
  return np.average(data[user])

def pearson(user1, user2, data):
  sum1 = np.sum(
    (data[user1] - average(user1, data)) * (data[user2] - average(user2, data))
  )
  sum2 = np.sqrt(
    np.sum(
      (data[user1] - average(user1, data))**2
    )
  )
  sum3 = np.sqrt(
    np.sum(
      (data[user2] - average(user2, data))**2
    )
  )
  return sum1/(sum2 * sum3)

def generate_sim_matrix(data):
  (rows, columns) = np.shape(data)
  l = [
    pearson(n, k, data)
    for n in range(rows)
    for k in range(rows)
  ]
  return np.array(l).reshape(rows, rows) # Use the list l to create a matrix and then reshape it

def neighbours(user, data):
  user_data_sim = generate_sim_matrix(data)[user]
  l = [
    (idx, user)
    for (idx, user) in enumerate(user_data_sim)
  ]
  res = filter(
    lambda i: i[0] != user,
    l 
  )
  return list(res)

def neighbours_sorted(user, data):
  res = sorted(
    neighbours(user, data),
    key = lambda i: i[1],
    reverse = True
  )
  return list(res)

def KNN(user, data, k):
  return neighbours_sorted(user, data)[:k]

def prediction_average(user, k, known_data, new_data):
  k_nearest_neighbours = KNN(user, known_data, k)
  res = [
    new_data[user[0]][0]
    for user in k_nearest_neighbours
  ]
  return sum(res)/len(k_nearest_neighbours)

def w(user1, user2, data, k):
  s_u1_u2 = pearson(user1, user2, data)
  sum_over_knn = sum(
    [
      id_value[1]
      for id_value in KNN(user1, data, k)
    ]
  )
  return s_u1_u2/sum_over_knn

def prediction_weighted_average(user, k, known_data, new_data):
  knn = KNN(user, known_data, k)
  res = sum(
    [
      (w(user, id_value[0], known_data, k) * new_data[id_value[0]])
      for id_value in knn
    ]
  )
  return res

def prediction_weighted_average_corrected(user, k, known_data, new_data):
  knn = KNN(user, known_data, k)
  res = average(user, known_data) + sum(
    [
      (w(user, id_value[0], known_data, k) * (new_data[id_value[0]] - average(id_value[0], known_data)))
      for id_value in knn
    ]
  )
  return res

def split_data(user, data):
  (rows, columns) = np.shape(data)
  validation_data = data[:,user].reshape(rows, 1)
  training_data = np.delete(data, user, 1)
  return (training_data, validation_data)
   
if __name__ == '__main__':
  print("Raw data (rows: users, columns: movies): \n" + str(movie_data))
  print("Similarity matrix - all data (rows and columns: users): \n" + str(generate_sim_matrix(movie_data)))
  
  k = 3
  print("Users and its neighbours sorted with k=" + str(k) + str(" (also all data)"))
  (rows, columns) = np.shape(movie_data)
  for user in range(rows):
    print("User " + str(user) + ": " + str(KNN(user, movie_data, k)))

  print("\n")
  (training_data, validation_data) = split_data(0, movie_data)
  print("training_data: \n" + str(training_data))
  print("validation_data: \n" + str(validation_data))
  print(prediction_weighted_average_corrected(0, 3, training_data, validation_data))
