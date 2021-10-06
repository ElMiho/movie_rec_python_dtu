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
  return np.average(
    data[user][data[user] != 0]
  )

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

def euclidean(user1, user2, data):
  sum = np.sum(
    (data[user1] - data[user2])**2
  )
  return 1/(1 + np.sqrt(sum))

def generate_sim_matrix(data, method):
  (rows, columns) = np.shape(data)
  l = [
    method(n, k, data)
    for n in range(rows)
    for k in range(rows)
  ]
  return np.array(l).reshape(rows, rows) # Use the list l to create a matrix and then reshape it

def neighbours(user, data, method):
  # Create the similarity matrix
  user_data_sim = generate_sim_matrix(data, method)[user]
  # Get a list with the index and value
  l = [
    (idx, value)
    for (idx, value) in enumerate(user_data_sim)
  ]
  # Removes the user from the list of neighbours
  res = filter(
    lambda i: i[0] != user,
    l 
  )
  return list(res)

def neighbours_sorted(user, data, method):
  res = sorted(
    neighbours(user, data, method),
    # Data format is (index, value) so this makes sure
    # that it sorts based on the value
    key = lambda i: i[1], 
    reverse = True # Closest neighbours first
  )
  return list(res)

def KNN(user, data, k, method):
  # Returns the first k elements
  return neighbours_sorted(user, data, method)[:k]

def prediction_average(user, movie, k, known_data, method):
  k_nearest_neighbours = KNN(user, known_data, k, method)
  res = [
    known_data[id_value[0]][movie]
    for id_value in k_nearest_neighbours
  ]
  return sum(res)/k

def w(user1, user2, data, k, method):
  s_u1_u2 = method(user1, user2, data)
  sum_over_knn = sum(
    [
      id_value[1]
      for id_value in KNN(user1, data, k, method)
    ]
  )
  return s_u1_u2/sum_over_knn

def prediction_weighted_average(user, movie, k, known_data, method):
  knn = KNN(user, known_data, k, method)
  res = sum(
    [
      (w(user, id_value[0], known_data, k, method) * known_data[id_value[0], movie])
      for id_value in knn
    ]
  )
  return res

def prediction_weighted_average_corrected(user, movie, k, known_data, method):
  knn = KNN(user, known_data, k, method)
  res = average(user, known_data) + sum(
    [
      (w(user, id_value[0], known_data, k, method) * (known_data[id_value[0], movie] - average(id_value[0], known_data)))
      for id_value in knn
    ]
  )
  return res

def split_data(user, movie, data):
  (rows, columns) = np.shape(data)
  validation_data = np.array(
    list(data)
  )[:,movie].reshape(rows, 1)
  training_data = np.array(list(data))
  training_data[user, movie] = 0
  return (training_data, validation_data)

def get_movie_raw_array(file):
  csv_file = open(file)
  csv_reader = csv.reader(csv_file)
  list_csv = list(csv_reader)
  array = np.array(list_csv)
  res = np.delete(
    np.delete(array, 0, 0),
    0, 1
  )
  return res

def clean_up_data(array):
  v = np.vectorize(lambda x: 0 if x == '' else int(x))
  return v(array)
   
def get_movie_data(file):
  r = get_movie_raw_array(file)
  return clean_up_data(r)

if __name__ == '__main__':
  data = movie_data
  method = euclidean
  print("Raw data (rows: users, columns: movies): \n" + str(data))
  print("Similarity matrix - all data (rows and columns: users): \n" + str(generate_sim_matrix(data, method)))
  print("\n")

  k = 3 # How many neighbours
  print("Users and its neighbours sorted with k=" + str(k) + str(" (also all data)"))
  (rows, columns) = np.shape(data)
  for user in range(rows):
    print("  User " + str(user) + ": " + str(KNN(user, data, k, method)))
  print("\n")

  # User and their removed movie
  user_removed_movie = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4)
  ]

  print("Prediction - average, weighted average and weighted average corrected")
  for user_movie in user_removed_movie:
    user = user_movie[0]
    movie = user_movie[1]
    (training_data, validation_data) = split_data(user, movie, data)
    actual_rating = validation_data[user][0]
    predicted_rating_average = prediction_average(
      user, 
      movie, 
      k, 
      training_data,
      method
    )
    predicted_rating_weighted_average = prediction_weighted_average(
      user, 
      movie, 
      k, 
      training_data,
      method
    )
    predicted_rating_weighted_average_corrected = prediction_weighted_average_corrected(
      user, 
      movie, 
      k, 
      training_data,
      method
    )
    print("  User: " + str(user) + ", movie: " + str(movie))
    print("    Actual rating: " + str(actual_rating))
    print("    Average rating: " + str(predicted_rating_average))
    print("    Weighted average rating: " + str(predicted_rating_weighted_average))
    print("    Weighted average corrected rating: " + str(predicted_rating_weighted_average_corrected))
