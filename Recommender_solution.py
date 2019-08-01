
"""
SNU CSE fRONTIER 
"""

import pandas as pd
import numpy as np

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    #print(user_data.head())
    
    # left: use only keys from left frame, similar to a SQL left outer join; preserve key order
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )
    #print(user_full.head())
    
    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    # remove already rated movie row
    recommendations = movies_df[~movies_df["MovieID"].isin(user_full['MovieID'])]
    
    # merge with prediction and movie information
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID')
    
    #rename userID column to prediction
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'})
    
    #sorting prediction value to descending order and 
    recommendations = recommendations.sort_values(['Predictions'], ascending = False)
    recommendations = recommendations.iloc[:num_recommendations, :-1]

    return user_full, recommendations

if __name__ == "__main__":
    
    #1. Loading the Data
    ratings_list = [i.strip().split("::") for i in open('./ml-1m/ratings.dat', 'r').readlines()]
    users_list = [i.strip().split("::") for i in open('./ml-1m/users.dat', 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open('./ml-1m/movies.dat', 'r',encoding = "ISO-8859-1").readlines()]

    print(ratings_list[0])
    print(users_list[0])
    print(movies_list[0])

    #Data type conversion to numpy array and pandas DataFrame
    ratings = np.array(ratings_list)
    users = np.array(users_list)
    movies = np.array(movies_list)

    ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])

    #convert string data type to int64 
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
    
    #check dataframe
    print(movies_df.head())
    print(ratings_df.head())

    #2. Make pivot table
    R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    print(R_df.head())

    #3. Normalize by each users mean convert it from a dataframe to a numpy array
    # DataFrame type to matrix
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_normalized = R - user_ratings_mean.reshape(-1, 1)
    
    #4. Singular Value Decomposition
    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(R_normalized, k = 50)
    sigma = np.diag(sigma)
    
    #5. Making Predictions
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    #6. Making Movie Recommendations
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
    print(preds_df.head())
    
    already_rated, predictions = recommend_movies(preds_df, 837, movies_df, ratings_df, 10)
    print(already_rated.head(10))
    print(predictions)