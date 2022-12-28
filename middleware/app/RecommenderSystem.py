import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ContentBasedRecommender:
    '''
    Contains a structure for getting recommendations based on a matrix and a movie title or id.

    Parameters
    ----------
    mapping_matrix : pd.DataFrame - A DataFrame containing the connection between the movieId, the tmdb_id and th movie title. The mapping_matrix dictates the order of the movies in all the other matrices. The shape of the DataFrame is (n_movies, 3).

    description_matrix : np.array - A Matrix containing the embeddings of the movie descriptions. The order of the movies is the same as in the mapping_matrix. The shape of the matrix is (n_movies, n_dimensions).

    title_matrix : np.array - A Matrix containing the embeddings of the movie descriptions. The order of the movies is the same as in the mapping_matrix. The shape of the matrix is (n_movies, n_dimensions).

    movie_info_matrix : np.array - A Matrix containing the numerical information (release year, average rating, number of votes, category etc.) for all movies. The order of the movies is the same as in the mapping_matrix. The shape of the matrix is (n_movies, n_columns).

    cast_info_matrix: np.array - A Matrix containing the embeddings of the cast information. The order of the movies is the same as in the mapping_matrix. The shape of the matrix is (n_movies, n_cast_infos).

    utility_matrix: np.array - A matrix containing a already calculated Sparse Rating Matrix. 

    ratings: pd.DataFrame - A DataFrame containing ratings in the long form with the shape (n_ratings, (userId,  (userId, movieId, rating)))

    Notes
    -----
    Similarity matrices are calculated by using the cosine similarity. The calculation happens during the call of the build_similarity_matrices method. The similarity matrices are stored in the class as attributes.

    The different similarities are then combined by using a weighted average. The weights are given by a input dictionary. 

    '''

    def __init__(self, mapping_matrix: pd.DataFrame, ratings: pd.DataFrame = None, description_matrix: np.array = None, title_matrix: np.array = None, movie_info_matrix: np.array = None, cast_info_matrix: np.array = None, scaling_function=sklearn.preprocessing.normalize, scaling_kwargs={'norm': 'max', 'axis': 0}):
        # initialize the matrices
        self.ratings = ratings
        self.mapping_matrix = mapping_matrix
        self.scaling_kwargs = scaling_kwargs
        self.scaling_function = scaling_function

        if scaling_kwargs == None:
            self.matrices = {'description': description_matrix, 'title': title_matrix, 'info': movie_info_matrix, 'cast': cast_info_matrix}
        else:
            self.matrices = {'description': scaling_function(description_matrix, **self.scaling_kwargs), 'title': scaling_function(
                title_matrix, **self.scaling_kwargs), 'info': scaling_function(movie_info_matrix, **self.scaling_kwargs), 'cast': scaling_function(cast_info_matrix, **self.scaling_kwargs)}

        self.similarity_matrix = None

        # testing if the matrices have the same length
        for key, matrix in self.matrices.items():
            if matrix is not None:
                assert matrix.shape[0] == self.mapping_matrix.shape[
                    0], f"The number of movies in the mapping matrix and the {key} matrix do not match."
                # assert no NaN values
                assert np.isnan(matrix).sum() == 0, f"The {key} matrix contains NaN values."
        print("All matrices have the same length.")
        self.item_matrix = np.hstack(
            [matrix for matrix in self.matrices.values() if matrix is not None])
        print("Item matrix created.")

    def calculate_utility_matrix(self, ratings_long: pd.DataFrame):
        '''
        This function calculates a new sparse rating matrix with the ratings in a long format.
        It calls the self.mapping_matrix to ensure that all movies from the ratings are in the Recommender.

        Params
        ------
            ratings_long:pd.DataFrame - Matrix with the ratings of different users in the shape (ratings, (userId, movieId, rating))
        Returns
        -------
            utility_matrix:np.array - Sparse Rating Matrix
        '''
        # remove ratings for movies that are not in the mapping matrix
        ratings_long = ratings_long[ratings_long.movieId.isin(
            self.mapping_matrix.movieId)]
        print(ratings_long)
        # calculate matrix
        utility_matrix = ratings_long.pivot(
            index='movieId', columns='userId', values='rating')
        # standardize ratings
        # scaler = StandardScaler()
        # utility_matrix = pd.DataFrame(scaler.fit_transform(utility_matrix), columns=utility_matrix.columns, index=utility_matrix.index)

        # for all missing missing movies add row with all nan
        missing_ids = self.mapping_matrix[self.mapping_matrix.movieId.isin(
            utility_matrix.index) == False].movieId
        missing_utility_matrix_rows = pd.DataFrame(
            index=missing_ids, columns=utility_matrix.columns)

        utility_matrix = pd.concat(
            [utility_matrix, missing_utility_matrix_rows])
        utility_matrix = utility_matrix.sort_index()

        # set dtype to float
        utility_matrix = utility_matrix.astype(float)

        # replace nan with 0
        utility_matrix = np.nan_to_num(utility_matrix, nan=1e-10).T

        self.utility_matrix = utility_matrix

        return utility_matrix

    def recommend(self, metric: str = "cosine"):
        '''
        This function calculates user profiles based on the utility matrix and the item matrix. 

        Params
        ------
            metric:str - Metric for calculating pairwise distances. Possible Values: 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'

        Returns
        -------
            recommendations_title:np.array - Array with the titles of the recommended movies in the shape (n_users, n_rec)
            recommendatinos_moviIds:np.array - Array with the movieIds of the recommended movies in the shape (n_users, n_rec)
            predicted_rating:np.array - Array with the predicted ratings for all movies in the shape (n_users, n_movies)
        '''
        # check shape
        assert self.utility_matrix.shape[1] == self.item_matrix.shape[0]

        # calculate user_profiles and distances
        user_profiles = self.utility_matrix @ self.item_matrix

        distances = pairwise_distances(
            user_profiles, self.item_matrix, metric=metric)

        # calculate recommendations
        recommendation_indexes = distances.argsort(axis=1)
        recommendations_title = np.array(self.mapping_matrix.title)[
            recommendation_indexes]
        recommendations_movieIds = np.array(self.mapping_matrix.movieId)[
            recommendation_indexes]

        # calculate prediction for movies that are not rated | ratings (movies, users) = item_matrix (movies, features) @ user_profiles.T (features, users)
        predicted_ratings = user_profiles @ self.item_matrix.T

        # scale predicted ratings between 0, 5
        predicted_ratings = sklearn.preprocessing.minmax_scale(
            predicted_ratings, feature_range=(0, 5), axis=0)

        return recommendations_title, recommendations_movieIds, predicted_ratings

    def eval_recommendations(self, ratings: pd.DataFrame, mask_size: int, rating_threshold: float = 0.5, n_rec: int = 50, random_state: int = 42, scale: bool = False):
        '''
        This function masks 'mask_size' rating for each user, calculates 'n_rec' recommendations and evalutes them based on the masked rating. 
        Users with less than 'min_n_ratings' ratings are exclueded from the evaluation.
        'mask_size' has to be smaller than 'min_n_ratings'.
        Good ratings are defined as ratings above 'rating_threshold' in the possible range of ratings.

        Params
        -------
            ratings:pd.DataFrame - The DataFrame on which the recommendations are based and evaluated on. The DataFrame has to have the columns 'userId', 'movieId' and 'rating'.
            mask_size:int - The number of ratings that are masked for each user.
            ratings_threshold:float - Threshold for the ratings. If the rating is higher than max_rating * rating_threshold the recommendation is considered as good. 
            n_rec:int - Number of recommendations to be made.
            random_state:int - Random state for the random number generator.
            scale:bool - Scaling of ratings for each user. If True the ratings are scaled to have a mean of 0.

        Returns
        -------
            correct_recommendations:int - Number of correct recommendations
            correct_omissions:int - Number of correct omissions
            false_positives:int - Number of false positives
            false_negatives:int - Number of false negatives
            total:int - Total number of recommendations

            recall:float - Recall
            precision:float - Precision
            f1_score:float - F1 Score
            MAE:float - Mean Absolute Error
            RMSE:float - Root Mean Squared Error
            MSE:float - Mean Squared Error
        '''

        # normalize ratings of each user
        if scale:
            for userId in ratings.userId.unique():
                index = list(ratings[ratings.userId == userId].index)
                ratings.loc[index, 'rating'] = ratings.loc[index, 'rating'] - ratings.loc[index, 'rating'].mean()
                
                if self.scaling_kwargs != None:
                    ratings.loc[index, 'rating'] = self.scaling_function(ratings[ratings.userId == userId]['rating'].values.reshape(-1,1), **self.scaling_kwargs)

            # assert no nan values
            assert ratings.isna().sum().sum() == 0

        # setup variables
        MAEs, RMSEs, MSEs, MAP = [], [], [], []
        good_recommendations, false_positives, false_negatives, correct_omissions, hits = 0, 0, 0, 0, 0

        # create threshold
        max_rating, min_rating = ratings.rating.max(), ratings.rating.min()
        threshold = abs(max_rating - min_rating) * rating_threshold

        # initiate dataframes
        test_data = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        train_data = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

        # mask ratings
        for userId in ratings.userId.unique():
            # get ratings of user
            user_ratings = ratings[ratings.userId == userId]

            # split ratings into train and test
            train_ratings, test_ratings = train_test_split(
                user_ratings, test_size=mask_size, random_state=random_state)

            # append to dataframes
            test_data = pd.concat([test_data, test_ratings])
            train_data = pd.concat([train_data, train_ratings])

        # calculate utility matrix
        self.calculate_utility_matrix(ratings_long=train_data)

        # calculate recommendations
        recommendation_title, recommendation_movieId, predicted_ratings = self.recommend()

        # iterate over all users
        for i, user_id in enumerate(ratings.userId.unique()):
            hit = False
            # setup variables for metrics
            avg_precision_at_k = []
            tmp_good_recommendations, tmp_false_positives, tmp_false_negatives, tmp_correct_omissions = 0, 0, 0, 0

            # movie_ids and ratings of the test ratings
            movie_ids_test = test_data[test_data.userId ==
                                       user_id].movieId.values
            ratings_test = test_data[test_data.userId == user_id].rating.values
            test_data_user = test_data[test_data.userId == user_id]

            # movie_ids from train data
            movie_ids_train = train_data[train_data.userId ==
                                         user_id].movieId.values

            # get predicted ratings for user
            predicted_ratings_user = predicted_ratings[i, :]

            # filter out relevant recommendations
            recommendations_movie_id_user = recommendation_movieId[i, :]
            ratings_test_user = predicted_ratings_user[np.isin(
                recommendations_movie_id_user, movie_ids_test)]
            recommendations_movie_id_user = recommendations_movie_id_user[np.isin(
                recommendations_movie_id_user, movie_ids_test)]

            # create temp dataframe for calculating metrics
            tmp_df = pd.DataFrame(
                {'movieId': recommendations_movie_id_user, 'rating': ratings_test_user})
            tmp_df = test_data_user.merge(tmp_df, on='movieId', how='left')
            # check if nan values are present
            assert tmp_df.rating_x.isna().sum() == 0, 'NaN values present in tmp_df'
            assert tmp_df.rating_y.isna().sum() == 0, 'NaN values present in tmp_df'
            # calculate metrics
            MAEs.append(mean_absolute_error(tmp_df.rating_x, tmp_df.rating_y))
            RMSEs.append(np.sqrt(mean_squared_error(
                tmp_df.rating_x, tmp_df.rating_y)))
            MSEs.append(mean_squared_error(tmp_df.rating_x, tmp_df.rating_y))

            # reset recommendations for movie_ids and remove movie_ids the user has already seen
            recommendations_movie_id_user = recommendation_movieId[i, :]
            recommendations_movie_id_user = recommendations_movie_id_user[~np.isin(
                recommendations_movie_id_user, movie_ids_train)][:n_rec]

            # loop over recommendations
            for j, movie_id in enumerate(recommendations_movie_id_user):
                if movie_id in movie_ids_test:
                    if test_data_user[test_data_user.movieId == movie_id].rating.values[0] >= threshold:
                        tmp_good_recommendations += 1
                        avg_precision_at_k.append(tmp_good_recommendations / (j + 1))
                        hit = True
                    else:
                        tmp_false_positives += 1
                else:
                    tmp_false_positives += 1

            # for masked movies not in recommendations
            diff = np.setdiff1d(movie_ids_test, recommendations_movie_id_user)

            if len(diff) > 0:
                for movie_id in diff:
                    if test_data_user[test_data_user.movieId == movie_id].rating.values[0] >= threshold:
                        tmp_false_negatives += 1
                    else:
                        tmp_correct_omissions += 1

            # calculate average precision at k
            if len(avg_precision_at_k) > 0:
                avg_precision_at_k = np.sum(avg_precision_at_k)
            else:
                avg_precision_at_k = 0

            # append to lists
            MAP.append(avg_precision_at_k)

            # update variables
            good_recommendations += tmp_good_recommendations
            false_positives += tmp_false_positives
            false_negatives += tmp_false_negatives
            correct_omissions += tmp_correct_omissions

            if hit:
                hits += 1

        MAE = np.mean(np.array(MAEs))
        RMSE = np.mean(np.array(RMSEs))
        MSE = np.mean(np.array(MSEs))
        precision = good_recommendations / (good_recommendations + false_positives)
        recall = good_recommendations / (good_recommendations + false_negatives)
        accuracy = (good_recommendations + correct_omissions) / (good_recommendations + false_positives + false_negatives + correct_omissions)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        # return good_recommendations, false_positives, false_negatives, correct_omissions
        return {
            'MAE': MAE,
            'RMSE': RMSE,
            'MSE': MSE,
            'RECALL': recall,
            'PRECISION': precision,
            'MAP': np.mean(np.array(MAP)),
            'F1': f1_score,
            'ACCURACY': accuracy,
            'HITS': hits,
            'HIT_RATE': hits / len(test_data.userId.unique())
        }

    def build_similarity_matrix(self, metric: str = 'cosine'):
        '''
        This function calculates the distances between all movies in the item_matrix.

        Params
        ------
            metric:str - Distance Metric
        '''
        self.similarity_matrix = pairwise_distances(
            self.item_matrix, self.item_matrix, metric="cosine")

    def get_recommendations_by_ids(self, movie_ids: np.array, n_rec: int = 10):
        '''
        Returns a list of recommended movies based on the movie_id.

        Parameters
        ----------
        movie_id : np.array - The movieId of the movie for which the recommendations should be calculated.

        n_rec : int, default=10 - The number of recommendations to be returned.

        Returns
        -------
        recommendations : list
            A list of tuples containing the movieId and the title of the recommended movies.
        '''

        if self.similarity_matrix is None:
            self.build_similarity_matrix()

        recommendation_indexes = self.similarity_matrix[movie_ids].argsort(axis=1)[
            :, :n_rec]
        recommendation_title = np.array(self.mapping_matrix.title)[
            recommendation_indexes]
        recommendation_movieIds = np.array(self.mapping_matrix.movieId)[
            recommendation_indexes]

        return recommendation_title, recommendation_movieIds

    def get_recommendations_for_user(self, ratings_user:pd.DataFrame, n_rec:int=10):
        '''
        Returns a lists of recommended movies and titles based on the ratings a user has given. 

        Params
        ------
            ratings_user:pd.DataFrame - A dataframe containing the ratings of a user.
            n_rec:int - The number of recommendations to be returned.

        Returns
        -------
            recommendations_title:list - A list of recommended movie titles.
            recommendations_movieId:list - A list of recommended movie ids.
        '''
        # calculate utilitiy matrix
        self.calculate_utility_matrix(ratings_long=ratings_user)

        # make recommendations
        recommendation_titles, recommendation_movie_ids, predicted_ratings = self.recommend()

        # filter recommendations user has already seen
        recommendation_titles = recommendation_titles[~np.isin(recommendation_movie_ids, ratings_user.movieId.values)]
        recommendation_movie_ids = recommendation_movie_ids[~np.isin(recommendation_movie_ids, ratings_user.movieId.values)]

        return recommendation_titles[:n_rec], recommendation_movie_ids[:n_rec]


    def get_movie_index(self, title: str = None, movie_id: int = None):
        '''
        If available, returns the index of a movie in the mapping matrix for a given movie title or movie id.
        '''
        # return index from title
        if title is not None:
            try:
                return self.mapping_matrix[self.mapping_matrix.title == title].index[0]
            except KeyError:
                print("Movie not found")
                return None
        # return index from movie_id
        elif movie_id is not None:
            try:
                return self.mapping_matrix[self.mapping_matrix.movieId == movie_id].index[0]
            except KeyError:
                print("Movie not found")
                return None

    def get_movie_title(self, movie_id: int = None, index: int = None):
        '''
        If available, returns the title of a movie for a given movie_id or index.
        '''
        # return title from movie_id
        if movie_id is not None:
            try:
                return self.mapping_matrix[self.mapping_matrix.movieId == movie_id].title[0]
            except KeyError:
                print("Movie not found")
                return None
        # return title from index
        elif index is not None:
            try:
                return self.mapping_matrix.title[index]
            except KeyError:
                print("Movie not found")
                return None

    def get_movie_id(self, title: str = None, index: int = None):
        '''
        If available, returns the movieId of a movie for a given title or index.
        '''
        # return movie_id from title
        if title is not None:
            try:
                return self.mapping_matrix[self.mapping_matrix.title == title].movieId[0]
            except KeyError:
                print("Movie not found")
                return None

        # return movie_id from index
        elif index is not None:
            try:
                return self.mapping_matrix.movieId[index]
            except KeyError:
                print("Movie not found")
                return None

    @staticmethod
    def build_movie_info_matrix(movies: pd.DataFrame) -> pd.DataFrame:
        '''
        Extracts numerical info about the movie from the movies data frame.

        Parameters
        ----------
        movies : pd.DataFrame
            The movies data frame.

        Returns
        -------
        movie_info_matrix : np.array
            A data frame containing the movieId, and all other numerical info.
        '''
        movie_info_matrix = movies[["popularity", "vote_average", "vote_count", 'Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance',
                                    'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir']].copy()

        # check for missing values
        assert movie_info_matrix.isnull().sum().sum() == 0

        return np.array(movie_info_matrix)

    @staticmethod
    def build_mapping_matrix(movies: pd.DataFrame) -> pd.DataFrame:
        '''
        Builds the mapping matrix with the columns movieId, tmdbId and title. 

        Params
        ------
            movies pd.DataFrame: Matrix with movie data and the three columns movieId, tmdbId and title

        Returns
        -------
            mapping_matrix pd.DataFrame: 
        '''

        # select columns
        mapping_matrix = movies[['movieId', 'tmdbId', 'title']]

        # testing
        assert mapping_matrix.isnull().sum().sum() == 0
        assert mapping_matrix.duplicated().sum() == 0

        return mapping_matrix

    @staticmethod
    def build_cast_info_matrix(movies: pd.DataFrame) -> pd.DataFrame:
        '''
        This function selects the correct columns from the movies DataFrame and prepares it for the NLP transformation.

            Parameters
            ----------
            movies : pd.DataFrame
                The movies DataFrame.

            Returns
            -------
            cast_info_matrix : pd.DataFrame
                A DataFrame containing the movieId and the cast column.
        '''
        cast_info_matrix = movies[[
            'movieId', 'actor1', 'actor2', 'actor3', 'actor4', 'actor5', 'director1']]
        return cast_info_matrix

    @staticmethod
    def build_title_matrix(movies: pd.DataFrame) -> pd.DataFrame:
        '''
        This function selects the title column and movieId column from the movies DataFrame and prepares it for NLP transformation.
        '''
        title_matrix = movies[['movieId', 'title']]
        return title_matrix
