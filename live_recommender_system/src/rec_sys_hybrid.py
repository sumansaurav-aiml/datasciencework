# -*- coding: utf-8 -*-
"""rec_sys_hybrid.py

This module generates three types of recommendation namely Collaborative Filtering,
Past Used top products and Trending products within each portal based on past activity
of users on items and exports as CSV
Finally it merges the three recommendations based on required number of recommended products
and exports as CSV.

Todo:
    * Nothing

"""

# libraries
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
from sklearn import preprocessing

from src.logger import Logger


class RecSysHybrid:
    """Generates recommended item list based on Past Activity of users on items.

    Attributes:
        jobid (int): jobid is unique job run sequence created in database.
        no_of_recommendation (int): How many maximum items will be recommended to each user.
        threshold_from_past (int): How many max recommendation from past can be there in the hybrid.

    """

    def __init__(self, jobid, no_of_recommendation, threshold_from_past, root_file_path):
        """Sets the jobid, logger and file path parameter at the time class initialization

        Args:
            jobid (int): jobid is unique job run sequence created in database.
            no_of_recommendation (int): How many maximum items will be recommended to each user.
            threshold_from_past (int): How many max recommendation from past can be there in the hybrid.
            root_file_path (string): Based on whether running Unit test, this folder will change

        """
        self._jobid = jobid
        #: logger is available in logger.py from python module logging
        self._logger = Logger(__name__, jobid, root_file_path).logger
        #: total number of maximum recommendation we want for each user
        self._no_of_recommendation = no_of_recommendation
        #: for the hybrid recommendation, how many max items should be from past
        self._threshold_from_past = threshold_from_past
        #: Path of the activity data CSV file.
        self._IN_FILE_PATH = './' + root_file_path + '/data/jobid' + str(
            self._jobid) + '/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'
        #: Path where the CSV files will be saved.
        self._OUT_FILE_RELATIVE_PATH = './' + root_file_path + '/data/jobid' + str(self._jobid) + '/predictionfiles/'

    def _reading_data(self):
        """Reads the Activity data CSV and returns it.

        Returns:
            Dataframe having Users Activity data.

        """

        self._logger.info("Reading data from CSV file USER_ACTIVITY_WITH_TIME_PORTAL")
        df = pd.read_csv(self._IN_FILE_PATH)
        # Dropping PortalName column since we have PortalId
        df.dropna(subset=['PORTALNAME'], inplace=True)
        return df

    @staticmethod
    def _combining_activation_download_actions(action):
        """Combines the action of creating campaign as ACTIVATECAMPAIGN and DOWNLOADCAMPAIGN.

        Args:
            action: Activity performed by user.

        Returns:
            a generalized action.

        """

        if action == 'ACTIVATECAMPAIGN':
            return "ACTIVATECAMPAIGN"
        elif action == 'DOWNLOADCAMPAIGN':
            return "ACTIVATECAMPAIGN"
        else:
            return action

    @staticmethod
    def _generalizing_different_downloads(action):
        """Generalizes all kinds of download actions.

        Args:
            action: Activity performed by user.

        Returns:
            a generalized action.

        """
        return "DOWNLOAD" if action.find("DOWNLOAD") >= 0 else action

    @staticmethod
    def _removing_unwanted_actions(df):
        """Drops rows for not required activies from Dataframe.

        Args:
            df: DataFrame having Activity data.

        Returns:
            Dataframe after dropping the not required activities.

        """
        # As we can see that "EnterCampaignStartmarketing" has been done only once, this seems to be a wrong data,
        # lets remove it.
        df = df.loc[df['ACTIVITY_TYPE'] != 'ENTERCAMPAIGNSTARTMARKETING']
        # Also lets remove the "Close" actions
        df = df.loc[df['ACTIVITY_TYPE'] != 'CLOSEASSETPREVIEW']
        df = df.loc[df['ACTIVITY_TYPE'] != 'CLOSESETUPASSETS']
        return df

    @staticmethod
    def _combining_pull_actions(action):
        """Generalizes Pull, Video play and Download of asset actions..

        Args:
            action: Activity performed by user.

        Returns:
            a generalized action.

        """
        if action == 'VIDEO_GETDEFAULTEMBEDCODE':
            return "PULL"
        elif action == 'DOWNLOAD':
            return "PULL"
        else:
            return action

    @staticmethod
    def _combining_enter_camp_actions(action):
        """Generalizes ENTERCAMPAIGNSTAB, RETURNCAMPAIGNOVERVIEW and other similar actions..

        Args:
            action: Activity performed by user.

        Returns:
            a generalized action.

        """
        if action == 'ENTERCAMPAIGNSTAB':
            return "ENTERCAMPAIGNOVERVIEW"
        elif action == 'RETURNCAMPAIGNOVERVIEW':
            return "ENTERCAMPAIGNOVERVIEW"
        elif action == 'ENTERPRODUCT':
            return "ENTERCAMPAIGNOVERVIEW"
        else:
            return action

    @staticmethod
    def _assign_weight_to_actions(action, df_action_by_item):
        """Decide weights for specific action based on the number of actions performed dynamically.
        Most frequent action will have least weight.

        Args:
            action: Activity performed by user.
            df_action_by_item: A dataframe having Action and its total count

        Returns:
            Weight for an action

        """
        # total count is the sum of count of all actions
        total_cnt_of_activity = df_action_by_item.COUNT.sum()
        if action == 'PULL':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        elif action == 'SETUPASSETS':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        elif action == 'OPENASSETPREVIEW':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        elif action == 'ENTERCAMPAIGNOVERVIEW':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        elif action == 'CAMPAIGNSTATUS':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        elif action == 'ACTIVATECAMPAIGN':
            return total_cnt_of_activity / df_action_by_item[df_action_by_item['ACTIVITY_TYPE'] == action].iat[0, 1]
        else:
            return 0

    def _return_user_item_rating(self, data):
        """Generates implicit rating from 1 to 5 based on the action taken on an item by a user.

        Args:
            data: DataFrame having Activity data.

        Returns:
            A Dataframe having User, Item and the Rating provided by the user on the item.

        """
        data_with_wt = data.copy()
        # Count of different Actions overall
        df_action_by_item = data_with_wt[['ACTIVITY_TYPE', 'ITEM_ID']].groupby(['ACTIVITY_TYPE'],
                                                                               as_index=False).count()
        df_action_by_item.rename(columns={'ITEM_ID': 'COUNT'}, inplace=True)
        # Applying the function to assign weights to each user action
        data_with_wt['WEIGHT'] = data_with_wt.apply(
            lambda x: self._assign_weight_to_actions(x['ACTIVITY_TYPE'], df_action_by_item), axis=1)
        # Getting total weight of each Item by each User and the number of actions
        data_with_wt = data_with_wt[['USER_ID', 'ITEM_ID', 'WEIGHT']].groupby(['USER_ID', 'ITEM_ID'], as_index=False)[
            "WEIGHT"].agg({'count': "count", 'sum': sum}).reset_index()
        # Calculating the Weighted weight on each item by each user, to evaluate how many action a user took to
        # produce this much weight on an item
        data_with_wt['RATING'] = data_with_wt['sum'] / data_with_wt['count']

        # Lets normalize the data in the range of 1 to 5 using minmaxscaler
        mm_scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
        # Scaling the Weights from 1 to 5 and storing it as Rating
        data_with_wt['RATING'] = mm_scaler.fit_transform(data_with_wt[['RATING']])
        data_with_wt.drop(['count', 'sum', 'index'], axis=1, inplace=True)
        return data_with_wt

    @staticmethod
    def _return_top_n_product(data, threshold=10, topn=10):
        """Function to return n-top rated products for portal

        Args:
            data: DataFrame having Activity data.
            threshold: minimum number of ratings a product should have got. Default - 10
            topn: maximum number of products to be considered as top rated. Default - 10

        Returns:
            A Dataframe with top rated item and its rating

        """
        top_n_prod = data.copy()
        top_n_prod = top_n_prod[['USER_ID', 'ITEM_ID', 'RATING']].groupby(['ITEM_ID']).agg(
            {"RATING": ["count", "mean"]})
        top_n_prod.columns = top_n_prod.columns.map('_'.join)
        top_n_prod = top_n_prod.reset_index()
        top_n_prod = top_n_prod[top_n_prod['RATING_count'] >= threshold]
        top_n_prod.drop(['RATING_count'], axis=1, inplace=True)
        top_n_prod.rename(columns={'RATING_mean': 'RATING'}, inplace=True)
        return top_n_prod.sort_values('RATING', ascending=False).head(topn)

    # getting user's item rating for each user per item per portal
    def _get_user_item_rating_per_portal(self, df):
        """For each portal, gets the rating on item provided by user

        Args:
            df: DataFrame having Activity data.

        Returns:
            A dictionary having portalname as keys and dataframe having user, item and rating as value.

        """
        user_item_portal_rating = dict()
        portals = list(df['PORTALNAME'].unique())
        # portals.remove('INGRAM CLOUDBLUE DEMO HUB')
        for portal in portals:
            portal_data = df[df['PORTALNAME'] == portal]  # extracting data for portal
            user_item_portal_rating[portal] = self._return_user_item_rating(portal_data)
        return user_item_portal_rating

    # Get popular items per Portal, here threshold is dynamically evaluated based on the number of users in each portal
    def _get_top_trending_item_per_portal(self, user_item_portal_rating, no_of_recommendation):
        """Gets the top trending items in each portal based on ratings.

        Note:
            `threshold` is minimum number of rating an item should get in order to be considered in trending list.
            Value is assigned based on the number of users.

        Args:
            user_item_portal_rating: A dictionary having portalname as keys and dataframe having user, item and rating as value.
            no_of_recommendation: max number of recommended items in each portal
        Returns:
            A Dataframe having Portal Name, Item and Rating

        """
        top_trending_prod_for_portals = pd.DataFrame(columns=['ITEM_ID', 'RATING', 'PORTAL'])
        for portal in user_item_portal_rating.keys():
            uniqueusers = len(user_item_portal_rating[portal].USER_ID.unique())
            if uniqueusers > 1000:
                threshold = 20
            elif 500 < uniqueusers <= 1000:
                threshold = 10
            else:
                threshold = 5
            top_prod_for_portal = self._return_top_n_product(user_item_portal_rating[portal], threshold,
                                                             no_of_recommendation)
            top_prod_for_portal['PORTAL'] = portal
            top_trending_prod_for_portals = top_trending_prod_for_portals.append(top_prod_for_portal, ignore_index=True)
        return top_trending_prod_for_portals

    def _exporting_trending_item_data(self, top_trending_prod_for_portals):
        """Exports the Trending product dataframe to CSV.

        Args:
            top_trending_prod_for_portals: Dataframe having Top Trending items for each portal.

        """
        self._logger.info("Exporting Top Trending recommendation to CSV file TOP_TRENDING_PROD_FOR_PORTALS")
        top_trending_prod_for_portals.to_csv(self._OUT_FILE_RELATIVE_PATH + 'TOP_TRENDING_PROD_FOR_PORTALS.CSV',
                                             index=False)
        self._logger.info("Finished Top Trending recommendation!")

    def _run_colab_filter_model(self, user_item_portal_rating):
        """Based on the user-item rating Dataframe, runs the user-user colab model and predicts the ratings for un-rated items.

        Args:
            user_item_portal_rating: A dictionary having portalname as keys and dataframe having user, item and rating as value.

        Returns:
            A Dataframe top_n_neighbours having userid and userid's of neighbours.
            A dictionary having keys as portalname and predicted rating on items as values.

        """
        recommended_items = dict()

        # This will have neighbours for each user, we will export this data for analysis.
        top_n_neighbours = pd.DataFrame(columns=['USER_ID', 'NEIGHBOURS', 'PORTAL'])

        for portal in user_item_portal_rating.keys():
            # this dataframe has all the users and items available for the portal from Training dataset. Ratings is
            # available on the index where a user has taken action on item, else it will be NaN. We will predict the
            # rating for all NaN.
            df_p = user_item_portal_rating[portal]

            # This pivoted table have user_id in rows as index and items as columns. This will be a sparse matrix
            # with ratings whereever user has already rated the item.
            df_p = pd.pivot_table(df_p, values='RATING', index='USER_ID', columns='ITEM_ID')

            # initializing Reader class from surprise with a rating scale of 1 to 5
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(user_item_portal_rating[portal], reader)

            # Preparing the Trainset from the entire data
            trainset = data.build_full_trainset()

            # looking for 20 neighbours for evaluation, min neighbour is 4 (neighbours might not be positively related)
            algo = KNNWithMeans(k=20, min_k=4, sim_options={'name': 'pearson_baseline', 'user_based': True})
            algo.fit(trainset)

            # Filling up the pivoted matrix with recommendation
            top_n_neighbours, recommended_items[portal] = self._fill_the_pivoted_matrix_with_prediction(df_p, algo,
                                                                                                        portal,
                                                                                                        top_n_neighbours)

        return recommended_items, top_n_neighbours

    def _fill_the_pivoted_matrix_with_prediction(self, df_p, algo, portal, top_n_neighbours):
        """Takes the pivoted matrix of items and users and based on the prediction, fills the matrix

        Args:
            df_p: a pivoted matrix of items and users rating.
            algo: user-user colab model trained on user-item rating data
            portal: portal name
            top_n_neighbours: empty dataframe which will have userid and its neighbours.

        Returns:
            A Dataframe top_n_neighbours having userid and userid's of neighbours.
            A pivoted dataframe having users as index and items in columns and predicted rating.

        """
        for i in df_p.index:  # for each user
            raw_neighbours = []
            neighbours_filled = False

            for j in df_p.columns:  # for each item
                if df_p[j].loc[i] > 0:  # do nothing for already rated items
                    donothing = 1
                else:
                    pred = algo.predict(i, j,
                                        verbose=False)  # predict the rating for the user-item combination
                    if not pred[4]["was_impossible"]:  # was_impossible=False means model was able to find neighbours
                        if pred[4]["actual_k"] >= 2:  # assign only if atleast 2 positive neighbours found
                            df_p.at[i, j] = pred[3]  # fill the sparse matrix with predicated rating

                            # below part is for exporting neighbours for each user
                            if not neighbours_filled:
                                raw_neighbours, top_n_neighbours, neighbours_filled = self._get_neighbours_of_user(algo,
                                                                                                                   raw_neighbours,
                                                                                                                   top_n_neighbours,
                                                                                                                   portal,
                                                                                                                   i,
                                                                                                                   pred)
        return top_n_neighbours, df_p

    @staticmethod
    def _get_neighbours_of_user(algo, raw_neighbours, top_n_neighbours, portal, i, pred):
        """Finds the top neighbours of a user from model

        Args:
            algo: user-user colab model
            raw_neighbours: A list which will take userids which are neighbours of a particular user.
            top_n_neighbours: A dataframe which has userid and its neighbours. A new row will get appended to it.
            portal: portal name
            i: userid
            pred: prediction on the item by a user.

        Returns:
            A list having all the top neighbours of a user.
            A Dataframe having userid and list of all the top neighbours of a user.
            neighbours_filled: True if neighbour for a user is filled.

        """
        neighbors = algo.get_neighbors(algo.trainset.to_inner_uid(i), k=pred[4]["actual_k"])
        for u in neighbors:
            raw_neighbours.append(algo.trainset.to_raw_uid(u))
        neighbours_filled = True
        new_row = {'USER_ID': i,
                   'NEIGHBOURS': raw_neighbours,
                   'PORTAL': portal}
        top_n_neighbours = top_n_neighbours.append(new_row, ignore_index=True)
        return raw_neighbours, top_n_neighbours, neighbours_filled

    def _exporting_top_n_neighbours_data(self, top_n_neighbours):
        """Exports the Dataframe of neighbours to CSV

        Args:
            top_n_neighbours: A dataframe which has userid and its neighbours. A new row will get appended to it.

        """
        top_n_neighbours.reset_index(inplace=True, drop=True)
        self._logger.info("Exporting Exporting Top Neighbours for a user to CSV file top_n_neighbours")
        top_n_neighbours.to_csv(self._OUT_FILE_RELATIVE_PATH + 'top_n_neighbours.CSV', index=False)

    @staticmethod
    def _get_top_n_colabfilter_rec_items(recommended_items, no_of_recommendation):
        """From the dictionary of portal and predicted ratings by users, generates a dataframe having top n recommendation for each user.

        Args:
            recommended_items: A dictionary having keys as portal name and values as Dataframe of predicted ratings.
            no_of_recommendation: Number of maximum recommended product for each user.

        Returns:
            A Dataframe having userid, portalname and recommended item. For each user there will be maximum N rows.

        """
        top_col_filter_rec_prod_for_portals_users = pd.DataFrame(columns=['USER_ID', 'ITEM_ID', 'RATING', 'PORTAL'])
        for portal in recommended_items.keys():
            top_rec_prod_for_portals_user = recommended_items[portal].stack().reset_index(name='RATING')
            top_rec_prod_for_portals_user['PORTAL'] = portal
            for user in top_rec_prod_for_portals_user.USER_ID.unique().tolist():
                topitem = top_rec_prod_for_portals_user[top_rec_prod_for_portals_user['USER_ID'] == user].sort_values(
                    by=['RATING'], ascending=False)[:no_of_recommendation]
                top_col_filter_rec_prod_for_portals_users = top_col_filter_rec_prod_for_portals_users.append(topitem,
                                                                                                             ignore_index=True)
        return top_col_filter_rec_prod_for_portals_users

    def _exporting_top_colfilter_rec_prod_for_portals_users_data(self, top_col_filter_rec_prod_for_portals_users):
        """Exports the Dataframe having userid, portalname and recommended items to CSV.

        Args:
            top_col_filter_rec_prod_for_portals_users: A Dataframe having userid, portalname and recommended item.

        """
        self._logger.info(
            "Exporting Collaborative Filtering recommendation to CSV file TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS")
        top_col_filter_rec_prod_for_portals_users.to_csv(
            self._OUT_FILE_RELATIVE_PATH + 'TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS.CSV', index=False)
        self._logger.info("Finished Collaborative Filtering recommendation!")

    @staticmethod
    def _get_top_n_items_from_past(user_item_portal_rating, no_of_recommendation):
        """From already rated items by user, recommends the most used/rated item for a user.

        Args:
            user_item_portal_rating: A dictionary having keys as portal name and values as Dataframe of existing ratings.
            no_of_recommendation: Number of maximum recommended product for each user.

        Returns:
            A Dataframe having userid, portalname and recommended item. For each user there will be maximum N rows.

        """
        top_own_past_rated_prod_for_portals_users = pd.DataFrame(columns=['USER_ID', 'ITEM_ID', 'RATING', 'PORTAL'])
        for portal in user_item_portal_rating.keys():
            top_past_prod_for_portals_user = user_item_portal_rating[portal].copy()
            top_past_prod_for_portals_user['PORTAL'] = portal
            for user in top_past_prod_for_portals_user.USER_ID.unique().tolist():
                topitem = top_past_prod_for_portals_user[top_past_prod_for_portals_user['USER_ID'] == user].sort_values(
                    by=['RATING'], ascending=False)[:no_of_recommendation]
                top_own_past_rated_prod_for_portals_users = top_own_past_rated_prod_for_portals_users.append(topitem,
                                                                                                             ignore_index=True)
        return top_own_past_rated_prod_for_portals_users

    def _exporting_top_n_items_from_past(self, top_own_past_rated_prod_for_portals_users):
        """Exports the Dataframe having userid, portalname and recommended items to CSV.

        Args:
            top_own_past_rated_prod_for_portals_users: A Dataframe having userid, portalname and recommended item.

        """
        self._logger.info(
            "Exporting Top past rated recommendation to CSV file TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS")
        top_own_past_rated_prod_for_portals_users.to_csv(
            self._OUT_FILE_RELATIVE_PATH + 'TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV', index=False)
        self._logger.info("Finished Top past rated recommendation!")

    def _get_top_n_items_from_hybrid(self, user_item_portal_rating, top_own_past_rated_prod_for_portals_users,
                                     top_col_filter_rec_prod_for_portals_users, top_trending_prod_for_portals,
                                     no_of_recommendation, threshold_from_past):
        """Recommend a mix of all three type of recommendation for each user. Hybrid = Trending items + Past used items + ColabFilter recommendation

        Args:
            user_item_portal_rating: A dictionary having keys as portal name and values as Dataframe of existing ratings.
            top_own_past_rated_prod_for_portals_users: Dataframe having past rated items.
            top_col_filter_rec_prod_for_portals_users: Dataframe having user user col recommended items.
            top_trending_prod_for_portals: Dataframe having trending item list for each portal
            no_of_recommendation: Number of maximum recommended product for each user.
            threshold_from_past: Maximum number of items to be considered from past in hybrid.

        Returns:
            A Dataframe having userid, portalname and recommended item. For each user there will be maximum N rows.

        """
        top_hybrid_rec_for_portals_users = pd.DataFrame(
            columns=['USER_ID', 'ITEM_ID', 'SORT_ORDER', 'PORTAL', 'MODEL_TYPE'])
        for portal in user_item_portal_rating.keys():
            # Items from Past for portal
            pastitem = top_own_past_rated_prod_for_portals_users[
                top_own_past_rated_prod_for_portals_users['PORTAL'] == portal]
            # Items from ColabFilter for portal
            recitem = top_col_filter_rec_prod_for_portals_users[
                top_col_filter_rec_prod_for_portals_users['PORTAL'] == portal]
            # Trending or Popular items within the Portal
            trenditem = top_trending_prod_for_portals[top_trending_prod_for_portals['PORTAL'] == portal].sort_values(
                by=['RATING'],
                ascending=False)[
                'ITEM_ID'].tolist()
            for user in pastitem.USER_ID.unique().tolist():
                # pass the above dataframes and grab Hybrid combination for each user
                top_hybrid_rec_for_portals_users = self._get_hybridrec_for_each_user_from_combination(portal, user,
                                                                                                      threshold_from_past,
                                                                                                      no_of_recommendation,
                                                                                                      pastitem, recitem,
                                                                                                      trenditem,
                                                                                                      top_hybrid_rec_for_portals_users)
        return top_hybrid_rec_for_portals_users

    def _get_hybridrec_for_each_user_from_combination(self, portal, user, threshold_from_past, no_of_recommendation,
                                                      pastitem, recitem, trenditem, top_hybrid_rec_for_portals_users):
        """From the Portal wise data, grab Hybrid recommendation for each user

        Args:
            portal: PortalName
            user: UserId
            threshold_from_past: Maximum number of items to be considered from past in hybrid.
            no_of_recommendation: Number of maximum recommended product for each user.
            pastitem: Dataframe having past rated items for a portal.
            recitem: Dataframe having user user col recommended items for a portal.
            trenditem: Dataframe having trending item list for a portal.
            top_hybrid_rec_for_portals_users: Dataframe having the Hybrid recommended items for users.

        Returns:
            A Dataframe having hybrid recommended items for a portal concated with existing records.

        """
        # getting individual user data
        trend_item = trenditem.copy()
        past = pastitem[pastitem['USER_ID'] == user].sort_values(by=['RATING'], ascending=False)[:threshold_from_past][
            'ITEM_ID'].tolist()
        rec = \
            recitem[(recitem['USER_ID'] == user) & (recitem['RATING'] > 1)].sort_values(by=['RATING'], ascending=False)[
                'ITEM_ID'].tolist()

        # Ensuring the that the final recommendation have items in order of past, colabfilter, trending
        past_pop_items, trend_pop_items, colrec_pop_items = self._grab_items_from_diff_recs(past, rec, trend_item,
                                                                                            no_of_recommendation)

        tempdf = self._blend_items_from_diff_recs(past_pop_items, trend_pop_items, colrec_pop_items, user, portal)
        top_hybrid_rec_for_portals_users = pd.concat([top_hybrid_rec_for_portals_users, tempdf])
        return top_hybrid_rec_for_portals_users

    @staticmethod
    def _grab_items_from_diff_recs(past, rec, trend_item, no_of_recommendation):
        """Generates three list of each type of recommendation for a user

        Args:
            past: List having past rated items for a user.
            rec: List having user user col recommended items for a user.
            trend_item: List having trending item list for a portal.
            no_of_recommendation: Number of maximum recommended product for each user.

        Returns:
            Three different list each containing top n recommended items.

        """
        colrec_pop_items = []
        trend_pop_items = []
        # ensuring the same item should not exist in more than one category
        if len(past) > 0:
            for element in past:
                if element in rec:
                    rec.remove(element)
                if element in trend_item:
                    trend_item.remove(element)
        if len(rec) > 0:
            for element in rec:
                if element in trend_item:
                    trend_item.remove(element)
        past_pop_items = past.copy()

        if len(past_pop_items) < no_of_recommendation:  # grab item from colab filter only if items from past is less than the required no of recommendation
            colrec_pop_items = rec[:no_of_recommendation - len(past_pop_items)]
            if (len(past_pop_items) + len(
                    colrec_pop_items)) < no_of_recommendation:  # grab item from trending only if items from past+colabfilter is less than the required no of recommendation
                trend_pop_items = trend_item[:no_of_recommendation - (len(past_pop_items) + len(colrec_pop_items))]

        return past_pop_items, trend_pop_items, colrec_pop_items

    # prepare a dataframe with list of items and its source i.e. whether it is from past, colabfilter or trending
    @staticmethod
    def _blend_items_from_diff_recs(past_pop_items, trend_pop_items, colrec_pop_items, user, portal):
        """Combines the three different list each containing top n recommended items.
        Finally creates a dataframe which has N rows, one for each recommended item, userid and portal

        Args:
            past_pop_items: List having top past rated items for a user.
            trend_pop_items: List having top trending item list for a portal.
            colrec_pop_items: List having top user user col recommended items for a user.
            user: userid
            portal: Portal name.

        Returns:
            Returns a dataframe which has N rows, one for each recommended item, userid and portal

        """
        tempdf = pd.DataFrame(columns=['USER_ID', 'ITEM_ID', 'PORTAL', 'MODEL_TYPE'])
        if len(past_pop_items) > 0:
            new_row = {'USER_ID': user,
                       'ITEM_ID': past_pop_items,
                       'PORTAL': portal,
                       'MODEL_TYPE': 'PAST'}
            tempdf = tempdf.append(new_row, ignore_index=True)
        if len(colrec_pop_items) > 0:
            new_row = {'USER_ID': user,
                       'ITEM_ID': colrec_pop_items,
                       'PORTAL': portal,
                       'MODEL_TYPE': 'COLREC'}
            tempdf = tempdf.append(new_row, ignore_index=True)
        if len(trend_pop_items) > 0:
            new_row = {'USER_ID': user,
                       'ITEM_ID': trend_pop_items,
                       'PORTAL': portal,
                       'MODEL_TYPE': 'TREND'}
            tempdf = tempdf.append(new_row, ignore_index=True)
        tempdf = tempdf.explode('ITEM_ID')
        tempdf.reset_index(inplace=True, drop=True)
        tempdf['SORT_ORDER'] = tempdf.index + 1
        return tempdf

    def _exporting_top_n_items_from_hybrid(self, top_hybrid_rec_for_portals_users):
        """Exports the Dataframe having userid, portalname and recommended items to CSV.

        Args:
            top_hybrid_rec_for_portals_users: A Dataframe having userid, portalname and recommended item.

        """
        self._logger.info("Exporting Hybrid recommendation to CSV file TOP_HYBRID_REC_FOR_PORTALS_USERS")
        top_hybrid_rec_for_portals_users.to_csv(self._OUT_FILE_RELATIVE_PATH + 'TOP_HYBRID_REC_FOR_PORTALS_USERS.CSV',
                                                index=False)
        self._logger.info("Finished Hybrid recommendation!")

    def _pre_process_dataframe(self, df):
        """From the main dataframe, generalizes some actions, removes not required actions.

        Args:
            df: DataFrame having Activity data.

        Returns:
            Preprocessed dataframe

        """
        # Generalizing the activation_download_actions action.
        df['ACTIVITY_TYPE'] = df.apply(lambda x: self._combining_activation_download_actions(x['ACTIVITY_TYPE']),
                                       axis=1)
        # Generalizing the pdf download action.
        df['ACTIVITY_TYPE'] = df.apply(lambda x: self._generalizing_different_downloads(x['ACTIVITY_TYPE']), axis=1)
        # removing unwanted actions
        df = self._removing_unwanted_actions(df)
        # Combining Download, Pull, Video.
        df['ACTIVITY_TYPE'] = df.apply(lambda x: self._combining_pull_actions(x['ACTIVITY_TYPE']), axis=1)
        # Combining ENTERCAMPAIGNSTAB, RETURNCAMPAIGNOVERVIEW, ENTERPRODUCT.
        df['ACTIVITY_TYPE'] = df.apply(lambda x: self._combining_enter_camp_actions(x['ACTIVITY_TYPE']), axis=1)
        return df

    def rec_sys_for_hybrid_model(self):
        """Main method to generate Hybrid recommendation based on other three recommendation from activity data

        """

        try:
            logger = self._logger
            #######################Getting Top Rated products###########################
            logger.info("Starting Top Trending recommendation")
            # Reading data from excel
            df = self._reading_data()
            if df.shape[0] == 0:
                raise Exception('No Data found')
            else:
                # Data Preprocessing
                df = self._pre_process_dataframe(df)
                # getting user's item rating for each user per item per portal
                user_item_portal_rating = self._get_user_item_rating_per_portal(df)
                top_trending_prod_for_portals = self._get_top_trending_item_per_portal(user_item_portal_rating,
                                                                                       self._no_of_recommendation)
                if top_trending_prod_for_portals.shape[0] > 0:
                    self._exporting_trending_item_data(top_trending_prod_for_portals)

                #########################Collaborating Filter###########################
                # Calculate score for every item using user user collaborating filtering
                logger.info("Starting Collaborative Filter recommendation")
                recommended_items, top_n_neighbours = self._run_colab_filter_model(user_item_portal_rating)
                if top_n_neighbours.shape[0] > 0:
                    self._exporting_top_n_neighbours_data(top_n_neighbours)
                top_col_filter_rec_prod_for_portals_users = self._get_top_n_colabfilter_rec_items(recommended_items,
                                                                                                  self._no_of_recommendation)
                if top_col_filter_rec_prod_for_portals_users.shape[0] > 0:
                    self._exporting_top_colfilter_rec_prod_for_portals_users_data(
                        top_col_filter_rec_prod_for_portals_users)

                ############################Users own Top Rated Prod from Past###############################
                logger.info("Starting Top past rated recommendation")
                top_own_past_rated_prod_for_portals_users = self._get_top_n_items_from_past(user_item_portal_rating,
                                                                                            self._no_of_recommendation)
                if top_own_past_rated_prod_for_portals_users.shape[0] > 0:
                    self._exporting_top_n_items_from_past(top_own_past_rated_prod_for_portals_users)

                ###############Hybrid from PastRated+CollabFilter+TopTrending###############
                logger.info("Starting Hybrid recommendation")
                top_hybrid_rec_for_portals_users = self._get_top_n_items_from_hybrid(user_item_portal_rating,
                                                                                     top_own_past_rated_prod_for_portals_users,
                                                                                     top_col_filter_rec_prod_for_portals_users,
                                                                                     top_trending_prod_for_portals,
                                                                                     self._no_of_recommendation,
                                                                                     self._threshold_from_past)
                if top_hybrid_rec_for_portals_users.shape[0] > 0:
                    self._exporting_top_n_items_from_hybrid(top_hybrid_rec_for_portals_users)

                ###################################################################################
        except Exception:
            logger.error("Exception occurred", exc_info=True)
            raise
        else:
            logger.info("Prediction done successfully!!!")
