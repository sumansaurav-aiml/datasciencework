from src.rec_sys_hybrid import RecSysHybrid
import pytest
import os


@pytest.fixture(scope='module')
def folder_path():
    # all test csv files are under tests folder
    folder_path = 'tests'
    # Clearing the prediction files and logs before starting test
    rootdir = './'+folder_path+'/data'
    for subdir, dirs, files in os.walk(rootdir):
        for dirname in dirs:
            if dirname.startswith("jobid"):
                if dirname != "jobid7" and dirname != "jobid8" and dirname != "jobid9" and dirname != "jobid10":
                    for _subdir, _dirs, _files in os.walk(os.path.join(rootdir, dirname)):
                        for filename in _files:
                            if filename.endswith(".log"):
                                open(os.path.join(rootdir, dirname, filename), 'w').close()
                        for _dirname in _dirs:
                            if _dirname.startswith("predictionfiles"):
                                for __subdir, __dirs, __files in os.walk(os.path.join(rootdir, dirname, _dirname)):
                                    for filename in __files:
                                        os.remove(os.path.join(rootdir, dirname, _dirname, filename))
    yield folder_path


@pytest.fixture(scope='module')
def job_1():
    jobid = 1
    yield jobid


@pytest.fixture(scope='module')
def job_2():
    jobid = 2
    yield jobid


@pytest.fixture(scope='module')
def job_3():
    jobid = 3
    yield jobid


@pytest.fixture(scope='module')
def job_4():
    jobid = 4
    yield jobid


@pytest.fixture(scope='module')
def job_5():
    jobid = 5
    yield jobid


def test_rec_sys_for_hybrid_model_no_file(job_1, folder_path):
    """
        Testing with empty folder,
        this should raise an exception of No file or Directory
    """
    total_files = 0
    with pytest.raises(Exception) as e:
        rec = RecSysHybrid(job_1, 20, 3, folder_path)
        rec.rec_sys_for_hybrid_model()
    assert "No such file or directory" in str(e.value)
    for base, dirs, files in os.walk('./'+folder_path+'/data/jobid'+str(job_1)+'/predictionfiles/'):
        for Files in files:
            total_files += 1
    if total_files > 0:
        assert False
    else:
        assert True


def test_rec_sys_for_hybrid_model_empty_file(job_2, folder_path):
    """
        testing with CSV without any header,
        this should raise exception of No Columns to parse
    """
    total_files = 0
    with pytest.raises(Exception) as e:
        rec = RecSysHybrid(job_2, 20, 3, folder_path)
        rec.rec_sys_for_hybrid_model()
    assert "No columns to parse from file" in str(e.value)
    for base, dirs, files in os.walk('./'+folder_path+'/data/jobid'+str(job_2)+'/predictionfiles/'):
        for _ in files:
            total_files += 1
    if total_files > 0:
        assert False
    else:
        assert True


def test_rec_sys_for_hybrid_model_only_header(job_3, folder_path):
    """
        test with a CSV with only header but no data,
        this will raise exception of No Data Found
    """
    total_files = 0
    with pytest.raises(Exception) as e:
        rec = RecSysHybrid(job_3, 20, 3, folder_path)
        rec.rec_sys_for_hybrid_model()
    assert "No Data found" in str(e.value)
    for base, dirs, files in os.walk('./'+folder_path+'/data/jobid'+str(job_3)+'/predictionfiles/'):
        for Files in files:
            total_files += 1
    if total_files > 0:
        assert False
    else:
        assert True


def test_rec_sys_for_hybrid_model_only_one_row(job_4, folder_path):
    """
        test with a CSV having only one row.
        model should find only recommendation and should generate 3 csv files
        it should not generate neighbour file and trending item file as
        for only 1 user, there can't be a neighbour, and to consider an item trending,
        there is a threshold of minimum users activity on item.
    """
    rec = RecSysHybrid(job_4, 20, 3, folder_path)
    rec.rec_sys_for_hybrid_model()

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_4)+'/predictionfiles/TOP_TRENDING_PROD_FOR_PORTALS.CSV'):
        assert False
    else:
        assert True

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_4)+'/predictionfiles/top_n_neighbours.CSV'):
        assert False
    else:
        assert True


def test_rec_sys_for_hybrid_model_full_data(job_5, folder_path):
    """
        Test with a valid sample csv, all prediction file should generate without exception
    """
    rec = RecSysHybrid(job_5, 20, 3, folder_path)
    rec.rec_sys_for_hybrid_model()

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_5)+'/predictionfiles/TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS.CSV'):
        assert True
    else:
        assert False

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_5)+'/predictionfiles/TOP_HYBRID_REC_FOR_PORTALS_USERS.CSV'):
        assert True
    else:
        assert False

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_5)+'/predictionfiles/TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV'):
        assert True
    else:
        assert False

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_5)+'/predictionfiles/TOP_TRENDING_PROD_FOR_PORTALS.CSV'):
        assert True
    else:
        assert False

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_5)+'/predictionfiles/top_n_neighbours.CSV'):
        assert True
    else:
        assert False


'''
@pytest.fixture(scope='module')
def instance():
    rec = RecSysHybrid(1, 20, 3)
    yield rec
    trend_file_path = './src/data/jobid1/predictionfiles/TOP_TRENDING_PROD_FOR_PORTALS.CSV'
    top_n_path = './src/data/jobid1/predictionfiles/top_n_neighbours.CSV'
    top_colab_path = './src/data/jobid1/predictionfiles/TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS.CSV'
    top_past_path = './src/data/jobid1/predictionfiles/TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV'
    hybrid_rec_path = './src/data/jobid1/predictionfiles/TOP_HYBRID_REC_FOR_PORTALS_USERS.CSV'
    if os.path.isfile(trend_file_path):
        os.remove(trend_file_path)
    if os.path.isfile(top_n_path):
        os.remove(top_n_path)
    if os.path.isfile(top_colab_path):
        os.remove(top_colab_path)
    if os.path.isfile(top_past_path):
        os.remove(top_past_path)
    if os.path.isfile(hybrid_rec_path):
        os.remove(hybrid_rec_path)


@pytest.fixture(scope='module')
def df(instance):
    df = instance._reading_data()
    yield df


def test__reading_data(instance):
    df = instance._reading_data()
    assert df.shape[0] == 4800
    assert df.shape[1] == 9


def test__combining_activation_download_actions(instance):
    assert instance._combining_activation_download_actions('ACTIVATECAMPAIGN') == 'ACTIVATECAMPAIGN'
    assert instance._combining_activation_download_actions('DOWNLOADCAMPAIGN') == 'ACTIVATECAMPAIGN'
    assert instance._combining_activation_download_actions('TEST') == 'TEST'


def test__generalizing_different_downloads(instance):
    assert instance._generalizing_different_downloads('DOWNLOADPDF') == 'DOWNLOAD'
    assert instance._generalizing_different_downloads('DOWN') == 'DOWN'


def test__removing_unwanted_actions(instance, df):
    df = instance._removing_unwanted_actions(df)
    assert df.loc[df['ACTIVITY_TYPE'] == 'ENTERCAMPAIGNSTARTMARKETING'].shape[0] == 0
    assert df.loc[df['ACTIVITY_TYPE'] == 'CLOSEASSETPREVIEW'].shape[0] == 0
    assert df.loc[df['ACTIVITY_TYPE'] == 'CLOSESETUPASSETS'].shape[0] == 0


def test__combining_pull_actions(instance):
    assert instance._combining_pull_actions('VIDEO_GETDEFAULTEMBEDCODE') == 'PULL'
    assert instance._combining_pull_actions('DOWNLOAD') == 'PULL'
    assert instance._combining_pull_actions('TEST') == 'TEST'


def test__combining_enter_camp_actions(instance):
    assert instance._combining_enter_camp_actions('ENTERCAMPAIGNSTAB') == 'ENTERCAMPAIGNOVERVIEW'
    assert instance._combining_enter_camp_actions('RETURNCAMPAIGNOVERVIEW') == 'ENTERCAMPAIGNOVERVIEW'
    assert instance._combining_enter_camp_actions('ENTERPRODUCT') == 'ENTERCAMPAIGNOVERVIEW'
    assert instance._combining_enter_camp_actions('TEST') == 'TEST'


def test__assign_weight_to_actions(instance, df):
    df_action_by_item = df[['ACTIVITY_TYPE', 'ITEM_ID']].groupby(['ACTIVITY_TYPE'], as_index=False).count()
    df_action_by_item.rename(columns={'ITEM_ID': 'COUNT'}, inplace=True)
    assert instance._assign_weight_to_actions('PULL', df_action_by_item) == 11.136890951276103
    assert instance._assign_weight_to_actions('SETUPASSETS', df_action_by_item) == 58.53658536585366
    assert instance._assign_weight_to_actions('OPENASSETPREVIEW', df_action_by_item) == 4.359673024523161
    assert instance._assign_weight_to_actions('ENTERCAMPAIGNOVERVIEW', df_action_by_item) == 3.061224489795918
    assert instance._assign_weight_to_actions('CAMPAIGNSTATUS', df_action_by_item) == 800.0
    assert instance._assign_weight_to_actions('DOWNLOADPDF_2125611', df_action_by_item) == 0


def test__return_user_item_rating(instance, df):
    rating_df = instance._return_user_item_rating(df)
    assert rating_df.shape[0] == 836
    assert rating_df[(rating_df['USER_ID'] == 125827) & (rating_df['ITEM_ID'] == 2949576)].iat[
               0, 2] == 1.0479607593786902
    assert rating_df[(rating_df['USER_ID'] == 125037) & (rating_df['ITEM_ID'] == 2949577)].iat[
               0, 2] == 1.081402557140805


def test__get_user_item_rating_per_portal(instance, df):
    user_item_rating = instance._get_user_item_rating_per_portal(df)
    assert len(user_item_rating.keys()) == 25
    assert user_item_rating['INGRAM US HUB'].shape[0] == 352
    assert user_item_rating['INGRAM UK HUB'].shape[0] == 10
    assert user_item_rating['INGRAM UK HUB'][(user_item_rating['INGRAM UK HUB']['USER_ID'] == 124691) & (
            user_item_rating['INGRAM UK HUB']['ITEM_ID'] == 2950032)].iat[0, 2] == 3.261484098939929
    assert user_item_rating['INGRAM US HUB'][(user_item_rating['INGRAM US HUB']['USER_ID'] == 119513) & (
            user_item_rating['INGRAM US HUB']['ITEM_ID'] == 2945206)].iat[0, 2] == 1.0425958805511069


def test__return_top_n_product(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    top_prod_for_portal = instance._return_top_n_product(user_item_portal_rating['INGRAM US HUB'], 5, 10)
    assert top_prod_for_portal.shape[0] == 10
    assert top_prod_for_portal.iat[0, 1] == 1.4224249649184757

    top_prod_for_portal = instance._return_top_n_product(user_item_portal_rating['INGRAM UK HUB'], 2, 10)
    assert top_prod_for_portal.shape[0] == 2
    assert top_prod_for_portal.iat[0, 1] == 3.0


def test__get_top_trending_item_per_portal(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    top_trending_prod_for_portals = instance._get_top_trending_item_per_portal(user_item_portal_rating, 10)
    assert top_trending_prod_for_portals.shape[0] == 34
    assert top_trending_prod_for_portals.iat[0, 1] == 1.4224249649184757
    assert top_trending_prod_for_portals.iat[1, 1] == 1.1413641488820987


def test__exporting_trending_item_data(instance, df):
    if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_TRENDING_PROD_FOR_PORTALS.CSV'):
        assert False
    else:
        assert True
        top_trending_prod_for_portals = instance._get_top_trending_item_per_portal(
            instance._get_user_item_rating_per_portal(df), 10)
        instance._exporting_trending_item_data(top_trending_prod_for_portals)
        if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_TRENDING_PROD_FOR_PORTALS.CSV'):
            assert True
        else:
            assert False


def test__get_neighbours_of_user(instance, df):
    top_n_neighbours = pd.DataFrame(columns=['USER_ID', 'NEIGHBOURS', 'PORTAL'])
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_item_portal_rating['INGRAM US HUB'], reader)
    trainset = data.build_full_trainset()
    algo = KNNWithMeans(k=20, sim_options={'name': 'pearson_baseline', 'user_based': True})
    algo.fit(trainset)
    raw_neighbours = []
    neighbours_filled = False
    pred = algo.predict(119513, 2935602,
                        verbose=True)
    raw_neighbours, top_n_neighbours, neighbours_filled = instance._get_neighbours_of_user(algo,
                                                                                           raw_neighbours,
                                                                                           top_n_neighbours,
                                                                                           'INGRAM US HUB',
                                                                                           119513,
                                                                                           pred)
    assert top_n_neighbours.shape[0] == 1
    assert len(raw_neighbours) == 2
    assert raw_neighbours == [139140, 130798]
    if neighbours_filled:
        assert True
    else:
        assert False


def test__fill_the_pivoted_matrix_with_prediction(instance, df):
    recommended_items = dict()
    top_n_neighbours = pd.DataFrame(columns=['USER_ID', 'NEIGHBOURS', 'PORTAL'])
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    df_p = user_item_portal_rating['INGRAM US HUB']
    df_p = pd.pivot_table(df_p, values='RATING', index='USER_ID', columns='ITEM_ID')
    # initializing Reader class from surprise with a rating scale of 1 to 5
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_item_portal_rating['INGRAM US HUB'], reader)
    trainset = data.build_full_trainset()

    algo = KNNWithMeans(k=20, sim_options={'name': 'pearson_baseline', 'user_based': True})
    algo.fit(trainset)

    # Filling up the pivoted matrix with recommendation
    top_n_neighbours, recommended_items['INGRAM US HUB'] = instance._fill_the_pivoted_matrix_with_prediction(df_p, algo,
                                                                                                             'INGRAM US HUB',
                                                                                                             top_n_neighbours)

    assert recommended_items['INGRAM US HUB'].shape == (11, 111)
    assert recommended_items['INGRAM US HUB'].iat[0, 3] == 1.0425958805511069
    assert recommended_items['INGRAM US HUB'].iat[1, 4] == 1.0058249148575011
    assert top_n_neighbours.shape == (9, 3)
    assert top_n_neighbours.iat[1, 1] == [139140, 121455, 122662]
    assert top_n_neighbours.iat[3, 1] == [144957, 130798, 120174]


def test__run_colab_filter_model(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
    assert recommended_items['INGRAM US HUB'].shape == (11, 111)
    assert recommended_items['INGRAM US HUB'].iat[0, 4] == 1.0449473273507608
    assert recommended_items['INGRAM US HUB'].iat[2, 4] == 1.0471138136568534
    assert recommended_items['INGRAM US HUB'].index.tolist() == [119513, 119588, 119848, 119899, 120174, 121455, 122662,
                                                                 130798, 139140, 144957, 153926]
    assert top_n_neighbours.iat[1, 1] == [139140, 121455, 122662]
    assert top_n_neighbours.iat[6, 1] == [139140, 119513, 119588, 153926, 122662, 119899]


def test__exporting_top_n_neighbours_data(instance, df):
    if os.path.isfile('./src/data/jobid1/predictionfiles/top_n_neighbours.CSV'):
        assert False
    else:
        assert True
        user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
        recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
        instance._exporting_top_n_neighbours_data(top_n_neighbours)
        if os.path.isfile('./src/data/jobid1/predictionfiles/top_n_neighbours.CSV'):
            assert True
        else:
            assert False


def test__get_top_n_colabfilter_rec_items(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
    top_col_filter_rec_prod_for_portals_users = instance._get_top_n_colabfilter_rec_items(recommended_items, 20)
    assert top_col_filter_rec_prod_for_portals_users.shape == (704, 4)
    assert top_col_filter_rec_prod_for_portals_users.iat[99, 2] == 1.400643080204513
    assert top_col_filter_rec_prod_for_portals_users.iat[98, 2] == 3.9753011976968278
    assert top_col_filter_rec_prod_for_portals_users.iat[97, 2] == 1.0149824424502536
    assert top_col_filter_rec_prod_for_portals_users[top_col_filter_rec_prod_for_portals_users['USER_ID'] == 119899][
               'ITEM_ID'].tolist() == [2952475,
                                       2962270,
                                       2949579,
                                       2950038,
                                       2935602,
                                       2959325,
                                       2951540,
                                       2952459,
                                       2952472,
                                       2952476,
                                       2952482,
                                       2952484,
                                       2959331,
                                       2950061,
                                       2959332,
                                       2959354,
                                       2959366,
                                       2961589,
                                       2962267,
                                       2965811]


def test__exporting_top_colfilter_rec_prod_for_portals_users_data(instance, df):
    if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS.CSV'):
        assert False
    else:
        assert True
        user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
        recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
        top_col_filter_rec_prod_for_portals_users = instance._get_top_n_colabfilter_rec_items(recommended_items, 20)
        instance._exporting_top_colfilter_rec_prod_for_portals_users_data(top_col_filter_rec_prod_for_portals_users)
        if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_COL_FILTER_REC_PROD_FOR_PORTALS_USERS.CSV'):
            assert True
        else:
            assert False


def test__get_top_n_items_from_past(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    top_own_past_rated_prod_for_portals_users = instance._get_top_n_items_from_past(user_item_portal_rating, 20)
    assert top_own_past_rated_prod_for_portals_users.shape == (668, 4)
    assert top_own_past_rated_prod_for_portals_users.iat[14, 2] == 5.0
    assert top_own_past_rated_prod_for_portals_users.iat[15, 2] == 4.147346480679814
    assert top_own_past_rated_prod_for_portals_users.iat[16, 2] == 2.5132275132275135
    assert top_own_past_rated_prod_for_portals_users[top_own_past_rated_prod_for_portals_users['USER_ID'] == 118059][
               'ITEM_ID'].tolist() == [2949564,
                                       2949566,
                                       2949567,
                                       2950038,
                                       2950063,
                                       2949563,
                                       2949565,
                                       2949568,
                                       2949570]


def test__exporting_top_n_items_from_past(instance, df):
    if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV'):
        assert False
    else:
        assert True
        user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
        top_own_past_rated_prod_for_portals_users = instance._get_top_n_items_from_past(user_item_portal_rating, 20)
        instance._exporting_top_n_items_from_past(top_own_past_rated_prod_for_portals_users)
        if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV'):
            assert True
        else:
            assert False


def test__grab_items_from_diff_recs(instance):
    past = ['A', 'B', 'C', 'D', 'E']
    rec = ['C', 'D', 'E', 'F', 'G', 'H']
    trend_item = ['H', 'I', 'J']
    past_pop_items, trend_pop_items, colrec_pop_items = instance._grab_items_from_diff_recs(past, rec, trend_item, 20)
    assert past_pop_items == ['A', 'B', 'C', 'D', 'E']
    assert colrec_pop_items == ['F', 'G', 'H']
    assert trend_item == ['I', 'J']


def test__blend_items_from_diff_recs(instance):
    past = ['A', 'B', 'C', 'D', 'E']
    rec = ['C', 'D', 'E', 'F', 'G', 'H']
    trend_item = ['H', 'I', 'J']
    past_pop_items, trend_pop_items, colrec_pop_items = instance._grab_items_from_diff_recs(past, rec, trend_item, 20)
    tempdf = instance._blend_items_from_diff_recs(past_pop_items, trend_pop_items, colrec_pop_items, 111, 'US')
    assert tempdf.shape == (10, 5)
    assert tempdf.ITEM_ID.tolist() == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    assert tempdf.MODEL_TYPE.unique().tolist() == ['PAST', 'COLREC', 'TREND']


def test__get_hybridrec_for_each_user_from_combination(instance, df):
    top_hybrid_rec_for_portals_users = pd.DataFrame(
        columns=['USER_ID', 'ITEM_ID', 'SORT_ORDER', 'PORTAL', 'MODEL_TYPE'])
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    top_own_past_rated_prod_for_portals_users = instance._get_top_n_items_from_past(user_item_portal_rating, 20)
    recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
    top_col_filter_rec_prod_for_portals_users = instance._get_top_n_colabfilter_rec_items(recommended_items, 20)
    top_trending_prod_for_portals = instance._get_top_trending_item_per_portal(user_item_portal_rating, 20)
    pastitem = top_own_past_rated_prod_for_portals_users[
        top_own_past_rated_prod_for_portals_users['PORTAL'] == 'INGRAM US HUB']
    # Items from ColabFilter for portal
    recitem = top_col_filter_rec_prod_for_portals_users[
        top_col_filter_rec_prod_for_portals_users['PORTAL'] == 'INGRAM US HUB']
    # Trending or Popular items within the Portal
    trenditem = top_trending_prod_for_portals[top_trending_prod_for_portals['PORTAL'] == 'INGRAM US HUB'].sort_values(
        by=['RATING'],
        ascending=False)[
        'ITEM_ID'].tolist()
    top_hybrid_rec_for_portals_users = instance._get_hybridrec_for_each_user_from_combination('INGRAM US HUB', 153926,
                                                                                              3,
                                                                                              20,
                                                                                              pastitem, recitem,
                                                                                              trenditem,
                                                                                              top_hybrid_rec_for_portals_users)
    assert top_hybrid_rec_for_portals_users.shape == (20, 5)
    assert top_hybrid_rec_for_portals_users[top_hybrid_rec_for_portals_users['MODEL_TYPE'] == 'COLREC'].count()[0] == 17
    assert top_hybrid_rec_for_portals_users.ITEM_ID.unique().tolist() == [2962270,
                                                                          2952475,
                                                                          2959325,
                                                                          2952461,
                                                                          2952482,
                                                                          2952481,
                                                                          2952478,
                                                                          2950036,
                                                                          2952477,
                                                                          2952472,
                                                                          2952462,
                                                                          2952460,
                                                                          2952458,
                                                                          2951540,
                                                                          2950064,
                                                                          2950062,
                                                                          2950052,
                                                                          2952473,
                                                                          2935602,
                                                                          2952483]


def test__get_top_n_items_from_hybrid(instance, df):
    user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
    top_own_past_rated_prod_for_portals_users = instance._get_top_n_items_from_past(user_item_portal_rating, 20)
    recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
    top_col_filter_rec_prod_for_portals_users = instance._get_top_n_colabfilter_rec_items(recommended_items, 20)
    top_trending_prod_for_portals = instance._get_top_trending_item_per_portal(user_item_portal_rating, 20)
    top_hybrid_rec_for_portals_users = instance._get_top_n_items_from_hybrid(user_item_portal_rating,
                                                                             top_own_past_rated_prod_for_portals_users,
                                                                             top_col_filter_rec_prod_for_portals_users,
                                                                             top_trending_prod_for_portals,
                                                                             20,
                                                                             3)
    assert top_hybrid_rec_for_portals_users.shape == (840, 5)
    assert top_hybrid_rec_for_portals_users[top_hybrid_rec_for_portals_users['USER_ID'] == 119899].shape == (20, 5)
    assert top_hybrid_rec_for_portals_users[top_hybrid_rec_for_portals_users['USER_ID'] == 119899].iat[1, 1] == 2949565


def test__exporting_top_n_items_from_hybrid(instance, df):
    if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_HYBRID_REC_FOR_PORTALS_USERS.CSV'):
        assert False
    else:
        assert True
        user_item_portal_rating = instance._get_user_item_rating_per_portal(df)
        top_own_past_rated_prod_for_portals_users = instance._get_top_n_items_from_past(user_item_portal_rating, 20)
        recommended_items, top_n_neighbours = instance._run_colab_filter_model(user_item_portal_rating)
        top_col_filter_rec_prod_for_portals_users = instance._get_top_n_colabfilter_rec_items(recommended_items, 20)
        top_trending_prod_for_portals = instance._get_top_trending_item_per_portal(user_item_portal_rating, 20)
        top_hybrid_rec_for_portals_users = instance._get_top_n_items_from_hybrid(user_item_portal_rating,
                                                                                 top_own_past_rated_prod_for_portals_users,
                                                                                 top_col_filter_rec_prod_for_portals_users,
                                                                                 top_trending_prod_for_portals,
                                                                                 20,
                                                                                 3)
        instance._exporting_top_n_items_from_hybrid(top_hybrid_rec_for_portals_users)
        if os.path.isfile('./src/data/jobid1/predictionfiles/TOP_HYBRID_REC_FOR_PORTALS_USERS.CSV'):
            assert True
        else:
            assert False

'''
