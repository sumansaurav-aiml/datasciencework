from src.import_activity.import_activity_data import ImportActivityData
import pytest
import os
from unittest import mock
import pandas as pd


@pytest.fixture(scope='module')
def cookie_details():
    cookie_details = [
        (117955, 2949567, 'ENTERCAMPAIGNSTAB', '11-02-2019', 'INGRAM ME HUB', 497189, 0, 'MICROSOFT AZURE'),
        (118023, 2949568, 'ENTERCAMPAIGNSTAB', '11-02-2019', 'INGRAM US HUB', 497189, 0, 'MICROSOFT AZURE')
    ]
    yield cookie_details


@pytest.fixture(scope='module')
def folder_path():
    # all test csv files are under tests folder
    folder_path = 'tests'
    yield folder_path


@pytest.fixture(scope='module')
def job_6(folder_path):
    jobid = 6
    filepath = './'+folder_path+'/data/jobid'+str(jobid)+'/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'
    if os.path.isfile(filepath):
        os.remove(filepath)
    yield jobid


@mock.patch('src.import_activity.import_activity_data.create_engine')
def test_import_activity_data(mock_create_engine, cookie_details, job_6, folder_path):
    mock_create_engine.return_value.connect.return_value.execute.return_value.fetchall.return_value = cookie_details
    import_activity = ImportActivityData(job_6, 'tests')
    import_activity.import_activity_data('dummu_conn_string')

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_6)+'/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'):
        assert True
        df = pd.read_csv(
            './' + folder_path + '/data/jobid' + str(job_6) + '/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV')
        assert df.shape == (2, 8)
    else:
        assert False


@mock.patch('src.import_activity.import_activity_data.create_engine')
def test_import_activity_data_no_data(mock_create_engine, cookie_details, job_6, folder_path):
    filepath = './'+folder_path+'/data/jobid'+str(job_6)+'/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'
    if os.path.isfile(filepath):
        os.remove(filepath)

    mock_create_engine.return_value.connect.return_value.execute.return_value.fetchall.return_value = []
    import_activity = ImportActivityData(job_6, 'tests')
    import_activity.import_activity_data('dummu_conn_string')

    if os.path.isfile('./'+folder_path+'/data/jobid'+str(job_6)+'/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'):
        assert False
    else:
        assert True







