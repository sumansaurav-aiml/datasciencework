from src.export_activity.export_hybrid_prod import ExportHybridProd
import pytest
from unittest import mock


@pytest.fixture(scope='module')
def folder_path():
    # all test csv files are under tests folder
    folder_path = 'tests'
    yield folder_path


@pytest.fixture(scope='module')
def job_7():
    jobid = 7
    yield jobid


@pytest.fixture(scope='module')
def job_8():
    jobid = 8
    yield jobid


@pytest.fixture(scope='module')
def job_9():
    jobid = 9
    yield jobid


@mock.patch('src.export_activity.export_hybrid_prod.create_engine')
def test_export_hybrid_rec_no_file(dummy_db, job_7, folder_path):
    with pytest.raises(Exception) as e:
        export_activity = ExportHybridProd(job_7, 'tests')
        export_activity.export_hybrid_rec('dummu_conn_string')
    assert "No such file or directory" in str(e.value)


@mock.patch('src.export_activity.export_hybrid_prod.create_engine')
def test_export_hybrid_rec_only_header(dummy_db, job_8, folder_path):
    with pytest.raises(Exception) as e:
        export_activity = ExportHybridProd(job_8, 'tests')
        export_activity.export_hybrid_rec('dummu_conn_string')
    assert "Data not found" in str(e.value)


@mock.patch('src.export_activity.export_hybrid_prod.create_engine')
def test_export_hybrid_rec(dummy_db, job_9, folder_path):
    try:
        export_activity = ExportHybridProd(job_9, 'tests')
        export_activity.export_hybrid_rec('dummu_conn_string')
        assert True
    except Exception:
        assert False
