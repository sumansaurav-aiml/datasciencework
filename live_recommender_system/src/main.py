# -*- coding: utf-8 -*-
"""main.py
This module run the import process which will import activity data from database,
Generates the hybrid recommendation which is a combination of Trending, Past and user-user col filter model.
Finally exports the recommended item Dataframes to Database.
Todo:
    * Nothing
"""

# import the module
from sqlalchemy import create_engine
from src.rec_sys_hybrid import RecSysHybrid
from src.import_activity.import_activity_data import ImportActivityData
from src.export_activity.export_hybrid_prod import ExportHybridProd
from src.export_activity.export_past_prod import ExportPastProd
from src.export_activity.export_col_filter_prod import ExportColFilterProd
from src.export_activity.export_top_trending_prod import ExportTrendingProd
from src.logger import Logger
import os


class RecSys:
    """Imports the data, runs the model and exports the recommendation back to database.
    Attributes:
        no_of_recommendation (int): How many maximum items will be recommended to each user.
        threshold_from_past (int): How many max recommendation from past can be there in the hybrid.
    Example:
        python main.py
    """

    def __init__(self, no_of_recommendation, threshold_from_past, is_unit_test):
        """Sets the Connection string of Database by reading the envi variables.
        Sets the number of recommendation to be produces for each user.
        Sets the number of items from Past to be used in Hybrid recommendation.
        Attributes:
            no_of_recommendation (int): How many maximum items will be recommended to each user.
            threshold_from_past (int): How many max recommendation from past can be there in the hybrid.
            is_unit_test (bool): Whether running code for Unittest, if Yes then change the path for file path
        """

        # Setting up Database connection string from Envi Variables. Provider here is Oracle but it can by other dbms
        # too.
        _db_prov = os.environ.get('APP_DBPROVIDER')
        _user = os.environ.get('APP_USER')
        _pwd = os.environ.get('APP_PASSWORD')
        _host = os.environ.get('APP_CONNECTIONSTRING')

        #: Connection String to connect database
        self._conn_string = _db_prov + '://' + _user + ':' + _pwd + '@' + _host
        #: A unique name of the process.
        self._job_name = 'RecommendationSystem'
        #: total number of maximum recommendation we want for each user
        self._no_of_recommendation = no_of_recommendation
        #: for the hybrid recommendation, how many max items should be from past
        self._threshold_from_past = threshold_from_past        
        #: if Yes then change the path for file path
        self._is_unit_test = is_unit_test
        #: categoryid of vendor
        self.category_id = 90

        if is_unit_test:
            self._root_file_path = "tests"
        else:
            self._root_file_path = "src"

    def run_main_job(self):
        """Imports the data, runs the model and exports the recommendation back to database.
        Returns:
            Exists with exit code - 1 for failure, 0 for success.
        """

        # inserting start of job in DB and getting jobid
        jobid = self._execute_job_start_script('Running')
        # creating folder structure to save the imported and exported csvs
        self._create_folders_to_save_csv(jobid)
        # initializing the Logger from here
        logger = Logger(__name__, jobid, self._root_file_path).logger
        try:
            # run all imports i.e. import data from Database and save as CSV
            self._run_all_imports(jobid, logger)
            # running the model i.e. creating the recommendation and saving the data in CSV
            self._run_the_rec_sys_model(jobid, logger)
            # run all exports i.e save data fro CSV into the Database
            self._run_all_export(jobid, logger)

        except Exception as error:
            logger.exception(error)
            # make an entry in database for job failure
            self._mark_exception_in_database_for_job_execution('Failed', jobid, error, logger)
            exit(1)

        else:

            # make an entry in database for job completion
            self._mark_job_completed_in_database('Completed', jobid, logger)
            logger.info("Job {} finished successfully!!!".format(jobid))
            exit(0)

    def _execute_job_start_script(self, status):
        """Connects to database and makes an entry in the database and gets the unique Jobid
        Args:
            status: Constant string.
        Returns:
            a unique jobid.
        """

        engine = create_engine(self._conn_string)
        with engine.connect() as conn:
            jobid = self._make_job_start_entry_in_db(conn, status)
        return jobid

    def _make_job_start_entry_in_db(self, conn, status):
        """Makes entry in database in transaction
        Args:
            status: Constant string.
        Returns:
            a unique jobid.
        """
        # beginning the transaction
        tran = conn.begin()
        # make an entry in DB about start of job and grab the jobid
        jobid = self._create_job_start_entry(conn, status)
        # committing the transaction
        tran.commit()
        # closing the connection
        conn.close()
        return jobid

    def _create_job_start_entry(self, conn, status):
        """executes the sql statement and gets the unique Jobid from database.
        Args:
            conn: Database connection string.
            status: Constant string.
        Returns:
            a unique jobid.
        """

        conn.execute("""
            INSERT INTO AIML_JOB_RUN_STATUS (AIML_JOB_RUN_ID,AIML_JOB_ID,MODEL_VERSION,STATUS) 
            VALUES (SEQ_AIML_JOB_RUN_ID.nextval, 
            (select AIML_JOB_ID from AIML_JOBS where AIML_JOB_NAME='""" + self._job_name + """' and category_id="""+str(self.category_id)+""" and status=1),
            (select MODEL_VERSION from AIML_JOBS where AIML_JOB_NAME='""" + self._job_name + """' and category_id="""+str(self.category_id)+""" and status=1),
            '""" + status + """')
            """)

        data = conn.execute("""SELECT MAX(AIML_JOB_RUN_ID) AS AIML_JOB_RUN_ID FROM AIML_JOB_RUN_STATUS
                WHERE AIML_JOB_ID = (select AIML_JOB_ID from AIML_JOBS where AIML_JOB_NAME='""" + self._job_name + """' and category_id="""+str(self.category_id)+""" and status=1)
                """)

        # get the inserted jobid
        return data.fetchone()[0]

    def _create_folders_to_save_csv(self, jobid):
        """Creates folder with foldername appended with jobid and cretes the respective folder to save CSVs.
        Args:
            jobid: A unique job run sequence.
        """

        parent_dir = "./"+self._root_file_path+"/data/jobid" + str(jobid)
        pred_directory = "predictionfiles"
        path = os.path.join(parent_dir, pred_directory)
        os.makedirs(path, exist_ok=True)
        train_dir = "trainingfiles"
        path = os.path.join(parent_dir, train_dir)
        os.makedirs(path, exist_ok=True)

    def _run_all_imports(self, jobid, logger):
        """run all imports i.e. import data from Database and save as CSV
        Args:
            jobid: A unique job run sequence.
        """

        logger.info("Started import for ImportActivityData")
        import_activity = ImportActivityData(jobid, self._root_file_path)
        import_activity.import_activity_data(self._conn_string)
        logger.info("Finished import for ImportActivityData")

    def _run_the_rec_sys_model(self, jobid, logger):
        """Reads the activity data and generates various recommendations.
        Args:
            jobid: A unique job run sequence.
        """

        logger.info("Started RecSysHybrid")
        rec = RecSysHybrid(jobid, self._no_of_recommendation, self._threshold_from_past, self._root_file_path)
        rec.rec_sys_for_hybrid_model()
        logger.info("Finished RecSysHybrid")

    def _run_all_export(self, jobid, logger):
        """run all exports i.e save data from CSV into the Database
        Args:
            jobid: A unique job run sequence.
        """

        logger.info("Started export for ExportHybridProd")
        exportthybridrec = ExportHybridProd(jobid, self._root_file_path)
        exportthybridrec.export_hybrid_rec(self._conn_string)
        logger.info("Finished import for ExportHybridProd")
        logger.info("Started export for ExportPastProd")
        exporttpastrec = ExportPastProd(jobid, self._root_file_path)
        exporttpastrec.export_past_rated_rec(self._conn_string)
        logger.info("Finished import for ExportPastProd")
        logger.info("Started export for ExportColFilterProd")
        exporttcolfiltertrec = ExportColFilterProd(jobid, self._root_file_path)
        exporttcolfiltertrec.export_colfilter_rec(self._conn_string)
        logger.info("Finished import for ExportColFilterProd")
        logger.info("Started export for ExportTrendingProd")
        exporttoptrendrec = ExportTrendingProd(jobid, self._root_file_path)
        exporttoptrendrec.export_trending_rec(self._conn_string)
        logger.info("Finished import for ExportTrendingProd")

    def _mark_exception_in_database_for_job_execution(self, status, jobid, error, logger):
        """Combines the action of creating campaign as ACTIVATECAMPAIGN and DOWNLOADCAMPAIGN.
        Args:
            status: Constant string.
            jobid: A unique job run sequence.
            error: Error occured during the jobrun.
        """

        engine = create_engine(self._conn_string)
        with engine.connect() as conn:
            logger.info("Inserting Job failure in DB!")
            tran = conn.begin()
            conn.execute("""
                UPDATE AIML_JOB_RUN_STATUS 
                SET FINISH_DATE_TIME = SYSDATE,
                STATUS='""" + status + """',
                STATUS_MESSAGE='""" + str(error).replace("'", "''")[0:2500] + """'
                WHERE AIML_JOB_RUN_ID=""" + str(jobid) + """
                """)

            tran.commit()
            logger.info("Closing Connection!")
            conn.close()

    def _mark_job_completed_in_database(self, status, jobid, logger):
        """Combines the action of creating campaign as ACTIVATECAMPAIGN and DOWNLOADCAMPAIGN.
        Args:
            status: Constant string.
            jobid: A unique job run sequence.
        """
        engine = create_engine(self._conn_string)
        with engine.connect() as conn:
            logger.info("Inserting Job Completion in DB!")
            tran = conn.begin()
            conn.execute("""
                UPDATE AIML_JOB_RUN_STATUS 
                SET FINISH_DATE_TIME = SYSDATE,
                STATUS='""" + status + """'
                WHERE AIML_JOB_RUN_ID=""" + str(jobid) + """
                """)

            tran.commit()
            logger.info("Closing Connection!")
            conn.close()


if __name__ == '__main__':

    # params: no_of_recommendation (how many items we want to recommend), threshold_from_past (
    # how many max recommendation from past can be there in the hybrid),
    # is_unit_test (if true, it will refer to different source folder for running the unit test cases)
    recmain = RecSys(20,
                     3,
                     False)

    recmain.run_main_job()