# -*- coding: utf-8 -*-
"""export_past_prod.py

This module exports the Past used products Data from CSV to Database table
by connecting to the database provider

Todo:
    * Nothing

"""

# import the module
from sqlalchemy import create_engine
import sqlalchemy as sa
import pandas as pd
from src.logger import Logger


class ExportPastProd:
    """exports the Past used products Data from CSV to Database table.

    Attributes:
        jobid (int): jobid is unique job run sequence created in database.
        root_file_path (string): Based on whether running Unit test, this folder will change
    """

    def __init__(self, jobid, root_file_path):
        """Sets the jobid, logger and file path parameter at the time class initialization

        Args:
            jobid (int): jobid is unique job run sequence created in database.

        """
        self._jobid = jobid
        #: logger is available in logger.py from python module logging
        self._logger = Logger(__name__, self._jobid, root_file_path).logger
        #: Path of the CSV file which has to be exported.
        self._IN_FILE_PATH = './'+root_file_path+'/data/jobid' + str(self._jobid) + '/predictionfiles/TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.CSV'

    def export_past_rated_rec(self, conn_string):
        """Reads the CSV and calls class method to export the data.

        Args:
            conn_string: Database connection string of the respective provider.

        """

        # read csv from folder
        df = pd.read_csv(self._IN_FILE_PATH)
        if df.shape[0] == 0:
            raise Exception("Data not found in the csv file: TOP_OWN_PAST_RATED_PROD_FOR_PORTALS_USERS.csv")
        self._modify_col_and_export(df, conn_string)

    def _modify_col_and_export(self, df, conn_string):
        """Modify the column datatype of the dataframe to match with the one in Database, connects DB and exports

        Args:
            df: The Dataframe to be exported to DB.
            conn_string: Database connection string of the respective provider.

        """
        # modify the column as per table in the database
        df = self._modify_columns(df)

        # create sqlalchemy engine
        self._logger.info("Connecting to DB!")
        engine = create_engine(conn_string)

        with engine.connect() as conn:
            self._logger.info("You're connected!")
            self._export_data_in_transaction(conn, engine, df)

    def _export_data_in_transaction(self, conn, engine, df):
        """Begins the transaction and inserts the data in transaction. Rollbacks in case of issue.

        Note:
            This method inserts DB in transaction, either entire data will go or none.

        Args:
            conn: Connection object provided by sqlalchemy.
            engine: DBAPI provided by sqlalchemy, delivered to the SQLAlchemy application through a connection pool and a Dialect
            df: The Dataframe to be exported to DB.

        """
        try:
            self._logger.info("Transaction begins!")
            tran = conn.begin()
            # export data to Database
            self._export_data_in_db(conn, engine, df)
            tran.commit()
            self._logger.info("Data exported successfully!!!")
        except Exception as error:
            self._logger.error("Error occurred, rolling back!", error)
            tran.rollback()
            raise
        finally:
            self._logger.info("Closing Connection!")
            conn.close()

    def _modify_columns(self, df):
        """Adds a column AIML_JOB_RUN_ID with value as _jobid and converts the datatype of all columns.

        Args:
            df: The Dataframe to be exported to DB.

        Returns:
            Dataframe and a disctionary of column name and its modified datatype.

        """
        # modify the column as per table
        df["AIML_JOB_RUN_ID"] = self._jobid
        df["AIML_JOB_RUN_ID"] = pd.to_numeric(df["AIML_JOB_RUN_ID"])
        df["USER_ID"] = pd.to_numeric(df["USER_ID"])
        df["RATING"] = pd.to_numeric(df["RATING"])
        object_columns = [c for c in df.columns[df.dtypes == 'object'].tolist()]
        for c in object_columns:
            df[c] = df[c].astype('|S')
        return df

    def _export_data_in_db(self, conn, engine, df):
        """Runs the SQL command to insert the data in DB.

        Args:
            conn: Connection object provided by sqlalchemy.
            engine: DBAPI provided by sqlalchemy, delivered to the SQLAlchemy application through a connection pool and a Dialect
            df: The Dataframe to be exported to DB.

        """
        self._logger.info("Truncating table PAST_RATEITEM_PORTAL_USER!")
        conn.execute("""TRUNCATE TABLE PAST_RATEITEM_PORTAL_USER""")
        self._logger.info("Inserting data into PAST_RATEITEM_PORTAL_USER!")
        df.to_sql('PAST_RATEITEM_PORTAL_USER', con=engine, if_exists='append', chunksize=1000, index=False)
        self._logger.info("Inserting data into PAST_RATEITEM_PORTAL_USER_HIST!")
        conn.execute("""INSERT INTO PAST_RATEITEM_PORTAL_USER_HIST (USER_ID,ITEM_ID,RATING,PORTAL,AIML_JOB_RUN_ID)
        SELECT USER_ID,ITEM_ID,RATING,PORTAL,AIML_JOB_RUN_ID FROM PAST_RATEITEM_PORTAL_USER""")
        self._logger.info("Rows inserted successfully: {}".format(df.shape[0]))
