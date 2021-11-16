# -*- coding: utf-8 -*-
"""import_activity_data.py

This module imports the User's activity Data from Database to CSV
by connecting to the database provider.

Todo:
    * Nothing

"""

# import the module
from sqlalchemy import create_engine
import pandas as pd
from src.logger import Logger


class ImportActivityData:
    """Imports the User's activity Data from Database to CSV.

    Attributes:
        jobid (int): jobid is unique job run sequence created in database.

    """

    def __init__(self, jobid, root_file_path):
        """Sets the jobid, logger and file path parameter at the time class initialization

        Args:
            jobid (int): jobid is unique job run sequence created in database.
            root_file_path (string): Based on whether running Unit test, this folder will change

        """
        self._jobid = jobid
        #: logger is available in logger.py from python module logging
        self._logger = Logger(__name__, self._jobid, root_file_path).logger
        #: Path to where the CSV File will be created.
        self._OUT_FILE_PATH = './'+root_file_path+'/data/jobid' + str(self._jobid) + '/trainingfiles/USER_ACTIVITY_WITH_TIME_PORTAL.CSV'

    def import_activity_data(self, conn_string):
        """Connects to the Database, fetches the data and saves in a CSV File.

        Args:
            conn_string: Database connection string of the respective provider.

        """

        activity_query = """
           SELECT DISTINCT
    lts.userid           AS "USER_ID",
    pm.productmasterid   AS "ITEM_ID",
    TRIM(upper(lta.type)) AS "ACTIVITY_TYPE",
    lta.datetime         AS "TIMESTAMP",
    (
        SELECT
            upper(p.name) AS name
        FROM
            customer.managedbundlecompanysite   mbc
            JOIN customer.portal_program             pp ON ( mbc.managedbundleid = pp.programid
                                                 AND pp.program_type_id = 2 )
            JOIN customer.portal                     p ON p.portalid = pp.portalid
        WHERE
            mbc.companysiteid = lts.companysiteid
            AND p.name IS NOT NULL
    ) AS portalname,
    lts.libtrackingsessionid,
    (
        SELECT
            is_aiml_test_user
        FROM
            framewrk.companysiteuser csu
        WHERE
            csu.userid = lts.userid
            AND csu.companysiteid = lts.companysiteid
    ) is_aiml_test_user,
    TRIM(upper(nvl(pm.displayname, pm.name))) AS "ITEM_NAME"
FROM
    customer.libtrackingactivity         lta
    JOIN customer.libtrackingsession          lts ON lta.libtrackingsessionid = lts.libtrackingsessionid
                                            AND nvl(lts.internaluser, 0) = 0
                                            AND lts.libcompanyid = 65774
                                            AND lta.campaignid != - 1
                                            AND lta.campaignid IS NOT NULL
    JOIN framewrk.users                       u ON u.usersid = lts.userid
                             AND u.creatorcompanysiteid = lts.companysiteid
                        ----------------------------------
    JOIN framewrk.companysiteregistration     csr ON lts.companysiteid = csr.companysiteid
                                                 AND nvl(csr.testsite, 0) = 0
    JOIN customer.campaign_plans              cp ON cp.campaign_plan_id = lta.campaignid
    JOIN customer.product_instance_campaign   pic ON pic.campaign_plan_id = decode(cp.managed_campaign_plan_id, NULL, cp.campaign_plan_id
    , 0, cp.campaign_plan_id,
                                                                                 cp.managed_campaign_plan_id)
    JOIN customer.product_instance            pi ON pi.product_instance_id = pic.product_instance_id
    JOIN customer.productmaster               pm ON pi.product_master_id = pm.productmasterid
    LEFT JOIN framewrk.usercontactemail            uce ON uce.userid = u.usersid
    LEFT JOIN framewrk.email                       e ON e.emailid = uce.emailid
WHERE
    lts.libcompanyid = 65774
    AND lts.companysiteid != lts.libcompanyid
    AND lower(emailaddress) NOT LIKE '%@ingrammicro.com%'
    AND (
       SELECT
                    COUNT(1)
                FROM
                    customer.product_instance pi2
                    JOIN customer.product_instance_bundle   pib ON pib.product_instance_id = pi2.product_instance_id
                    JOIN customer.portal_program            pp ON pib.managed_bundle_id = pp.programid
                WHERE
                    pi2.product_master_id = pm.productmasterid
                    AND pi2.is_active_version = 1
                    AND pi2.workflow_status_id = 2
                    AND pp.program_type_id = 2
    ) > 0
UNION
SELECT DISTINCT
    lts.userid           AS "USER_ID",
    pm.productmasterid   AS "ITEM_ID",
    TRIM(upper(lta.type)) AS "ACTIVITY_TYPE",
    lta.datetime         AS "TIMESTAMP",
    (
        SELECT
            upper(p.name) AS name
        FROM
            customer.managedbundlecompanysite   mbc
            JOIN customer.portal_program             pp ON ( mbc.managedbundleid = pp.programid
                                                 AND pp.program_type_id = 2 )
            JOIN customer.portal                     p ON p.portalid = pp.portalid
        WHERE
            mbc.companysiteid = lts.companysiteid
            AND p.name IS NOT NULL
    ) AS portalname,
    lts.libtrackingsessionid,
    (
        SELECT
            is_aiml_test_user
        FROM
            framewrk.companysiteuser csu
        WHERE
            csu.userid = lts.userid
            AND csu.companysiteid = lts.companysiteid
    ) is_aiml_test_user,
    TRIM(upper(nvl(pm.displayname, pm.name))) AS "ITEM_NAME"
FROM
    customer.libtrackingactivity       lta
    JOIN customer.libtrackingsession        lts ON lta.libtrackingsessionid = lts.libtrackingsessionid
                                            AND nvl(lts.internaluser, 0) = 0
                                            AND lts.libcompanyid = 65774
                                            AND lta.campaignid = - 1
                                            AND lta.type = 'EnterProduct'
    JOIN framewrk.users                     u ON u.usersid = lts.userid
                             AND u.creatorcompanysiteid = lts.companysiteid
                        ----------------------------------
    JOIN framewrk.companysiteregistration   csr ON lts.companysiteid = csr.companysiteid
                                                 AND nvl(csr.testsite, 0) = 0
    JOIN customer.product_instance          pi ON pi.product_instance_id = lta.value
    JOIN customer.productmaster             pm ON pi.product_master_id = pm.productmasterid
    LEFT JOIN framewrk.usercontactemail          uce ON uce.userid = u.usersid
    LEFT JOIN framewrk.email                     e ON e.emailid = uce.emailid
WHERE
    lts.libcompanyid = 65774
    AND lts.companysiteid != lts.libcompanyid
    AND lower(emailaddress) NOT LIKE '%@ingrammicro.com%'
    AND (
        SELECT
                    COUNT(1)
                FROM
                    customer.product_instance pi2
                    JOIN customer.product_instance_bundle   pib ON pib.product_instance_id = pi2.product_instance_id
                    JOIN customer.portal_program            pp ON pib.managed_bundle_id = pp.programid
                WHERE
                    pi2.product_master_id = pm.productmasterid
                    AND pi2.is_active_version = 1
                    AND pi2.workflow_status_id = 2
                    AND pp.program_type_id = 2
    ) > 0   
UNION
SELECT DISTINCT
    lts.userid           AS "USER_ID",
    pm.productmasterid   AS "ITEM_ID",
    TRIM(upper(lta.type)) AS "ACTIVITY_TYPE",
    lta.datetime         AS "TIMESTAMP",
    (
        SELECT
            upper(p.name) AS name
        FROM
            customer.managedbundlecompanysite   mbc
            JOIN customer.portal_program             pp ON ( mbc.managedbundleid = pp.programid
                                                 AND pp.program_type_id = 2 )
            JOIN customer.portal                     p ON p.portalid = pp.portalid
        WHERE
            mbc.companysiteid = lts.companysiteid
            AND p.name IS NOT NULL
    ) AS portalname,
    lts.libtrackingsessionid,
    (
        SELECT
            is_aiml_test_user
        FROM
            framewrk.companysiteuser csu
        WHERE
            csu.userid = lts.userid
            AND csu.companysiteid = lts.companysiteid
    ) is_aiml_test_user,
    TRIM(upper(nvl(pm.displayname, pm.name))) AS "ITEM_NAME"
FROM
    customer.libtrackingactivity         lta
    JOIN customer.libtrackingsession          lts ON lta.libtrackingsessionid = lts.libtrackingsessionid
                                            AND nvl(lts.internaluser, 0) = 0
                                            AND lts.libcompanyid = 65774
                                            AND lta.campaignid = - 1
                                            AND lta.type IN (
        'ActivateCampaign',
        'DownloadCampaign'
    )
    JOIN framewrk.users                       u ON u.usersid = lts.userid
                             AND u.creatorcompanysiteid = lts.companysiteid
                        ----------------------------------
    JOIN framewrk.companysiteregistration     csr ON lts.companysiteid = csr.companysiteid
                                                 AND nvl(csr.testsite, 0) = 0
    JOIN customer.product_instance_campaign   pic ON pic.campaign_plan_id = lta.value
    JOIN customer.product_instance            pi ON pi.product_instance_id = pic.product_instance_id
    JOIN customer.productmaster               pm ON pi.product_master_id = pm.productmasterid
    LEFT JOIN framewrk.usercontactemail            uce ON uce.userid = u.usersid
    LEFT JOIN framewrk.email                       e ON e.emailid = uce.emailid
WHERE
    lts.libcompanyid = 65774
    AND lts.companysiteid != lts.libcompanyid
    AND lower(emailaddress) NOT LIKE '%@ingrammicro.com%'
    AND (
        SELECT
                    COUNT(1)
                FROM
                    customer.product_instance pi2
                    JOIN customer.product_instance_bundle   pib ON pib.product_instance_id = pi2.product_instance_id
                    JOIN customer.portal_program            pp ON pib.managed_bundle_id = pp.programid
                WHERE
                    pi2.product_master_id = pm.productmasterid
                    AND pi2.is_active_version = 1
                    AND pi2.workflow_status_id = 2
                    AND pp.program_type_id = 2
    ) > 0
    """
        # create sqlalchemy engine
        self._logger.info("Connecting to DB!")
        engine = create_engine(conn_string)
        conn = engine.connect()
        self._logger.info("You're connected!")
        self._logger.info("Fetching User Activity data!")
        data = conn.execute(activity_query)
        all_rows = data.fetchall()
        self._logger.info(all_rows)
        self._export_activity_data_to_csv(all_rows)
        self._logger.info("Closing Connection!")
        conn.close()

    def _export_activity_data_to_csv(self, all_rows):
        """Creates the Dataframe from tuple of tuples and export the dataframe to CSV.

        Args:
            all_rows: Tuple of Tuples returned from sqlalchemy fetchall().

        """
        # Converting fetched data to dataframe
        df = pd.DataFrame(list(all_rows), columns=["USER_ID", "ITEM_ID", "ACTIVITY_TYPE", "TIMESTAMP", "PORTALNAME",
                                                   "LIBTRACKINGSESSIONID", "IS_AIML_TEST_USER", "ITEM_NAME"])
        self._logger.info("Saving data to csv:{}".format('USER_ACTIVITY_WITH_TIME_PORTAL.CSV'))
        if df.shape[0] > 0:
            # Exporting Dataframe to Folder as CSV
            df.to_csv(self._OUT_FILE_PATH, index=False)
