# -*- coding: utf-8 -*-
"""logger.py

This module is a wrapper to the logging module provided by python.
This module helps to have all the logger setting at a centralized place.

Todo:
    * Nothing

"""

import logging


class Logger:

    def __init__(self, logger_name, jobid, root_file_path):
        """Sets the module level function getLogger and jobid parameter at the time class initialization

        Args:
            logger_name (str): Filename in which the logging happens.
            jobid (int): jobid is unique job run sequence created in database.

        """
        self.logger = logging.getLogger(logger_name)
        # logger level by default is INFO, can be changed to error, critical, warning, exception, log
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s',
                                      datefmt='%d-%b-%y %H:%M:%S')
        file_handler = logging.FileHandler('./'+root_file_path+'/data/jobid' + str(jobid) + '/JobId' + str(jobid) + '.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
