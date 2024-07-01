"""
Implementation of base service.
"""

import contextlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


class BaseService():

    pause_duration = 60 # seconds
    attempts = 30
    attempt_timeout = 10 # seconds
    subfolders = ['arrived', 'waiting', 'ready']
    optional_folders = [
        'tessellation_visualizations', 
        'augmentation_visualizations', 
        'extracted_features',
    ]
    environment_vars = ['SLURM_JOBID', 'SLURM_NODELIST', 'SLURM_GPUS_PER_NODE']

    def __init__(
        self,
        directory: Union[str, Path], 
        config_file: str,
    ) -> None:
        """
        Initialize service for data transfer.
        
        Args:
            directory:  Directory where all superbatches are stored.
            config_file:  Name of config file.
        """
        # configure logging
        start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logging.basicConfig(
            level=logging.INFO,
            filename=f'{start}_{self.__class__.__name__}_log.txt',
            format='%(asctime)s - %(message)s',
            datefmt='%d/%m/%Y %I:%M:%S %p',
            encoding='utf-8',
        )
        self.logger = logging.getLogger(__name__)
        # connect exception hook
        def exception_logger(type, value, tb):
            self.logger.exception(''.join(traceback.format_exception(type, value, tb)))
            sys.__excepthook__(type, value, tb)
        sys.excepthook = exception_logger

        # log environment variables if available
        environment_dict = dict(os.environ)
        for var in self.environment_vars:
            if var in environment_dict:
                self.logger.info(f'{var}: {environment_dict[var]}')
        
        # define and log ID
        self.ID = datetime.now().strftime('%d%m%y%H%M%S%f')
        self.logger.info(f'ID: {self.ID}')

        # check if the directory exists
        self.directory = Path(directory)
        while not self.directory.exists():
            self.logger.info('Directory for superbatches does not yet exist')
            self.logger.info(f'Retry transfer after {self.pause_duration} seconds')
            time.sleep(self.pause_duration)

        # check if the config file exists
        while config_file not in os.listdir(self.directory):
            self.logger.info('Config file does not exist in the directory')  
            self.logger.info(f'Retry transfer after {self.pause_duration} seconds')
            time.sleep(self.pause_duration)  

        # load the config file
        self.config = self.load_json(self.directory / config_file)
        self.logger.info(f'Pipeline config: {str(self.config)}')
        # load the status file
        self.status = self.load_json(self.directory / self.config['status_file'])
        # load the data file
        self.data = self.load_json(self.directory / self.config['dataset_file'])
        # load the order file
        self.order = self.load_json(self.directory / self.config['order_file'])

    def attempt(func):
        """
        Decorator for attempting a method several times.
        """
        def wrapper(self, *args, **kwargs):
            for i in range(self.attempts):
                try:
                    output = func(self, *args, **kwargs)
                except Exception as error:
                    self.logger.info(f'Unsuccessful attempt at executing: {func.__name__}')
                    time.sleep(self.attempt_timeout)
                    if i == self.attempts-1:
                        raise error
                else:
                    return output
        return wrapper

    @attempt
    def load_json(self, 
        path: Union[str, Path], 
        renamed_path: Optional[Union[str, Path]] = None, 
        revert_renaming: bool = True,
    ) -> Any:
        """
        Loads information from JSON file, optionally renaming the file before and after.

        Args:
            path:  Path to JSON file.
            renamed_path:  Renamed path to JSON file.
            revert_renaming:  Indicates whether file renaming is reverted 
                after loading information (only if renamed_path is not None).

        Returns:
            info:  Information from JSON file.
        """
        if renamed_path is not None:
            os.rename(path, renamed_path)
            with open(renamed_path, 'r') as f:
                info = json.loads(f.read())
            if revert_renaming:
                os.rename(renamed_path, path)
        else:
            with open(path, 'r') as f:
                info = json.loads(f.read())

        return info

    def get_path(self, path_instance: Path) -> str:
        """ 
        Convert Path instance to string depending on the OS.

        Args:
            path_instance: Path instance.

        Returns:
            path_instance: path converted to string depending on the OS.
        """
        if not isinstance(path_instance, Path):
            path_instance = Path(path_instance)
        
        if self.config['remote_OS'].lower() in ['windows', 'win32', 'win']:
            return str(path_instance.as_uri())
        else:
            return str(path_instance.as_posix())

    def load_order(self):
        """ 
        Load the order file.
        """
        order_path = self.directory / self.config['order_file']
        self.order = self.load_json(order_path)

    def load_status(self):
        """ 
        Load the status file.
        """
        status_path = self.directory / self.config['status_file']
        self.status = self.load_json(status_path)

    @contextlib.contextmanager
    def update_local_status(self):
        """ 
        Wrapper code to update the status file.
        """
        # define paths
        status_path = self.directory / self.config['status_file']
        locked_status_path = str(status_path).replace('.json', f'_locked_{self.ID}.json')
        # load remote status information
        self.status = self.load_json(status_path, locked_status_path, revert_renaming=False)
        yield
        # update remote status information file
        with open(locked_status_path, 'w') as f:
            f.write(json.dumps(self.status))
        os.rename(locked_status_path, status_path)