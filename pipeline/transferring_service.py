"""
Implementation of whole slide image (WSI) transfer service.
"""

import contextlib
import getpass
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import paramiko
from natsort import natsorted
from paramiko import SSHClient
from scp import SCPClient


GIGABYTE = (1024**3)

class TransferringService():

    pause_duration = 60 # seconds
    attempts = 120
    attempt_timeout = 30 # seconds
    socket_timeout = 40
    max_transfer_lead = 2 # superbatch(es)
    decimals = 2
    subfolders = ['arrived', 'waiting', 'ready']
    optional_folders = [
        'tessellation_visualizations', 
        'augmentation_visualizations', 
        'extracted_features',
    ]
    environment_vars = ['SLURM_JOBID', 'SLURM_NODELIST', 'SLURM_GPUS_PER_NODE']

    def __init__(
        self, 
        config: dict[str, Any], 
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """
        Initialize service for data transfer.
    
        Note:  It is assumed that all filenames for the data that are going to be
            be transferred are unique.
        
        Args:
            config:  Dictionary with data transfer configuration.
            df:  Dataframe with dataset information. 
                 The following columns must be availale:
                 - 'set': indicates partition (e.g., training, validation, test)
                 - 'paths': string with one path or sequence with multiple paths
            verbose:  Indicates whether file transfer progress is printed.
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
        self.progress = progress if verbose else None
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

        # initialize instance variables
        self.config = config
        self.remote_dir = Path(self.config['remote_dir'])
        self.logger.info(f'Pipeline config: {str(self.config)}')
        self.ssh_client = None
        self.sftp_client = None

        # check if the max size of the remote directory is valid
        if self.config['max_size_remote_dir'] <= 0:
            raise ValueError('Maximum storage space must exceed zero.')
        for variant in self.config['variants']:
            if self.config['variants'][variant]['max_size'] is not None:
                if self.config['max_size_remote_dir'] < self.config['variants'][variant]['max_size']:
                    raise ValueError(
                        'Maximum size of a superbatch variant cannot exceed '
                        'the maximum size of the remote directory.'
                    )
        # check if 'set' and 'paths' are columns in the dataframe
        if 'set' not in df.columns:
            raise ValueError("Column 'set' not in dataframe.")
        if 'paths' not in df.columns:
            raise ValueError("Column 'paths' not in dataframe.")
        columns = list(df.columns)
        columns.remove('set')

        # get all filenames
        filenames = []
        for paths in list(df['paths']):
            if isinstance(paths, str):
                paths = [paths]
            filenames.extend([Path(path).name for path in paths])
        # check if all filenames are unique
        if len(filenames) != len(set(filenames)):
            raise ValueError('There are files with the same filename in the dataset.')

        # convert the dataframe with data information to a dictionary with sets
        # as keys and list with tuples with the content of the original columns
        # as values
        self.data = {}
        for dataset in list(set(df['set'])):
            if dataset not in self.config['variants']:
                raise ValueError(
                    f"Unknown variant for data category: {dataset}.",
                )
            datapoints = []
            for specimen_index, row in df.iterrows():
                if row['set'] != dataset:
                    continue
                datapoint = {'specimen_index': specimen_index}
                for column in df.columns:
                    if column != 'set':
                        value = row[column]
                        # convert numpy datatypes to prevent JSON serialization error
                        if isinstance(value, np.integer):
                            value = int(value)
                        if isinstance(value, np.floating):
                            value = float(value)
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        datapoint[column] = value
                datapoints.append(datapoint)
            self.data[dataset] = datapoints
      
        # initialize a dictionary to keep track of the status for each superbatch
        # and the order of the data for each epoch
        self.order = {}
        self.status = {
            'transfer_completed': False, 
            'preprocessing_completed': False, 
            'feature_extraction_completed': (
                False if self.config['feature_extraction'] else 'not applicable'
            ),
            'accumulated_size_remote_dir': 0, # GB
            'current_size_remote_dir': 0, # GB
            'superbatch_order': [],
        } 
    
    def establish_connection(self) -> None:
        """
        Initialize SSH client and SFTP client.
        """
        # ask the user for the host, username, and password
        host = input('Host: ')
        username = input('Username: ')
        password = getpass.getpass()
        
        # initialize SSH client and SFTP client
        self.logger.info(f'Establish SSH and SFTP connection to {host} as user {username}')
        self.ssh_client = get_ssh_client(host, username, password)
        self.sftp_client = self.ssh_client.open_sftp()

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
    def copy_to_remote(
        self,
        local_path: str, 
        remote_path: str, 
    ) -> None:
        """ 
        Create scp client and transfer data.
        """
        scp = SCPClient(
            self.ssh_client.get_transport(), 
            progress=self.progress,        
            socket_timeout=self.socket_timeout,
        )    
        scp.put(local_path, remote_path)
        scp.close()

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
        path = self.get_path(path)
        if renamed_path is not None:
            renamed_path = self.get_path(renamed_path)
            self.sftp_client.rename(path, renamed_path)
            with self.sftp_client.open(renamed_path, 'r') as f:
                info = json.loads(f.read())
            if revert_renaming:
                self.sftp_client.rename(renamed_path, path)
        else:
            with self.sftp_client.open(path, 'r') as f:
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

    def load_status(self) -> None:
        """ 
        Load the status from the remote status file.
        """
        remote_status_path = self.get_path(self.remote_dir / self.config['status_file'])
        self.status = self.load_json(remote_status_path)

    @contextlib.contextmanager
    def update_remote_status(self) -> None:
        """ 
        Wrapper code to update the remote status file.
        """
        # define paths
        remote_status_path = self.get_path(self.remote_dir / self.config['status_file'])
        locked_remote_status_path = remote_status_path.replace('.json', f'_locked_{self.ID}.json')
        # load remote status information
        self.status = self.load_json(remote_status_path, locked_remote_status_path, revert_renaming=False)
        yield
        # update remote status information file
        with self.sftp_client.open(locked_remote_status_path, 'w') as f:
            f.write(json.dumps(self.status))
        self.sftp_client.rename(locked_remote_status_path, remote_status_path)

    @contextlib.contextmanager
    def update_remote_order(self) -> None:
        """ 
        Wrapper code to update the remote order file.
        """
        # define paths
        remote_order_path = self.get_path(self.remote_dir / self.config['order_file'])
        locked_remote_order_path = remote_order_path.replace('.json', f'_locked_{self.ID}.json')
        # load remote order information
        self.order = self.load_json(remote_order_path, locked_remote_order_path, revert_renaming=False)
        yield
        # update remote order information file
        with self.sftp_client.open(locked_remote_order_path, 'w') as f:
            f.write(json.dumps(self.order))
        self.sftp_client.rename(locked_remote_order_path, remote_order_path)

    def init_remote_dir(self) -> None:
        """
        Initialize the remote directory by creating the remote folder and files
        to keep track of the status of the if necessary.
        """
        # check if the remote directory already exists, if not, create it
        remote_folders = self.sftp_client.listdir(self.get_path(self.remote_dir.parent))
        if self.remote_dir.name not in remote_folders:
            self.sftp_client.mkdir(self.get_path(self.remote_dir))
            self.logger.info(f'Created remote directory: {self.remote_dir}')

        # check if a config file, data info file, order file, and status file exist 
        # in the remote directory, if not, create each of them
        files = [
            ('config_file', self.config, 'check'), 
            ('dataset_file', self.data, 'check'),
            ('order_file', self.order, 'replace'),
            ('status_file', self.status, 'replace'),
        ]
        remote_folders = self.sftp_client.listdir(self.get_path(self.remote_dir))
        for file, info, action in files:
            path = self.get_path(self.remote_dir / self.config[file])
            if self.config[file] not in remote_folders:
                with self.sftp_client.open(path, 'w') as f:
                   f.write(json.dumps(info))
                self.logger.info(f'Created remote file: {path}')
            else:
                remote_info = self.load_json(path)
                self.logger.info(f'Read remote file: {path}')
                # if the action is 'check', check if there are differences 
                # between the remote file and the specified information
                info = json.loads(json.dumps(info))
                if action == 'check':
                    if remote_info != info:
                        message = (f"There are differences between '{self.config[file]}'"
                                   " (remote) and the specified information.")
                        raise ValueError(message)
                # if the action is 'replace', replace the specified information 
                # by the remote information
                elif action == 'replace':
                    info = remote_info
                else:
                    raise ValueError('Invalid action.')

    def start(self) -> None:
        """ 
        Start the transfer of superbatches with data to remote location.
        """   
        # check if connection has been established
        if self.ssh_client is None or self.sftp_client is None:
            raise ConnectionError('No connection to remote server has been '
                                  'established yet.') 
        
        # record the total size of all files that belong to each specimen
        if self.config['record_size']:
            self.logger.info('Start recording the total size of all files '
                             'that belong to each specimen')
            for variant in self.data:
                sizes = []
                for i, specimen in enumerate(self.data[variant]):
                    # get the size of all files of the specimen
                    specimen_size = sum([os.path.getsize(path) for path in specimen['paths']])
                    specimen_size /= GIGABYTE
                    # add the size to specimen
                    self.data[variant][i]['size'] = round(specimen_size, self.decimals)
                    sizes.append(specimen_size)
                # determine whether the maximum specimen size exceeds the 
                # maximum superbatch size
                max_specimen_size = max(sizes)
                if self.config['variants'][variant]['max_size'] is not None:
                    max_superbatch_size = self.config['variants'][variant]['max_size']
                    if max_specimen_size > max_superbatch_size:
                        raise ValueError(f'The maximum specimen size ({max_specimen_size:0.2f} GB) '
                                         f'for the {variant} set exceeds the specified maximum '
                                         f'superbatch size of {max_superbatch_size:0.2f} GB.')
            self.logger.info('Finished recording the total size of all files '
                             'that belong to each specimen')

        # initialize remote directory
        self.init_remote_dir()

        # define states for logging
        report_at_start = True
        notify_max_size_reached = False

        # determine the order of the superbatch transfer based on the variant priority
        priority_variants = []
        for variant in self.config['variants']:
            if variant in self.data and not self.config['variants'][variant]['skip']:
                priority_variants.append((self.config['variants'][variant]['priority'], variant))
        priority_variants = natsorted(priority_variants)

        # loop over superbatch variants from high to low priority
        for _, variant in priority_variants:
            # get the names of the superbatch folders in the remote directory
            remote_folders = []
            for item in self.sftp_client.listdir(self.get_path(self.remote_dir)):
                if '.json' not in item:
                    remote_folders.append(item)
            remote_folders = natsorted(remote_folders)
            if report_at_start:
                report_at_start = False
                if len(remote_folders):
                    folders = ", ".join(remote_folders)
                    self.logger.info(f'Existing remote folders: {folders}')

            # add variant to status and order information if necessary
            self.logger.info(f'Start data transfer for superbatch type: {variant}')
            with self.update_remote_status():
                if variant not in self.status:
                    self.status[variant] = {
                        'transfer_completed': False,
                        'preprocessing_completed': False,
                        'feature_extraction_completed': (
                            False if self.config['feature_extraction'] 
                            else 'not applicable'
                        ),
                        'epochs': {},
                    }
            with self.update_remote_order():
                if variant not in self.order:
                    self.order[variant] = {}

            # start by checking the available information from previous superbatches
            # (if there is any), to continue where the process left off 
            continue_superbatch = False
            current_batch = 0
            
            # determine whether to start a new superbatch or continue with one
            for epoch in self.status[variant]['epochs']:
                for superbatch in self.status[variant]['epochs'][epoch]['superbatches']:
                    self.logger.info(f'Checking superbatch: {superbatch} (epoch {epoch})')
                    # check if the superbatch folder still exists
                    if superbatch not in remote_folders:
                        raise NotADirectoryError((
                            f"Superbatch folder '{superbatch}' is missing from "
                            " the remote directory."
                        ))
                    # check if the transfer has finished for the superbatch
                    elif self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['transferred']:
                        continue
                    
                    # raise an error if the transfer is unfinished for more than one superbatch 
                    if continue_superbatch is True:
                        raise ValueError(
                            'The transfer is unfinished for more than one superbatch.'
                        )
                    else:
                        # determine information for restarting data transfer from
                        # where the process left off
                        continue_superbatch = True
                        current_epoch = int(epoch)
                        current_index = self.status[variant]['epochs'][epoch]['transfer_index']

            # check if the transfer has already finished for this variant
            if (self.config['variants'][variant]['epochs'] == len(self.status[variant]['epochs']) 
                and not continue_superbatch):
                self.logger.info(f'Data transfer completed for superbatch type: {variant}')
                continue

            # initialize epoch, batch, and index variables
            epoch = current_epoch if continue_superbatch else 0
            batch = current_batch if continue_superbatch else 0
            index = current_index if continue_superbatch else 0
            if continue_superbatch:
                self.logger.info((
                    f'Continuing data transfer: [{variant}] epoch {epoch}, '
                    f'superbatch {batch}, index {index}'),
                )
            else:
                self.logger.info((
                    f'Starting data transfer: [{variant}] epoch {epoch}, '
                    f'superbatch {batch}, index {index}'),
                )
            # continue loop until the specified maximum number of iterations 
            # has been reached
            while epoch < self.config['variants'][variant]['epochs']:
                # check if the maximum superbatch transfer lead has been reached
                self.load_status()
                max_lead_reached = False
                if self.max_transfer_lead is not None:
                    superbatch_index = len(self.status['superbatch_order'])-1-self.max_transfer_lead
                    if superbatch_index >= 0:
                        selected_superbatch = self.status['superbatch_order'][superbatch_index]
                        epochs = self.status[selected_superbatch.split('-')[0]]['epochs']
                        for e in epochs:
                            if selected_superbatch in epochs[e]['superbatches']:
                                if not epochs[e]['superbatches'][selected_superbatch]['preprocessed']:
                                    max_lead_reached = True
                                    break
                # if the transfer lead exceeds the maximum lead, 
                # pause the transfer of slides
                if max_lead_reached:
                    self.logger.info(
                        f'Maximum transfer lead ({self.max_transfer_lead} superbatches) reached'
                    )
                    # retry to continue the transfer after a certain amount of time
                    self.logger.info(f'Retry transfer after {self.pause_duration} seconds')
                    time.sleep(self.pause_duration)
                    continue
                
                # add epoch to status information if necessary
                with self.update_remote_status():
                    if str(epoch) not in self.status[variant]['epochs']:
                        self.status[variant]['epochs'][str(epoch)] = { 
                            'transfer_index': index,
                            'superbatches': {},
                        }
                # add epoch to order information if necessary
                with self.update_remote_order():
                    if str(epoch) not in self.order[variant]:
                        order = list(range(len(self.data[variant])))
                        random.seed(self.config['seed']+epoch)
                        random.shuffle(order)
                        self.order[variant][str(epoch)] = order
                    else:
                        order = self.order[variant][str(epoch)]

                # define the name of the next superbatch
                superbatch = f'{variant}-{batch}'  

                # create the superbatch folder
                superbatch_path = self.remote_dir / superbatch
                if superbatch not in self.sftp_client.listdir(self.get_path(self.remote_dir)):
                    self.sftp_client.mkdir(self.get_path(superbatch_path))
                    self.logger.info(f'Created new superbatch directory: {superbatch_path}')
                                    
                # select optional folders
                selected_optional_folders = []
                if self.config['preprocessing_settings']['save_tessellation_visualizations']:
                    selected_optional_folders.append(self.optional_folders[0])
                if self.config['feature_extraction']:
                    selected_optional_folders.append(self.optional_folders[2])
                    if 'augmentation_config' in self.config['feature_extraction_settings']:
                        augmentation_config = self.config['feature_extraction_settings']['augmentation_config']
                        if augmentation_config['save_augmentation_visualizations']:
                            selected_optional_folders.append(self.optional_folders[1])
                                    
                # create subfolders for superbatch
                for subfolder in self.subfolders+selected_optional_folders:
                    if subfolder not in self.sftp_client.listdir(self.get_path(superbatch_path)):
                        new_subfolder_path = self.get_path(superbatch_path / subfolder)
                        self.sftp_client.mkdir(new_subfolder_path)
                        self.logger.info(f'Created new superbatch subdirectory: {new_subfolder_path}')
                
                # add the superbatch to the status
                with self.update_remote_status():
                    if superbatch not in self.status['superbatch_order']:
                        self.status['superbatch_order'].append(superbatch)
                    if superbatch not in self.status[variant]['epochs'][str(epoch)]['superbatches']:
                        self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch] = {
                            'transferred': False,
                            'preprocessed': False,
                            'features_extracted': (
                                False if self.config['feature_extraction'] 
                                else 'not applicable'
                            ),
                            'deleted': False,
                            'transfer_index_range': [index, index],
                            'locked_preprocessing_indices': {},
                            'locked_extraction_indices': {},
                            'finished_preprocessing_indices': [],
                            'finished_extraction_indices': [],
                            'accumulated_size': 0, # GB
                            'current_size': 0, # GB 
                        }

                # start file transfer
                continue_transfer = True
                while continue_transfer:
                    # check if the end of the epoch was reached
                    if index == len(self.data[variant]):
                        continue_transfer = False
                        with self.update_remote_status():
                            self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['transferred'] = True
                        epoch += 1
                        index = 0
                        if epoch != self.config['variants'][variant]['epochs']:
                            self.logger.info((
                                f'Continuing with next epoch: [{variant}] epoch '
                                f'{epoch}, superbatch {batch}, index {index}'
                            ))
                    else:
                        # get the path(s)
                        paths = self.data[variant][order[index]]['paths']
                        paths = [paths] if isinstance(paths, str) else paths                    
                        # find total size for all files
                        size = sum([os.path.getsize(path) for path in paths])/GIGABYTE

                        # check if the file transfer should be continued for this superbatch
                        if self.config['variants'][variant]['max_size'] is not None:
                            if size > self.config['variants'][variant]['max_size']:
                                raise ValueError((
                                    'Size of all files that belong together exceed the maximum size of the superbatch.'
                                ))
                            elif (self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['accumulated_size'] + size 
                                  > self.config['variants'][variant]['max_size']):
                                continue_transfer = False
                                with self.update_remote_status():
                                    self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['transferred'] = True

                    if continue_transfer:
                        self.load_status()
                        # if the total capacity of the remote directory has been reached,
                        # check repeatedly if space has become available
                        if self.status['current_size_remote_dir']+size > self.config['max_size_remote_dir']:
                            # log maximum storage size reached once
                            if not notify_max_size_reached:
                                max_size = self.config['max_size_remote_dir']
                                self.logger.info(f'Total storage space ({max_size} GB) reached')
                                notify_max_size_reached = True
                            # retry to continue the transfer after a certain amount of time
                            self.logger.info(f'Retry transfer after {self.pause_duration} seconds')
                            time.sleep(self.pause_duration)
                        
                        else:
                            # reset state
                            if notify_max_size_reached:
                                notify_max_size_reached = False
                            # initialize lists to store paths
                            first_paths = []
                            second_paths = []
                            for path in paths:
                                # define specific paths
                                filename = Path(path).name
                                first_path = self.remote_dir/superbatch/self.subfolders[0]/filename
                                second_path = self.remote_dir/superbatch/self.subfolders[1]/filename

                                # store first_path and second_path
                                first_paths.append(first_path)
                                second_paths.append(second_path)

                                # remove file if it is already present in the remote location
                                filenames = self.sftp_client.listdir(self.get_path(first_path.parent))
                                if str(first_path.name) in filenames:
                                    self.sftp_client.remove(self.get_path(first_path))
                                    self.logger.info((
                                        'Removed already existing file (possibly '
                                        f'corrupted): {first_path}'
                                    ))
                                # copy file to remote location                       
                                self.copy_to_remote(path, self.get_path(first_path))
                                self.logger.info(f'Transferred file to: {first_path}')

                            # update all status information 
                            index += 1
                            with self.update_remote_status():
                                self.status['accumulated_size_remote_dir'] += size
                                self.status['current_size_remote_dir'] += size
                                self.status[variant]['epochs'][str(epoch)]['transfer_index'] = index
                                self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['accumulated_size'] += size
                                self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['current_size'] += size
                                self.status[variant]['epochs'][str(epoch)]['superbatches'][superbatch]['transfer_index_range'][1] += 1

                            # loop over the transferred files
                            for first_path, second_path in zip(first_paths, second_paths):
                                # remove file if it is already present in the remote location
                                filenames = self.sftp_client.listdir(self.get_path(second_path.parent))
                                if str(second_path.name) in filenames:
                                    self.sftp_client.remove(self.get_path(second_path))
                                    self.logger.info(f'Removed already existing file: {second_path}')
                                # move the file from the first to the second subfolder
                                self.sftp_client.rename(self.get_path(first_path), self.get_path(second_path))
                                self.logger.info(f'Moved file to: {second_path}')
                            
                            if index != len(self.data[variant]):
                                self.logger.info((
                                    f'Continuing with next index: [{variant}] epoch '
                                    f'{epoch}, superbatch {batch}, index {index}'
                                ))
                batch += 1
                if epoch != self.config['variants'][variant]['epochs']:
                    self.logger.info((
                        f'Continuing with next superbatch: [{variant}] epoch '
                        f'{epoch}, superbatch {batch}, index {index}'
                    ))
                else:
                    self.logger.info(('Data transfer completed for superbatch '
                                     f'type: {variant}'))
            
            # update the status to signal that the transfer has completed for the variant
            with self.update_remote_status():
                self.status[variant]['transfer_completed'] = True

        # update the status to signal that the transfer has completed
        with self.update_remote_status():
            self.status['transfer_completed'] = True
        self.logger.info('Transfer completed')


# define helper functions
def progress(filename: str, size: str, sent: str) -> None:
    sys.stdout.write(
        f"{filename}'s progress: {float(sent)/float(size)*100:0.1f}%    \r"
    )


def get_ssh_client(host: str, username: str, password: str) -> None:
    """ 
    Create ssh client and connect.
    """
    ssh_client = SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, username=username, password=password)

    return ssh_client
