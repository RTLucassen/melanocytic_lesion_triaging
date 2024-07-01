"""
Implementation of whole slide image (WSI) feature extraction service.
"""

import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Callable, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
from natsort import natsorted
from skimage.transform import rescale
from torch.utils.data import DataLoader, SequentialSampler

from pipeline.feature_extraction_utils import (
    TileDataset, 
    read_tile_information, 
    custom_collate_fn,
    get_augmentation_transform,
)
from pipeline.base_service import BaseService


GIGABYTE = (1024**3)

class FeatureExtractionService(BaseService):

    max_specimen_per_iter = 20
    max_lock_time = 15*60 # seconds
    time_format = '%d/%m/%y %H:%M:%S.%f'
    augmentation_examples = 5
    downscale_factor_examples = 16

    def __init__(
        self, 
        directory: Union[str, Path], 
        config_file: str,
        extraction_func: Callable,
        device: str = 'cpu',
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        Initialize service for feature extraction.

        Args:
            directory:  Directory where all superbatches are stored.
            config_file:  Name of config file.
            extraction_func:  Function for training a model.
            device:  Name of device for feature extraction model inference.
            num_workers:  Number of workers for the dataloader.
            pin_memory:  Indicates whether pinned memory is used for the dataloader.
        """
        super().__init__(directory, config_file)

        # initialize instance attributes
        self.extraction_func = extraction_func
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
        # define settings attribute
        self.settings = self.config['feature_extraction_settings']
                
    def start(self):
        """ 
        Start the feature extraction for the preprocessed images in the superbatches.
        """
        # check if feature extraction was enabled in the config
        if not self.config['feature_extraction']:
            raise ValueError('Feature extraction was not enabled in the config file.')

        # start the feature extraction loop
        continue_extraction = True
        while continue_extraction:
            # update order and status variables
            self.load_order()
            self.load_status()
            extraction_count = 0
            superbatch_count = 0
            pause_extraction = True
            # loop over superbatches variants
            for variant in self.config['variants']:
                # check if the variant should be skipped
                if self.config['variants'][variant]['skip']:
                    continue
                # check if the variant is in the status and order
                elif (variant not in self.status) or (variant not in self.order):
                    continue
                variant_completed = True
                # loop over all superbatches
                for epoch in list(self.status[variant]['epochs'].keys()):
                    # check if epoch is in the variant status and order
                    if ((epoch not in self.status[variant]['epochs']) 
                        or (epoch not in self.order[variant])):
                        continue
                    for superbatch in list(self.status[variant]['epochs'][epoch]['superbatches'].keys()):
                        # check if the maximum count has been reached for one 
                        # feature extraction iteration (i.e., outermost loop)
                        # to force more frequent status updates
                        if self.max_specimen_per_iter is not None:
                            if extraction_count >= self.max_specimen_per_iter:
                                continue

                        # get the superbatch path and status
                        superbatch_path = self.directory / superbatch
                        superbatch_status = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]
                        # skip the extraction if the features have already 
                        # been extracted or if the data has already been 
                        # deleted for the superbatch
                        if superbatch_status['features_extracted'] == 'not applicable':
                            raise ValueError('Feature extraction was not enabled in the config file.')
                        elif (superbatch_status['deleted'] or (superbatch_status['features_extracted'] == True)):
                            superbatch_count += 1
                            continue
                        
                        variant_completed = False
                        self.logger.info(f'Start feature extraction for superbatch: {superbatch} (epoch {epoch})')
                        
                        # check if locked indices have exceeded the maximum lock time
                        with self.update_local_status():
                            locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices']
                            indices_to_delete = []
                            for locked_index, (_, lock_time) in locked_indices.items():
                                lock_time = datetime.strptime(lock_time, self.time_format)
                                if (datetime.now()-lock_time) > timedelta(seconds=self.max_lock_time):
                                    indices_to_delete.append(locked_index)
                            # delete indices from list of locked indices
                            for locked_index in indices_to_delete:
                                self.logger.info(f'Specimen unlocked: {locked_indices[locked_index]}')
                                del locked_indices[locked_index]

                        # create the feature information file if it does not exist 
                        feature_information_path = superbatch_path / self.settings['output_filename']
                        if not feature_information_path.exists():
                            with open(feature_information_path, 'w') as f:
                                f.write('')
                            self.logger.info(f'Created file: {feature_information_path}')

                        # get all specimens (collections of files) in the superbatch
                        order = self.order[variant][epoch]
                        start, end = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['transfer_index_range']
                        indices = order[start:end]
                        superbatch_paths = [self.data[variant][i]['paths'] for i in indices]
                        superbatch_filenames = []
                        for paths in superbatch_paths:
                            if isinstance(paths, str):
                                paths = [paths]
                            if self.config['local_OS'].lower() in ['windows', 'win32', 'win']:
                                superbatch_filenames.append([
                                    PureWindowsPath(path).name for path in paths
                                ])
                            else:
                                superbatch_filenames.append([
                                    PurePath(path).name for path in paths
                                ])

                        # define paths
                        first_path = superbatch_path / self.subfolders[2]
                        second_path = superbatch_path / self.optional_folders[2]
                        
                        # check if the preprocessing has finished for the current superbatch
                        # and no more files are locked or waiting for feature extraction
                        if (self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['preprocessed']
                            and not len(self.status[variant]['epochs'][epoch]['superbatches'][superbatch]["locked_extraction_indices"])
                            and not len(os.listdir(first_path))):
                            # update the status of the superbatch
                            with self.update_local_status():
                                self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['features_extracted'] = True
                            self.logger.info(f'Finished extracting features for superbatch: {superbatch} (epoch {epoch})')

                        else:
                            # select all specimens for which at least one file is present
                            selection = []
                            # loop over all filenames for files that are waiting
                            for filename in os.listdir(first_path):
                                # loop over all specimen of the superbatch
                                for index, specimen_filenames in zip(indices, superbatch_filenames):
                                    if filename in specimen_filenames:
                                        selected = (index, specimen_filenames)
                                        if selected not in selection:
                                            selection.append(selected)

                            # check if all files that belong together as specimen
                            # have been preprocessed
                            for index, specimen_filenames in selection:
                                # check if the maximum count has been reached for one 
                                # feature extraction iteration (i.e., outermost loop)
                                # to force more frequent status updates
                                if self.max_specimen_per_iter is not None:
                                    if extraction_count >= self.max_specimen_per_iter:
                                        break
                                
                                # get all filenames (mostly relevant to repeat each iteration when 
                                # using multiple nodes performing the feature extraction in parallel)
                                available_filenames = os.listdir(first_path)
                                # determine if all filenames for a specimen are present
                                complete = True
                                missing = 0
                                for filename in specimen_filenames:
                                    if filename not in available_filenames:
                                        complete = False
                                        missing += 1
                                
                                # remove all files for specimens with absent files after preprocessing has completed 
                                self.load_status()
                                if (str(index) not in self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices']
                                    and self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['preprocessed'] 
                                    and not complete and missing < len(specimen_filenames)):
                                    # delete files and calculate the emptied storage space
                                    # NOTE: the missing file is not accounted for
                                    space_emptied = 0
                                    for filename in specimen_filenames:
                                        path = first_path / filename
                                        if path.exists():
                                            space_emptied += os.path.getsize(path)
                                            os.remove(path)
                                            self.logger.info((f'Deleted file: {path} (all or in part missing after transfer)'))
                                    space_emptied /= GIGABYTE
                                    
                                    # update the occupied storage space in the status file
                                    with self.update_local_status():
                                        old_size = self.status["current_size_remote_dir"]
                                        self.status['current_size_remote_dir'] -= space_emptied
                                        self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['current_size'] -= space_emptied
                                    self.logger.info((f'The occupied storage space decreased from {old_size:0.2f}GB '
                                                      f'to {self.status["current_size_remote_dir"]:0.2f}GB'))
                                                                    
                                # continue the feature extraction if all specimen files are present        
                                elif complete:
                                    # check if the specimen index is locked (mostly relevant when using 
                                    # multiple nodes performing the feature extraction in parallel)
                                    with self.update_local_status():
                                        locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices']
                                        finished_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['finished_extraction_indices']
                                        if str(index) in locked_indices:
                                            continue
                                        elif str(index) in finished_indices:
                                            continue
                                        else:
                                            current_time = datetime.now().strftime(self.time_format)
                                            locked_indices[str(index)] = (self.ID, current_time)  
                                            self.logger.info(f'Specimen locked: {locked_indices[str(index)]}')

                                    extraction_count += 1
                                    pause_extraction = False
                                    # get the specimen information
                                    specimen_information = deepcopy(self.data[variant][index])
                                    del specimen_information['paths']
                                    
                                    # separate the slides from the other files based on the suffix
                                    slide_filenames = []
                                    other_filenames = []
                                    suffixes = self.settings['suffixes_for_feature_extraction']
                                    for filename in natsorted(specimen_filenames):
                                        if any([suffix in filename for suffix in suffixes]):
                                            slide_filenames.append(filename)
                                        else:
                                            other_filenames.append(filename)

                                    # read the tile information
                                    specimen_index = specimen_information["specimen_index"]
                                    tile_information, _ = read_tile_information(
                                        tile_information_path=superbatch_path/self.config['preprocessing_settings']['output_filename'], 
                                        image_directory=first_path,
                                    )
                                    
                                    self.logger.info(f'Start feature extraction: {slide_filenames}') 
                                    extra_space_occupied = 0
                                    # get the features
                                    try:
                                        feature_variants = self._extract_features(
                                            tile_information={specimen_index: tile_information[specimen_index]},
                                            apply_augmentation=(variant in self.settings['augmentation_config']['superbatch_variants']),
                                            visualization_path=superbatch_path / self.optional_folders[1],
                                            callback=lambda: self.update_lock_time(variant, epoch, superbatch, index), 
                                        )
                                    except Exception as error:
                                        self.logger.info(f'Feature extraction aborted due to {type(error).__name__}:\n{error}')
                                    else:
                                        # skip saving the results and moving the files
                                        # if the ID of the locked index does not match
                                        self.load_status()  
                                        locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices']
                                        if str(index) not in locked_indices:
                                            continue
                                        elif locked_indices[str(index)][0] != self.ID:
                                            continue
                                        
                                        # save the extracted features as a single or separate files
                                        for j, features in feature_variants.items():
                                            feature_filenames = []
                                            if self.settings['save_as_one_file']:
                                                feature_filename = f'{specimen_information["specimen_index"]}.pth'
                                                if len(feature_variants) > 1:
                                                    feature_filename = feature_filename.replace('.pth', f'_v{j}.pth')
                                                feature_filenames.append(feature_filename)
                                                feature_path = second_path / feature_filenames[-1]
                                                torch.save(list(features.values()), feature_path)
                                                extra_space_occupied += os.path.getsize(feature_path)
                                                self.logger.info(f'Saved {len(features)} extracted features as one file to: {feature_path}') 
                                            else:
                                                for feature_index, feature in features.items():
                                                    feature_filename = f'{"_".join([str(i) for i in feature_index])}.pth'
                                                    if len(feature_variants) > 1:
                                                        feature_filename = feature_filename.replace('.pth', f'_v{j}.pth')
                                                    feature_filenames.append(feature_filename)
                                                    feature_path = second_path / feature_filenames[-1]
                                                    torch.save(feature, feature_path)
                                                    extra_space_occupied += os.path.getsize(feature_path)
                                                self.logger.info(f'Saved {len(features)} extracted features in the directory {second_path} '
                                                                f'with the following filenames: {feature_filenames}') 
                                        
                                            # save feature information
                                            with open(feature_information_path, 'a') as f:
                                                f.write(f'{slide_filenames}\n')
                                                f.write(f'{json.dumps(specimen_information)}\n')
                                                f.write(f'{json.dumps(feature_filenames)}\n')
                                        
                                        if len(feature_variants) == 1:
                                            self.logger.info('Saved feature information')
                                        else:
                                            self.logger.info(f'Saved feature information ({len(feature_variants)} variants)')
                            
                                    # remove the whole slide images for which features have been extracted
                                    space_emptied = 0
                                    for filename in slide_filenames:
                                        slide_path = first_path / filename
                                        if slide_path.exists():
                                            space_emptied += os.path.getsize(slide_path)
                                            os.remove(slide_path)
                                            self.logger.info((f'Deleted file: {slide_path}'))

                                    # update the status after deleting the data
                                    space_difference = (space_emptied - extra_space_occupied)/GIGABYTE
                                    with self.update_local_status():
                                        old_size = self.status["current_size_remote_dir"]
                                        self.status['current_size_remote_dir'] -= space_difference
                                        self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['current_size'] -= space_difference
                                    self.logger.info((f'The occupied storage space decreased from {old_size:0.2f}GB '
                                                      f'to {self.status["current_size_remote_dir"]:0.2f}GB'))
                                    
                                    # unlock specimen
                                    with self.update_local_status():
                                        self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['finished_extraction_indices'].append(str(index))
                                        locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices']
                                        if str(index) in locked_indices:
                                            self.logger.info(f'Specimen unlocked: {locked_indices[str(index)]}')
                                            del locked_indices[str(index)]

                # update the status if for all superbatches from a variant the features have been extracted
                if (variant_completed and self.status[variant]['preprocessing_completed']
                    and not self.status[variant]['feature_extraction_completed']):
                    with self.update_local_status():
                        self.status[variant]['feature_extraction_completed'] = True

            # check if the transfer has completed and all superbatches have been preprocessed
            if (superbatch_count == len(self.status['superbatch_order'])
                and self.status['preprocessing_completed']):
                continue_extraction = False
            # check if pausing is necessary
            if continue_extraction and pause_extraction:
                self.logger.info('No files to extract features from at the moment')
                self.logger.info(f'Retry feature extraction after {self.pause_duration} seconds')
                time.sleep(self.pause_duration)

        # update the status to signal that the feature extraction has completed
        with self.update_local_status():
            self.status['feature_extraction_completed'] = True
        self.logger.info('feature extraction completed')      

    def update_lock_time(self, variant: str, epoch: int, superbatch: str, index: int) -> None:
        """
        Update the lock time for a specimen in the local status.

        Args:
            variant:  Dataset variant (training, validation, test).
            epoch:  Epoch value.
            superbatch:  Superbatch name.
            index:  Specimen index value.
        """  
        raise_error = False
        with self.update_local_status():
            locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_extraction_indices'] 
            if str(index) not in locked_indices:
                raise ValueError('Locked index was not found.')
            elif locked_indices[str(index)][0] != self.ID:
                raise_error = True
            else:
                current_time = datetime.now().strftime(self.time_format)
                locked_indices[str(index)] = (self.ID, current_time)                
        
        if raise_error:
            raise ValueError('ID does not match for locked index.')

    def _extract_features(self, 
        tile_information: dict[int, dict[str, Any]],
        apply_augmentation: bool,
        visualization_path: Path,
        callback: Optional[Callable] = None,
    ) -> dict[tuple[int, int, int, int], dict[str, Any]]:
        """
        Performs feature extraction on the preprocessed whole slide images.
        
        Args:
            tile_information:  Dictionary with specimen index as key and as value
                a dictionary with the tile information.
            apply_augmentation:  Indicates whether augmentation should be applied.
            visualization_path:  Path to save augmentation visualization images.
            callback:  Callback function.

        Returns:
            features:  Dictionary with features. 
        """
        # check if any augmentation configuration is provided
        if 'augmentation_config' in self.settings and apply_augmentation:
            augmentation_config = self.settings['augmentation_config']
            
            # define the number of times a data augmentation transform is sampled
            if 'feature_variants' not in augmentation_config:
                feature_variants = 1
            elif not isinstance(augmentation_config['feature_variants'], int):
                raise ValueError('The value corresponding to `feature_variants` must be an integer.')
            elif augmentation_config['feature_variants'] < 1:
                raise ValueError('The value corresponding to `feature_variants` must be larger than 0.')
            else:
                feature_variants = augmentation_config['feature_variants']

            # initialize the image augmentation transform and level (i.e., whether 
            # the same augmentation is applied to all tiles from the same specimen, 
            # slide, cross-section, or randomly per tile)
            if 'transform' in augmentation_config:
                transform = get_augmentation_transform(augmentation_config['transform'])
                if 'level' in augmentation_config:
                    level = augmentation_config['level']
                    if level not in ['specimen', 'slide', 'cross-section', 'tile']:
                        raise ValueError('The value corresponding to `level` must be either '
                                         'specimen, slide, cross-section, or tile.')                      
                else:
                    raise ValueError('The value corresponding to `level` must be specified '
                                     'if an augmentation transform function is provided.')
            else:
                transform = None
                level = None
        else:
            feature_variants = 1
            transform = None
            level = None

        # initialize a dataset object for loading tiles
        dataset = TileDataset(
            tile_information=tile_information,
            magnification=self.config['preprocessing_settings']['extraction_magnification'],
            variants=feature_variants,
            transform=transform,
            first_not_augmented=self.settings['augmentation_config']['first_not_augmented'],
            level=level,
            seed=self.config['seed'],
            logger=self.logger,
        )

        # save augmentation examples
        if 'augmentation_config' in self.settings and apply_augmentation:
            if self.settings['augmentation_config']['save_augmentation_visualizations']:
                # loop over specified number of randomly selected tiles with augmentations
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                N_examples = min(self.augmentation_examples, len(indices))
                for index in indices[:N_examples]:
                    # get augmented tiles
                    tile_index, _, tile_variants = dataset.__getitem__(index)

                    # concatenate augmented tiles and rescale them
                    tile_variants = torch.concat(tile_variants, dim=2).moveaxis(0, -1)
                    tile_variants = rescale(
                        image=tile_variants.numpy(), 
                        scale=1/self.downscale_factor_examples, 
                        channel_axis=-1,
                    )
                    tile_variants = (tile_variants*255).astype(np.uint8)

                    # define filename for example augmentation image
                    filename = '_'.join([str(i) for i in tile_index.tolist()])+'.png'
                    sitk.WriteImage(
                        sitk.GetImageFromArray(tile_variants[None, ...]), 
                        visualization_path / filename,
                    )
 
        # initialize dataloader
        tile_dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=SequentialSampler(dataset),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
        )
        features = self.extraction_func(
            tile_dataloader=tile_dataloader, 
            configs=self.settings['extraction_config'],
            device=self.device,
            callback=callback,
        )
        
        return features        