"""
Implementation of whole slide image (WSI) preprocessing service.
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from copy import deepcopy
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path, PurePath, PureWindowsPath
from typing import Callable, Optional, Union

import matplotlib
matplotlib.use('Agg')
import torch
from natsort import natsorted
from slideloader import SlideLoader
from slidesegmenter import SlideSegmenter

from pipeline.base_service import BaseService
from pipeline.preprocessing_utils import (
    anonymize, 
    reformat_dicom, 
    tessellate,
    visualize_tessellation,
    combine,
)

GIGABYTE = (1024**3)

class PreprocessingService(BaseService):

    max_specimen_per_iter = 20 # specimens
    max_difference = 0.1 # magnification
    max_lock_time = 15*60 # seconds
    time_format = '%d/%m/%y %H:%M:%S.%f'

    def __init__(
        self, 
        directory: Union[str, Path], 
        config_file: str,
        unique_filenames_func: Optional[Callable] = None,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize preprocessing service.

        Args:
            directory:  Directory where all superbatches are stored.
            config_file:  Name of config file.
            unique_filenames_func:  Function that returns the unique filenames.
            device:  Name of device for segmentation model inference.
        """
        super().__init__(directory, config_file)

        # define function for getting unique filenames
        self.unique_filenames_func = unique_filenames_func

        # define attributes for preprocessing hyperparameters and settings
        self.settings = self.config['preprocessing_settings']
        self.device = device

        # initialize the SlideLoader and SlideSegmenter instance
        self.loader = SlideLoader({'max_difference': 0.49/4096})
        self.segmenter = SlideSegmenter(
            channels_last = True,
            tissue_segmentation = True,
            pen_marking_segmentation = self.settings['exclude_pen_markings'],
            separate_cross_sections = True,
            model_folder = '2023-08-13',
            device = self.device,
        )

    def start(self):
        """ 
        Start preprocessing all incoming files.
        """
        # start preprocessing loop
        continue_preprocessing = True
        while continue_preprocessing:
            # update order and status variables and initialize state variables
            self.load_order()
            self.load_status()
            preprocess_count = 0
            superbatch_count = 0
            pause_preprocessing = True
            # loop over superbatch variants
            for variant in self.config['variants']:
                # check if the variant should be skipped
                if self.config['variants'][variant]['skip']:
                    continue
                # check if the variant is in the status and order
                elif (variant not in self.status) or (variant not in self.order):
                    continue
                variant_completed = True
                # loop over all epochs
                for epoch in list(self.status[variant]['epochs'].keys()):
                    # check if epoch is in the variant status and order
                    if ((epoch not in self.status[variant]['epochs']) 
                        or (epoch not in self.order[variant])):
                        continue
                    # loop over superbatches
                    for superbatch in list(self.status[variant]['epochs'][epoch]['superbatches'].keys()):
                        # check if the maximum count has been reached for one 
                        # preprocessing iteration (i.e., outermost loop)
                        # to force more frequent status updates
                        if self.max_specimen_per_iter is not None:
                            if preprocess_count >= self.max_specimen_per_iter:
                                continue

                        # get the superbatch path and status
                        superbatch_path = self.directory / superbatch
                        superbatch_status = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]
                        # skip preprocessing if the images have already been 
                        # preprocessed, if features have already been extracted, 
                        # of if the data has already been deleted for the superbatch
                        if (superbatch_status['deleted'] or superbatch_status['features_extracted'] == True
                            or superbatch_status['preprocessed']):
                            superbatch_count += 1
                            continue
                        
                        variant_completed = False
                        self.logger.info(f'Start preprocessing superbatch: {superbatch} (epoch {epoch})')
                        
                        # check if locked indices have exceeded the maximum lock time
                        with self.update_local_status():
                            locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                            indices_to_delete = []
                            for locked_index, (_, lock_time) in locked_indices.items():
                                lock_time = datetime.strptime(lock_time, self.time_format)
                                if (datetime.now()-lock_time) > timedelta(seconds=self.max_lock_time):
                                    indices_to_delete.append(locked_index)
                            # delete indices from list of locked indices
                            for locked_index in indices_to_delete:
                                self.logger.info(f'Specimen unlocked: {locked_indices[locked_index]} {locked_indices}')
                                del locked_indices[locked_index]

                        # create the tile information file if it does not exist 
                        tile_information_path = superbatch_path / self.settings['output_filename']
                        if not tile_information_path.exists():
                            with open(tile_information_path, 'w') as f:
                                f.write('')
                            self.logger.info(f'Created file: {tile_information_path}')

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
                        first_path = superbatch_path / self.subfolders[1]
                        second_path = superbatch_path / self.subfolders[2]

                        # check if the transfer has finished for the current superbatch
                        # and no more files are locked or waiting to be preprocessed
                        if (self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['transferred']
                            and not len(self.status[variant]['epochs'][epoch]['superbatches'][superbatch]["locked_preprocessing_indices"])
                            and not len(os.listdir(first_path))):
                            # update the status of the superbatch
                            with self.update_local_status():
                                if not self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['preprocessed']:
                                    self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['preprocessed'] = True
                            self.logger.info(f'Finished preprocessing superbatch: {superbatch} (epoch {epoch})')

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
                            # have been transferred
                            for index, specimen_filenames in selection:
                                # check if the maximum count has been reached for one 
                                # preprocessing iteration (i.e., outermost loop)
                                # to force more frequent status updates
                                if self.max_specimen_per_iter is not None:
                                    if preprocess_count >= self.max_specimen_per_iter:
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
                                
                                # remove all files for specimens with absent files after the transfer has completed 
                                self.load_status()
                                if (str(index) not in self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                                    and self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['transferred'] 
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
                                
                                # continue preprocessing if all specimen files are present        
                                elif complete:
                                    # check if the specimen index is locked (mostly relevant when using 
                                    # multiple nodes performing the feature extraction in parallel)
                                    with self.update_local_status():
                                        locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                                        finished_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['finished_preprocessing_indices']
                                        if str(index) in locked_indices:
                                            continue
                                        elif str(index) in finished_indices:
                                            continue
                                        else:
                                            current_time = datetime.now().strftime(self.time_format)
                                            locked_indices[str(index)] = (self.ID, current_time)
                                            self.logger.info(f'Specimen locked: {locked_indices[str(index)]} {locked_indices}')
                                    
                                    preprocess_count += 1
                                    pause_preprocessing = False
                                    # get the specimen information
                                    specimen_information = deepcopy(self.data[variant][index])
                                    del specimen_information['paths']
                                    
                                    # get the size of all specimen files before preprocessing
                                    unpreprocessed_size = 0
                                    for filename in specimen_filenames:
                                        path = first_path / filename
                                        if path.exists():
                                            unpreprocessed_size += os.path.getsize(path)
                                    unpreprocessed_size /= GIGABYTE
                                        
                                    # separate the slides from the other files based on the suffix
                                    slide_filenames = []
                                    other_filenames = []
                                    suffixes = self.settings['suffixes_for_preprocessing']
                                    for filename in specimen_filenames:
                                        if any([suffix in filename for suffix in suffixes]):
                                            slide_filenames.append(filename)
                                        else:
                                            other_filenames.append(filename)
                                    
                                    # get the unique filenames
                                    if self.unique_filenames_func is None:
                                        slide_names = {slide.split('.')[0] for slide in slide_filenames}
                                    else:
                                        slide_names = self.unique_filenames_func(slide_filenames)
                                    
                                    # group slides with the same filename
                                    # (e.g., levels of DICOM slides can be stored as separate files)
                                    slides = {}
                                    for name in slide_names:
                                        slides[name] = [s for s in slide_filenames if name in s]

                                    # loop over slides for preprocessing
                                    aborted = False
                                    combined_tile_information = []
                                    for name, filenames in slides.items():                                                                             
                                        # get the paths
                                        slide_paths = natsorted([first_path / filename for filename in filenames])
                                        visualization_path = superbatch_path / self.optional_folders[0] / f'{name}.png'

                                        # preprocess the slide
                                        try:
                                            tile_information = self._slide_preprocessing(slide_paths, visualization_path)
                                        except Exception as error:
                                            self.logger.info(f'Slide preprocessing aborted due to {type(error).__name__}:\n{error}')
                                            aborted = True
                                            if self.settings['skip_all_if_aborted']:
                                                break
                                        else:
                                            # store the tile information
                                            combined_tile_information.extend([
                                                f'{filenames}\n',
                                                f'{json.dumps(specimen_information)}\n',
                                                f'{json.dumps(tile_information)}\n',
                                            ])
                                        
                                        # update lock time
                                        with self.update_local_status():
                                            locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                                            if str(index) not in locked_indices:
                                                break
                                            elif locked_indices[str(index)][0] != self.ID:
                                                break
                                            else:
                                                current_time = datetime.now().strftime(self.time_format)
                                                locked_indices[str(index)] = (self.ID, current_time)
                                        
                                    # skip saving the results and moving the files
                                    # if the ID of the locked index does not match
                                    self.load_status()  
                                    locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                                    if str(index) not in locked_indices:
                                        continue
                                    elif locked_indices[str(index)][0] != self.ID:
                                        continue
                                    elif aborted and self.settings['skip_all_if_aborted']:
                                        self.logger.info('WSI tile information was not saved because preprocessing was aborted.')
                                        # delete files and calculate the emptied storage space
                                        for filename in specimen_filenames:
                                            path = first_path / filename
                                            if path.exists():
                                                os.remove(path)
                                                self.logger.info((f'Deleted file: {path} (because of preprocessing error)'))
                                        
                                        # update the occupied storage space in the status file
                                        if unpreprocessed_size > 0:
                                            with self.update_local_status():
                                                old_size = self.status["current_size_remote_dir"]
                                                self.status['current_size_remote_dir'] -= unpreprocessed_size
                                                self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['current_size'] -= unpreprocessed_size
                                            self.logger.info((f'The occupied storage space decreased from {old_size:0.2f}GB '
                                                            f'to {self.status["current_size_remote_dir"]:0.2f}GB'))
                                    else:
                                        # save the tile information
                                        with open(tile_information_path, 'a') as f:
                                            for line in combined_tile_information:
                                                f.write(line)
                                        self.logger.info(f'Saved WSI tile information')
                                        
                                        # move the files to the next subfolder and determine the size after preprocessing
                                        preprocessed_size = 0
                                        for filename in specimen_filenames:
                                            if (first_path / filename).exists():
                                                preprocessed_size += os.path.getsize(first_path / filename)
                                                os.rename(first_path / filename, second_path / filename)
                                                self.logger.info(f'Moved file to: {second_path / filename}') 
                                        preprocessed_size /= GIGABYTE 

                                        # update the occupied storage space in the status file
                                        size_difference = unpreprocessed_size-preprocessed_size
                                        if size_difference != 0:
                                            with self.update_local_status():
                                                old_size = self.status["current_size_remote_dir"]
                                                self.status['current_size_remote_dir'] -= size_difference
                                                self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['current_size'] -= size_difference
                                            self.logger.info((f'The occupied storage space changed from {old_size:0.2f}GB '
                                                            f'to {self.status["current_size_remote_dir"]:0.2f}GB'))

                                    # unlock specimen
                                    with self.update_local_status():
                                        self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['finished_preprocessing_indices'].append(str(index))
                                        locked_indices = self.status[variant]['epochs'][epoch]['superbatches'][superbatch]['locked_preprocessing_indices']
                                        if str(index) in locked_indices:
                                            self.logger.info(f'Specimen unlocked: {locked_indices[str(index)]} {locked_indices}')
                                            del locked_indices[str(index)]
                                        
                # update the status if all superbatches from a variant have been preprocessed
                self.load_status()
                if (variant_completed and self.status[variant]['transfer_completed'] 
                    and not self.status[variant]['preprocessing_completed']):
                    with self.update_local_status():
                        self.status[variant]['preprocessing_completed'] = True

            # check if the transfer has completed and all superbatches have been preprocessed
            self.load_status()
            if (superbatch_count == len(self.status['superbatch_order'])
                and self.status['transfer_completed']):
                continue_preprocessing = False
            # check if pausing is necessary
            elif continue_preprocessing and pause_preprocessing:
                self.logger.info('No files to preprocess at the moment')
                self.logger.info(f'Retry preprocessing after {self.pause_duration} seconds')
                time.sleep(self.pause_duration)

        # update the status to signal that the preprocessing has completed
        with self.update_local_status():
            self.status['preprocessing_completed'] = True
        self.logger.info('Preprocessing completed')                            

    def _slide_preprocessing(self, slide_paths: list[Path], visualization_path: Path,
    ) -> dict[int,list[tuple[tuple[int,int], tuple[int,int]]]]:
        """
        Performs slide preprocessing, which includes anonymization, segmentation
        of tissue cross-sections and pen markings, tessellation, and optionally
        the visualization of the tiles on top of the image and segmentation.
        
        Args:
            slide_paths:  Paths to all slides that belong to a single specimen.
            visualization_path:  Path to save tesselation visualization image.
        
        Returns:
            tile_information:  Dictionary with a list per channel of tuples with 
                the tile locations (top left corner) and shapes.
        """
        # get the size of all whole slide image files
        size = sum([os.path.getsize(path) for path in slide_paths])/GIGABYTE
        filenames = ', '.join([path.name for path in slide_paths])
        self.logger.info(f'Start preprocessing: {filenames} ({size:0.2f}GB)')                                        

        # (1) reformat DICOM files 
        is_dicom = False
        for slide_path in slide_paths:
            if slide_path.suffix == '.dcm':
                reformat_dicom(slide_path)
                is_dicom = True
        
        if is_dicom:
            self.logger.info('Finished reformatting DICOM WSI')
           
        # (2) anonymize the slide
        for slide_path in slide_paths:
            try:
                os.chmod(slide_path, 0o644)
                anonymize(slide_path)
            except Exception as error:
                self.logger.info(
                    f'Anonymization of {slide_path.name} was unsuccessful:\n{error}'
                )
        self.logger.info(f'Finished anonymizing WSI')
        
        # (3) load the slide and determine the exact magnification for reading 
        # the image at a low magnification
        self.loader.load_slide(slide_paths)
        magnification_levels = self.loader.get_properties()['magnification_levels']
        if (max(magnification_levels)+self.max_difference 
            < self.settings['extraction_magnification']):
            raise ValueError('Maximum magnification available is smaller than '
                             'the specified extraction magnification.')
        else:
            ratio = (self.settings['extraction_magnification'] 
                     / self.settings['segmentation_magnification'])
            magnification = None
            for level in magnification_levels:
                if abs(self.settings['extraction_magnification']-level) < self.max_difference:
                    magnification = level / ratio
            if magnification is None:
                magnification = self.settings['segmentation_magnification']

        # (4) read the image and crop it if necessary   
        image = self.loader.get_image(
            magnification=magnification,
        )
        self.loader.close()
        self.logger.info(
            f'Finished loading image ({magnification:0.2f}x)'
        )
        # get the height and width of the image at the segmentation and 
        # feature extraction magnification
        height, width, channels = image.shape
        true_height, true_width = self.loader.get_dimensions(magnification*ratio)
        # calculate the number of pixels to crop at the bottom and right of
        # the segmentation image to prevent issues when loading tiles at a
        # higher magnification
        crop_height = ceil(max(0, height*ratio-true_height)/ratio)
        crop_width = ceil(max(0, width*ratio-true_width)/ratio)
        if crop_height > 0 or crop_width > 0:
            image = image[:height-crop_height, :width-crop_width, ...]
            self.logger.info(f'Cropped the image from {(height, width, channels)} to {image.shape}')

        # (5) segment the tissue (and pen markings) regions on the image
        if self.settings['exclude_pen_markings']:
            try:
                tissue, pen_marking = self.segmenter.segment(
                    image=image/255, 
                    tissue_threshold=self.settings['tissue_threshold'],
                    pen_marking_threshold=self.settings['pen_marking_threshold'],
                )
            except torch.cuda.OutOfMemoryError:
                self.logger.info('Segmentation is performed on the CPU (not enough GPU memory available)')
                self.segmenter.change_device('cpu')
                tissue, pen_marking = self.segmenter.segment(
                    image=image/255, 
                    tissue_threshold=self.settings['tissue_threshold'],
                    pen_marking_threshold=self.settings['pen_marking_threshold'],
                )
                self.segmenter.change_device(self.device)
        else:
            try:
                tissue = self.segmenter.segment(
                    image=image/255, 
                    tissue_threshold=self.settings['tissue_threshold'],
                )
            except torch.cuda.OutOfMemoryError:
                self.logger.info('Segmentation is performed on the CPU (not enough GPU memory available)')
                self.segmenter.change_device('cpu')
                tissue = self.segmenter.segment(
                    image=image/255, 
                    tissue_threshold=self.settings['tissue_threshold'],
                )
                self.segmenter.change_device(self.device)
            pen_marking = None
        self.logger.info(f'Finished segmenting WSI tissue regions')
       
        # (6) tessellate the tissue segmentation
        uncorrected_tile_information = tessellate(
            segmentation=tissue,
            shape=tuple([int(s/ratio) for s in self.settings['tile_shape']]),
            stride=tuple([int(s/ratio) for s in self.settings['stride']]),
            min_tissue_fraction=self.settings['min_tissue_fraction'],
            exclusion_map=pen_marking,
            exceed_image=self.settings['tiles_exceed_image'],
        )
        self.logger.info(f'Finished tessellating WSI tissue regions')

        # (7) [optional] save a visualization of the tiles on top of the image 
        # and segmentation
        if self.settings['save_tessellation_visualizations']:                                           
            visualize_tessellation(
                images=[image, combine(tissue, pen_marking)], 
                tile_information=uncorrected_tile_information, 
                output_path=self.get_path(visualization_path),
                line_color='hsv',
                line_width=1,
                downscale_factor=2,
                axis=0,
            )
            self.logger.info(f'Finished visualizing WSI tessellation')
        
        # (8) convert the tile coordinates to the extraction magnification
        tile_information = {}
        if len(uncorrected_tile_information):
            for cross_section, tiles in uncorrected_tile_information.items():
                corrected_tiles = []
                for position, location, shape in tiles:
                    corrected_tiles.append((
                        position, 
                        (int(location[0]*ratio), int(location[1]*ratio)),
                        (int(shape[0]*ratio), int(shape[1]*ratio)),
                    ))
                tile_information[cross_section] = corrected_tiles

        return tile_information