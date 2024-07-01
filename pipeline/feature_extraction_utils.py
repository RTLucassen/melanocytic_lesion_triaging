"""
Implementation of code to extract ViT features from whole slide image tiles.
"""

import json
import random
from datetime import datetime, timedelta
from logging import Logger
from math import ceil
from pathlib import Path
from typing import Any, Callable, Optional, Union

import albumentations as A
import numpy as np
import torch
from einops import rearrange
from slideloader import SlideLoader
from torch.utils.data import DataLoader, SequentialSampler

from HIPT.ViT import ViT, convert_state_dict


class TileDataset(torch.utils.data.Dataset):
    
    max_difference = 0.1

    def __init__(
        self, 
        image_directory: Optional[Union[str, Path]] = None, 
        tile_information: Optional[dict[int, dict[str, Any]]] = None,
        tile_information_path: Optional[Union[str, Path]] = None, 
        magnification: float = 20,
        channels_first: bool = True,
        variants: int = 0,
        transform: Optional[Callable] = None,
        first_not_augmented: bool = True,
        level: Optional[str] = None,
        chunk_shape: tuple[int, int] = (256, 256),
        seed: Optional[Union[str, int]] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initializes dataset instance to load tiles from whole slide images.

        Args:
            image_directory:  Path to folder where all images are stored.
            tile_information:  Dictionary with tile information.
            tile_information_path:  Path to file with tile information.
            magnification:  Magnification at which the tile is loaded.
            channels_first:  Indicates whether the channels dimension should be 
                before the spatial dimensions (i.e., True), or after the spatial
                dimensions (i.e., False).
            variants:  Number of augmented variants.
            transform:  Function for data augmentation.
            first_not_augmented:  Indicates whether the first feature variant
                should be extracted from the original tile without augmentation,
                even if a function for data augmentation was provided.
            level:  The level at which the same image augmentation is applied,
                which can be 'specimen', 'slide', 'cross-section', or 'tile'.
            seed:  Seed value for data augmentation.
            chunk_shape:  Shape of chunks to load the tile in pixels as (height, width).
            logger:  Logger instance.
        """
        # check if the input is valid
        if ((tile_information is not None and tile_information_path is not None)
            or (tile_information is None and tile_information_path is None)):
            raise ValueError('Provide an argument for either `tile_information` '
                             'or `tile_information_path`.')

        # read the tile information from the tile information file
        if tile_information_path is not None:
            self.data_dict, self.indices = read_tile_information(
                tile_information_path=tile_information_path,
                image_directory=image_directory,
            )
        elif tile_information is not None:
            self.data_dict = tile_information
            self.indices = []
            # get the indices from the data dictionary
            for specimen_index in self.data_dict:
                image_info = self.data_dict[specimen_index]['images']
                for image_index in image_info:
                    tile_info = image_info[image_index]['tiles']
                    for cross_section_index, tiles in tile_info.items():
                        for tile_index, _ in enumerate(tiles):
                            self.indices.append((specimen_index, image_index, 
                                                 cross_section_index, tile_index))
        
        # create mapping between tile indices and unique specimen indices
        self.index_mapping = {}
        unique_specimen_index = 0
        for index in self.indices:
            if index[:-1] not in self.index_mapping:
                self.index_mapping[index[:-1]] = unique_specimen_index
                unique_specimen_index += 1        

        # initialize slide loader and instance attributes
        self.loader = SlideLoader({'max_difference': 0.49/4096})
        self.magnification = magnification
        self.channels_first = channels_first
        self.variants = variants
        self.transform = transform
        self.first_not_augmented = first_not_augmented
        self.level = level
        self.chunk_shape = chunk_shape
        self.seed = seed
        self.logger = logger

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> Optional[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
        """
        Load a tile and prepare it.

        Args:
            index:  Index for selecting tile from the dataset.

        Returns:
            tile_index:  Torch tensor indicating the tile index, which is the 
                combination of the specimen index, image index, cross-section index, 
                and tile index.
            position:  Torch tensor indicating the horizontal and vertical position.
            tile_variants:  Whole slide image tile variants (with different augmentations).
        """
        # get the indices
        specimen_index, image_index, cross_section_index, tile_index = self.indices[index]
        # get the path to the image (or multiple images in case of DICOM) 
        # and the tile coordinate
        paths = self.data_dict[specimen_index]['images'][image_index]['paths']
        tile_coord = (self.data_dict[specimen_index]['images'][image_index]
                      ['tiles'][cross_section_index][tile_index])
        # load the tile
        position, location, shape = tile_coord
        location = tuple(location)
        shape = tuple(shape)

        # load the slide and determine the exact magnification for extraction
        self.loader.load_slide(paths)
        magnification_levels = self.loader.get_properties()['magnification_levels']
        if (max(magnification_levels)+self.max_difference < self.magnification):
            raise ValueError('Maximum magnification available is smaller than '
                             'the specified extraction magnification.')
        else:
            exact_magnification = None
            for level in magnification_levels:
                if abs(self.magnification-level) < self.max_difference:
                    exact_magnification = level
            if exact_magnification is None:
                exact_magnification = self.magnification
        
        # read the tile       
        try:
            if (shape[0] <= self.chunk_shape[0]) and (shape[1] <= self.chunk_shape[1]):
                tile = self.loader.get_tile(
                    magnification=exact_magnification, 
                    location=tuple(location), 
                    shape=tuple(shape),
                    channels_last=True,
                )
            else:
                tile = self.__get_tile_in_chunks(
                    magnification=exact_magnification, 
                    location=location, 
                    shape=shape,
                    channels_last=True,
                    chunk_shape=self.chunk_shape,
                )
        except Exception as error:
            if self.logger is not None:
                self.logger.info(f'Tile reading aborted due to the following error:\n{error}')
            return None
        else:
            original_tile = tile
            tile_variants = []
            for i in range(self.variants):
                # apply image augmentation transform to tile if specified
                if self.transform is not None:
                    # define the seed depending on the level
                    seed = '' 
                    levels =  ['specimen', 'slide', 'cross-section', 'tile']
                    for level, level_index, length in zip(levels, self.indices[index], [6, 3, 3, 4]):
                        seed += str(level_index).zfill(length)
                        if level == self.level:
                            break
                    seed += str(i).zfill(len(str(self.variants)))
                    if self.seed is not None:
                        seed += str(self.seed)
                    random.seed(seed)
                    # apply augmentation transform 
                    # (except for the first iteration if specified)
                    if self.first_not_augmented and i == 0:
                        tile = original_tile
                    else:
                        tile = self.transform(image=original_tile)['image']

                # change position of channels dimension
                if self.channels_first:
                    tile = np.transpose(tile, (2,0,1))

                # convert tile to torch Tensor 
                tile = torch.from_numpy(tile)
                tile_variants.append(tile)

        # convert tile index and position to torch tensors
        tile_index = torch.tensor([*self.indices[index]], dtype=int)
        unique_index = self.index_mapping[self.indices[index][:-1]]
        position = torch.tensor([unique_index, *position], dtype=int)

        return tile_index, position, tile_variants

    def __get_tile_in_chunks(
        self,
        magnification: float,
        location: tuple[int, int],
        shape: tuple[int, int] = (256, 256), 
        chunk_shape: tuple[int, int] = (256, 256),
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get a tile in chunks from the whole slide image based on the specified
        magnification, location, and shape.
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            location:  Location of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
            chunk_shape:  Shape of chunks to load the tile in pixels as (height, width).
        
        Returns:
            tile:  Whole slide image tile [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # determine locations and shapes of the chunks
        absolute_locations = []
        relative_locations = []
        shapes = []
        height, width = shape
        for y in range(ceil(height/chunk_shape[0])):
            for x in range(ceil(width/chunk_shape[1])):
                relative_locations.append((x*chunk_shape[1], 
                                           y*chunk_shape[0]))
                absolute_locations.append((location[0]+x*chunk_shape[1], 
                                           location[1]+y*chunk_shape[0]))
                shapes.append((
                    min(height, (y+1)*chunk_shape[0])-(y*chunk_shape[0]),
                    min(width, (x+1)*chunk_shape[1])-(x*chunk_shape[1]),
                ))
        # get chunks from the image at the requested magnification
        chunks = self.loader.get_tiles(
            magnification=magnification, 
            locations=absolute_locations,
            shapes=shapes,
            channels_last=channels_last,
        )
        # stitch the chunks together to obtain full tile
        tile_shape = (height, width, 3) if channels_last else (3, height, width)
        tile = np.zeros(tile_shape, dtype=np.uint8)

        # add values of the chunks to empty tile
        for (x, y), (height, width), chunk in zip(relative_locations, shapes, chunks):
            if channels_last:
                tile[y:y+height, x:x+width, :] = chunk
            else:
                tile[:, y:y+height, x:x+width] = chunk

        return tile


# helper function for reading tile information
def read_tile_information(
    tile_information_path: Union[str, Path], 
    image_directory: Optional[Union[str, Path]] = None, 
) -> tuple[dict[int, dict[str, Any]], list[tuple[int, int, int, int]]]:
    """
    Reads tile information

    Args:
        tile_information_path:  Path to file with tile information.
        image_directory:  Path to folder where all images are stored.

    Returns:
        data_dict:  Dictionary with image paths and tile coordinate information.
        indices:  List of 4-tuples with for each tile (specimen index, 
            image index, cross-section index, tile index) 
    """
    # read tile information from file
    with open(tile_information_path, 'r') as f:
        lines = f.readlines()

    # check if the number of lines is a multiple of three
    if len(lines) % 3 != 0:
        raise ValueError('The number of lines must be a multiple of three.')

    # initialize dictionary and lists for storing the tile information 
    data_dict = {}
    indices = []
    # loop over tile information per whole slide image
    for i in range(int(len(lines)/3)):
        image_filenames = eval(lines[i*3])
        if image_directory is not None:
            image_paths = [(Path(image_directory)/name) for name in image_filenames]
        else:
            image_paths = image_filenames
        specimen_information = json.loads(lines[i*3+1])
        tile_information = {int(k): v for k,v in json.loads(lines[i*3+2]).items()}

        # add information to dictionary and indices to list
        specimen_index = specimen_information['specimen_index']
        if specimen_index in data_dict:
            image_index = max(data_dict[specimen_index]['images'].keys())+1
            data_dict[specimen_index]['images'][image_index] = {
                'paths': image_paths,
                'tiles': tile_information,
            }
        else:
            image_index = 0
            data_dict[specimen_index] = {
                **specimen_information,
                'images': {
                    image_index: {
                        'paths': image_paths,
                        'tiles': tile_information,
                    },
                },
            }

        for cross_section_index, tiles in tile_information.items():
            for tile_index, _ in enumerate(tiles):
                indices.append((specimen_index, image_index, 
                                cross_section_index, tile_index))
    
    return data_dict, indices


def get_augmentation_transform(config: dict[str, Any]) -> Callable:
    """
    Get an augmentation transform based on the specified configuration.

    Args:
        config:  Dictionary with augmentation transform names as keys and
            dictionaries with argument names and corresponding values.
    
    Returns:
        tranform:  Augmentation transform.
    """
    transforms = []
    for name, settings in config.items():
        if name == 'ColorJitter':
            if settings['p'] > 0:
                transforms.append(
                    A.ColorJitter(**settings),
                )
        elif name == 'RGBShift':
            if settings['p'] > 0:
                transforms.append(
                    A.RGBShift(**settings),
                )
        elif name == 'GaussNoise':
            if settings['p'] > 0:
                transforms.append(
                    A.GaussNoise(**settings),
                )
        elif name == 'GaussianBlur':
            if settings['p'] > 0:
                transforms.append(
                    A.GaussianBlur(**settings),
                )
        else:
            NotImplementedError('Augmentation not implemented.')
    
    # combine augmentation transforms
    transform = A.Compose(transforms)

    return transform


class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        features: dict[tuple[int, int, int, int], dict[str, Any]]
    ) -> None:
        """
        Initializes dataset instance to return features.

        Args:
            features:  Dictionary with features and corresponding positions.
        """
        self.features = features
        self.indices = list(features.keys())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load a feature variants for training.

        Args:
            index:  Index for selecting feature variants from the dataset.

        Returns:
            tile_index:  Torch tensor indicating the tile index, which is the 
                combination of the specimen index, image index, cross-section index, 
                and tile index.
            position:  Torch tensor indicating the horizontal and vertical position.
            feature_variants:  Feature vectors variants extracted from (augmented)
                whole slide image tiles.
        """
        tile_index = self.indices[index]
        position = self.features[tile_index]['position']
        feature_variants = self.features[tile_index]['feature_variants']

        # convert tile index and position to torch tensors
        tile_index = torch.tensor([*tile_index], dtype=int)
        position = torch.tensor([*position], dtype=int)

        return tile_index, position, feature_variants


# helper function to collate items in batch that removes None 
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch):
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None


# feature extraction function
def extract_ViT_features(
    tile_dataloader: DataLoader, 
    configs: dict[str, Any],
    device: str = 'cpu',
    logger: Optional[Logger] = None,
    callback: Optional[Callable] = None,
    callback_timeout: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        tile_dataloader:  DataLoader class for loading whole slide image tiles 
            or features and corresponding positions.
        configs: list of dictionaries with the sequential feature extraction 
            configurations.
        device:  Name of device for feature extraction model inference.
        callback:  Callback function.
        callback_timeout:  Minimum time out in number of seconds after a callback 
            before the next callback.

    Returns:
        features:  Dictionary with feature vectors and corresponding positions.
    """
    features = {}
    dataloader = tile_dataloader
    # loop over the feature extraction configurations
    for i, config in enumerate(configs):
        # configure dataset and dataloader for features
        if i > 0:
            dataset = FeatureDataset(features)
            # initialize dataloader
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                sampler=SequentialSampler(dataset),
                collate_fn=custom_collate_fn,
            )
        # initialize model
        model = ViT(**config['model_arguments'])
        state_dict = convert_state_dict(
            path=config['state_dict_path'],
            pos_embed_config=config['model_arguments']['pos_embed_config'],
            pytorch_attn_imp=config['model_arguments']['pytorch_attn_imp'],
        )
        model.load_state_dict(state_dict=state_dict)
        model.to(device)
        model.eval()
        
        time = datetime.now()
        features = {}
        with torch.no_grad():
            # read whole slide image tiles or features using dataloader
            for batch in dataloader:
                if batch is None:
                    continue
                tile_index, position, data_variants = batch
                tile_index = tuple(tile_index.tolist()[0])
                position = tuple(position.tolist()[0])

                # loop over the tile or feature vector variants
                for j, data in enumerate(data_variants):
                    # split the tile into smaller tiles if necessary
                    if config['tile_shape'] is not None:
                        height, width = config['tile_shape']
                        data = data.unfold(2, height, height).unfold(3, width, width)
                        N_tiles = data.shape[2:4]         
                        data = rearrange(data, 'b c p1 p2 w h -> (b p1 p2) c w h')
                        data = data/255

                    # model inference
                    feature = (model(data.to(device, non_blocking=True))).to('cpu')
                    if feature is not None:
                        if config['tile_shape'] is not None:
                            feature = feature.reshape(
                                (*N_tiles, config['model_arguments']['embed_dim']),
                            )
                            feature = torch.permute(feature, (2, 0, 1))
                        else:
                            feature = feature[0, ...]
                        # change the structure of the feature dictionary depending
                        # the features are from an intermediate (separation at top
                        # level based on tile) or the final level (separation at top
                        # level based on variant)
                        if i == len(configs)-1:
                            if j not in features:
                                features[j] = {}
                            features[j][tile_index] = {'feature': feature,
                                                       'position': position}
                        else:
                            # check if the tile index already exists
                            if tile_index in features:
                                features[tile_index]['feature_variants'].append(feature)
                            else:
                                features[tile_index] = {'feature_variants': [feature],
                                                        'position': position}
                # log feature extraction
                if logger is not None:
                    if len(data_variants) == 1:
                        logger.info(f'Extracted feature (from {tuple(data.shape)} '
                                    f'to {tuple(feature.shape)})')
                    else:
                        logger.info(f'Extracted {len(data_variants)} feature variants'
                                    f' (from {tuple(data.shape)} to {tuple(feature.shape)})')
                
                # execute callback
                if callback is not None:
                    current_time = datetime.now()
                    if current_time-time > timedelta(seconds=callback_timeout):
                        callback()
                        time = current_time

        # return model to cpu
        model.to('cpu')
        
    return features