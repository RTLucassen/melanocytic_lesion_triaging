"""
Generate and save attention maps and the top N tiles with the highest attention. 
"""

import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import SimpleITK as sitk
from slideloader import SlideLoader
from tqdm import tqdm

from pipeline.feature_extraction_utils import read_tile_information


# define experiment settings
fold = 'test_uncertainty'
experiment_name = 'test'
magnification = 1.25
opacity = 0.60
colormap = 'coolwarm'
line_width = 1
line_color = (0,0,0)
show_values = False
save_image = True
save_attn_image = True
save_top_N_tiles = 1

# define processing settings
extraction_magnification = 20.0
max_difference = 0.1

# define selected specimen
selected_specimen = []

# define paths
if platform.system() == 'Linux':
    models_directory = r'\projects\melanocytic_lesion_triaging\models'
    superbatch_directory = r'\projects\melanocytic_lesion_triaging\superbatches'
elif platform.system() == 'Windows':
    models_directory = r'projects\melanocytic_lesion_triaging\models'
    superbatch_directory = r'projects\melanocytic_lesion_triaging\superbatches'
else:
    raise NotImplementedError

output_folder = 'attention_maps'

if __name__ == '__main__':

    # initialize SlideLoader instance
    loader = SlideLoader({'max_difference': 0.49/4096})

    # load colormap
    attn_color = matplotlib.colormaps[colormap]

    # convert to pathlib Path objects
    model_directory = Path(models_directory) / experiment_name
    superbatch_directory = Path(superbatch_directory)
    output_directory = model_directory / output_folder

    # create output folder if it does not exist yet
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        raise FileExistsError('Output folder already exists.')

    # load attention results
    with open(model_directory / f'attention_{fold}.json', 'r') as f:
        attn_dict = {
            int(k): {eval(p): a for p,a in v.items()} for k,v in json.loads(f.read()).items()
        }

    # load dataset information (which contains the full paths of the WSIs)
    with open(superbatch_directory / 'dataset.json', 'r') as f:
        dataset_dict = {
            info['specimen_index']:info for info in json.loads(f.read())[fold]
        }

    # load tile information
    tile_dict = {}
    for folder in os.listdir(superbatch_directory):
        if fold in folder:
            path = superbatch_directory / folder / 'tile_information.txt'
            tile_dict = {**tile_dict, **(read_tile_information(path)[0])}

    # loop over attention results
    for specimen_index, attention in tqdm(attn_dict.items()):
        # get the paths to the images
        all_paths = dataset_dict[specimen_index]['paths']
        specimen = dataset_dict[specimen_index]['specimen']

        if len(selected_specimen):
            if specimen not in selected_specimen:
                continue

        # take the mean attention values
        attention = {pos: np.mean(values) for pos, values in attention.items()}
        max_attn_value = max(attention.values())

        # loop over images for a specimen
        section_index = 0
        for image_tile_dict in tile_dict[specimen_index]['images'].values():
            # loop over the image filenames and get the full path
            paths = []
            for name in image_tile_dict['paths']:
                for path in all_paths:
                    if name in path:
                        paths.append(Path(path))
                        break
            
            # loop over list of tiles per cross-section
            tile_attn_dict = {}
            for section_tiles in image_tile_dict['tiles'].values():
                # skip cross-sections without tiles
                if len(section_tiles):
                    for tile_coord in section_tiles:
                        # get the position, location, and size of tile
                        pos, loc, size = tile_coord
                        # get the corresponding attention value
                        attn_value = attention[(section_index, *pos)]
                        # store attention value with coordinate in dictionary
                        tile_attn_dict[(tuple(loc), tuple(size))] = attn_value
                    section_index += 1

            # load image
            loader.load_slide(paths)
            image = loader.get_image(magnification)

            if save_image:
                image_output_path = output_directory / f'{specimen}_{paths[0].stem}.png'
                sitk.WriteImage(sitk.GetImageFromArray(image[None, ...]), image_output_path.as_posix())

            if save_attn_image:
                # correct coordinates
                tile_magnification = extraction_magnification
                for level in loader.get_properties()['magnification_levels']:
                    if abs(tile_magnification-level) < max_difference:
                        tile_magnification = level
                # calculate the correction ratio
                ratio = tile_magnification / magnification

                # loop over tiles
                for tile_coord, attn_value in tile_attn_dict.items():
                    # correct coordinates for the magnification of the output image
                    top = int(tile_coord[0][1]/ratio)
                    left = int(tile_coord[0][0]/ratio)
                    height = int(tile_coord[1][1]/ratio)
                    width = int(tile_coord[1][0]/ratio)
                    # crop the tile from the image
                    tile = image[top:top+height, left:left+width]
                    # create the tile with the heatmap color based on the attention weight
                    color = np.array([int(x*255) for x in attn_color(attn_value/max_attn_value)][:-1])
                    attn_tile = np.ones_like(tile)*color[None, None, :]
                    # combine the tile and heatmap color                
                    combined_tile = attn_tile*opacity+tile*(1-opacity)
                    # place tile in image and add lines at the tile edges
                    image[top:top+height, left:left+width] = combined_tile
                    image = cv2.rectangle(
                        img=image.astype(np.uint8), pt1=(left, top), 
                        pt2=(left+width, top+height), color=line_color, 
                        thickness=line_width,
                    )
                    if show_values:
                        image = cv2.putText(
                            img=image, text=f'{attn_value:0.3f}', 
                            org=(int(left+width*0.15), int(top+height*0.55)), 
                            fontFace=1, fontScale=4, color=line_color, thickness=4
                        )

                # Save the image with squares drawn
                attn_image_output_path = output_directory / f'{specimen}_{paths[0].stem}_attn.png'
                cv2.imwrite(attn_image_output_path.as_posix(), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # save top N tiles
            if not (save_top_N_tiles == 0 or save_top_N_tiles is None):
                # order tiles based on descending attention value
                tile_info = sorted([(value, coord) for coord, value in tile_attn_dict.items()], reverse=True)
                # loop over the top N tiles and save the tile
                N_tiles = min(len(tile_attn_dict), save_top_N_tiles)
                for i, (att, coord) in enumerate(tile_info[:N_tiles]):
                    tile = loader.get_tile(tile_magnification, coord[0], coord[1])
                    tile_output_path = output_directory / f'{specimen}_{paths[0].stem}_tile_{i}.png'
                    sitk.WriteImage(sitk.GetImageFromArray(tile[None, ...]), tile_output_path.as_posix())              