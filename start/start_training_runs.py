"""
Start model training.
"""

import json    
import platform
from natsort import natsorted
from pathlib import Path

from training.training_service import TrainingService

# define paths
if platform.system() == 'Linux':
    superbatch_directory = '/projects/melanocytic_lesion_triaging/superbatches'
    model_directory = '/projects/melanocytic_lesion_triaging/models'
    device = 'cuda'
    workers = 4
    pin_memory = True
    progress_bar = False
elif platform.system() == 'Windows':
    superbatch_directory = r'projects\melanocytic_lesion_triaging\superbatches'
    model_directory = r'projects\melanocytic_lesion_triaging\models'
    device = 'cpu'
    workers = 0
    pin_memory = False
    progress_bar = True
else:
    raise NotImplementedError
pipeline_config_file = 'pipeline.json'
training_configs = Path('/repositories/melanocytic_lesion_triaging/training_configs')

if __name__ == '__main__':

    # loop over training run configurations
    for training_config_path in natsorted(training_configs.iterdir()):
        # load training config
        with open(training_config_path, 'r') as f:
            training_config = json.loads(f.read())

        try:
            service = TrainingService(
                model_directory=model_directory,
                superbatch_directory=superbatch_directory,
                pipeline_config_file=pipeline_config_file,
                training_config=training_config,
                device=device,
                num_workers=workers,
                pin_memory=pin_memory,
                progress_bar=progress_bar,
            )
            service.start()
        except FileExistsError:
            print('Continuing with the next training configuration.')