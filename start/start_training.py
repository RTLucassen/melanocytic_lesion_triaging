"""
Start model training.
"""

import json    
import platform

from training.training_service import TrainingService

# define paths
if platform.system() == 'Linux':
    superbatch_directory = '/projects/melanocytic_lesion_triaging/superbatches'
    model_directory = '/projects/melanocytic_lesion_triaging/models'
    device = 'cuda'
    workers = 4
    pin_memory = True
elif platform.system() == 'Windows':
    superbatch_directory = r'projects\melanocytic_lesion_triaging\superbatches'
    model_directory = r'projects\melanocytic_lesion_triaging\models'
    device = 'cpu'
    workers = 0
    pin_memory = False
else:
    raise NotImplementedError
pipeline_config_file = 'pipeline.json'
training_config_path = 'training.json'

if __name__ == '__main__':

    # load training config
    with open(training_config_path, 'r') as f:
        training_config = json.loads(f.read())

    service = TrainingService(
        model_directory=model_directory,
        superbatch_directory=superbatch_directory,
        pipeline_config_file=pipeline_config_file,
        training_config=training_config,
        device=device,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    service.start()