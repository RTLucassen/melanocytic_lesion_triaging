"""
Start model evaluation.
"""

import os
import platform
from pathlib import Path

import numpy as np
from natsort import natsorted

from evaluation.evaluation_service import EvaluationService

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
    pin_memory = False,
else:
    raise NotImplementedError
pipeline_config_file = 'pipeline.json'
training_config_file = 'training.json'
overwrite = False

if __name__ == '__main__':

    experiment_names = os.listdir(model_directory)

    # loop over experiments
    for experiment_name in natsorted(experiment_names):
        # get the validation fold 
        fold = f'fold_{experiment_name.split("_")[1][1]}'
        # check if the evaluation was already performed
        if ((Path(model_directory) / experiment_name / f'results_{fold}.xlsx').exists()
            and not overwrite):
            continue
        # start evaluation  
        service = EvaluationService(
            experiment_name=experiment_name,
            thresholds=np.arange(0, 1.001, 0.005),
            model_directory=model_directory,
            superbatch_directory=superbatch_directory,
            pipeline_config_file=pipeline_config_file,
            training_config_file=training_config_file,
            device=device,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        service.start(fold)