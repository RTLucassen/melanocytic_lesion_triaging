"""
Start model evaluation.
"""

import platform

import numpy as np

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

if __name__ == '__main__':

    service = EvaluationService(
        experiment_name='exp068_f1_baseline5',
        thresholds=np.arange(0, 1.001, 0.005),
        sensitivity_levels=[0.99, 0.98, 0.95],
        bootstrap_iterations=10000,
        bootstrap_stratified=True,
        confidence_level=0.95,
        calibration_bins=20,
        model_directory=model_directory,
        superbatch_directory=superbatch_directory,
        pipeline_config_file=pipeline_config_file,
        training_config_file=training_config_file,
        device=device,
        num_workers=workers,
        pin_memory=pin_memory,
        image_extension='.png',
    )
    service.start(['test'])