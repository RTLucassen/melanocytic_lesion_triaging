"""
Implementation of whole slide image (WSI) evaluation service.
"""

import io
import json
import os
import platform
import random
import time
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from natsort import natsorted
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
)
from torch.utils.data import DataLoader, SequentialSampler
from torchinfo import summary
from tqdm import tqdm

from HIPT.ViT import ViT
from pipeline.base_service import BaseService
from training.training_utils import (
    FeatureDataset, 
    get_patient_vector, 
    get_label,
    read_feature_information,
)

class EvaluationService(BaseService):

    def __init__(
        self, 
        experiment_name: str,
        thresholds: Union[float, list[float]],
        sensitivity_levels: Optional[Union[float, list[float]]],
        bootstrap_iterations: Optional[int],
        bootstrap_stratified: bool,
        confidence_level: float,
        calibration_bins: int,
        model_directory: Union[str, Path],
        superbatch_directory: Union[str, Path], 
        pipeline_config_file: str,
        training_config_file: str,
        device: str = 'cpu',
        num_workers: int = 0,
        pin_memory: bool = False,
        image_extension: str = '.png',
    ) -> None:
        """
        Initialize service for evaluation.

        Args:
            experiment_name:  Name of experiment folder 
            thresholds:  One or more binarization thresholds for evaluating 
                the model performance.
            sensitivity_levels:  Levels of sensitivity to evaluate the specificity for.
            bootstrap_iterations:  Number of iterations used for bootstrapping 
                to calculate confidence intervals.
            bootstrap_stratified:  Indicates whether stratified sampling is used.
            confidence_level:  Confidence level for interval calculated using bootstrapping.
            calibration_bins:  Number of bins in calibration plot and calculation.
            model_directory:  Directory where all model files are stored.
            superbatch_directory:  Directory where all superbatches are stored.
            pipeline_config_file:  Name of pipeline config file.
            training_config_file:  Name of training config file.
            device:  Name of device for feature extraction model inference.
            num_workers:  Number of workers for the dataloader.
            pin_memory:  Indicates whether pinned memory is used for the dataloader.
            image_extension:  Name of image extension for saving plots of results.
        """
        super().__init__(superbatch_directory, pipeline_config_file)

        # initialize additional instance attributes
        self.experiment_name = experiment_name
        self.thresholds = thresholds
        self.sensitivity_levels = sensitivity_levels
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_stratified = bootstrap_stratified
        self.confidence_level = confidence_level
        self.calibration_bins = calibration_bins
        self.model_directory = Path(model_directory)
        self.experiment_directory = self.model_directory / self.experiment_name
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_extension = image_extension

        # load training config
        with open(self.experiment_directory / training_config_file, 'r') as f:
            self.settings = json.loads(f.read())

        # check if model and experiment directory exist
        if not self.model_directory.exists():
            raise FileExistsError('Directory for models does not exist.')
        elif not self.experiment_directory.exists():
            raise FileExistsError('Experiment directory does not exist.')
        
    def start(self, variants: Union[list[str], str]) -> None:
        """ 
        Start the evaluation after checking if the data has been transferred 
        and preprocessed (and optionally features extracted).

        Args:
            variants:  One or more superbatch variants on which the model
                is evaluated.
        """
        # perform check to determine if evaluation is possible
        if not self.config['feature_extraction']:
            raise ValueError('Training is expected to be performed on feature vectors.')

        # if a single variant was specified, put it in a list
        if isinstance(variants, str):
            variants = [variants]

        self.load_status()
        # check if the specified variants are valid
        for variant in variants:
            if variant not in self.config['variants']:
                raise ValueError(f'Invalid variant: {variant}')

        self.logger.info((
            'Check if preprocessing and feature extraction have been completed '
            f'for the following variants: {", ".join(variants)}'
        ))
        continue_checking = True
        while continue_checking:
            all_completed = True
            for variant in variants:
                if variant not in self.status:
                    all_completed = False
                elif not self.status[variant]['feature_extraction_completed']:
                    all_completed = False
            if all_completed:
                continue_checking = False
                self.logger.info((
                    'Preprocessing and feature extraction have been completed '
                    f'for the following variants: {", ".join(variants)}'
                ))
            else:
                self.logger.info(f'Retry check after {self.pause_duration} seconds')
                time.sleep(self.pause_duration)
                self.load_status()

        # start evaluation
        for variant in variants:
            self.evaluate_model(variant)
        self.logger.info('Finished evaluation')
    
    def evaluate_model(self, variant: Union[list[str], str]) -> None:
        """
        Start model evaluation.

        Args:
            variant:  Superbatch variant on which to evaluate the model.
        """
        # configure the device
        if self.device == 'cuda':
            if torch.cuda.is_available():
                self.logger.info(f'CUDA available: {torch.cuda.is_available()}')
            else:
                self.device = 'cpu'
                self.logger.info('CUDA unavailable')
        self.logger.info(f'Device: {self.device}')

        # seed randomness
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        self.logger.info(f'Using seed: {seed}')

        # initialize model
        model = ViT(**self.settings['model_config']['model_arguments'])
        # get the last checkpoint saved
        checkpoints_directory = self.experiment_directory / 'checkpoints'
        last_checkpoint = natsorted(os.listdir(checkpoints_directory))[-1]
        self.logger.info(f'Load the model parameters from last checkpoint: {last_checkpoint}')
        # load model parameters
        state_dict = torch.load(checkpoints_directory / last_checkpoint, 
                                map_location=self.device)
        model.load_state_dict(state_dict=state_dict['model_state_dict'])
        model.to(self.device)
        model.eval()

        # capture model summary in variable
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model=model, depth=4, col_names=['num_params'])
        self.logger.info('\n'+f.getvalue())
        self.logger.info(model)
            
        # compile the model
        if (self.settings['model_config']['compile_model'] 
            and platform.system() == 'Linux'):
            model = torch.compile(model)

        self.logger.info(f'Start model evaluation for superbatch type: {variant}')

        # get all superbatches for the first epoch of the variant
        superbatches = list(
            self.status[variant]['epochs']['0']['superbatches'].keys(),
        )
        feature_information = {}
        for superbatch in superbatches:
            # define paths to the feature information file 
            # and the feature directory
            filename = self.config['feature_extraction_settings']['output_filename']
            feature_information_path = self.directory / superbatch / filename
            feature_directory = self.directory / superbatch / self.optional_folders[2]
            
            # read the feature information for the superbatch 
            superbatch_feature_information = read_feature_information(
                feature_information_path=feature_information_path,
                feature_directory=feature_directory,
            )
            # add it to the combined feature information dictionary
            feature_information = {
                **superbatch_feature_information,
                **feature_information, 
            }

        # initialize dataset and dataloader instances
        dataset = FeatureDataset(
            feature_information=feature_information,
            patient_vector_func=get_patient_vector,
            label_func=get_label,
            length = None,
            only_first_variant=True,
            interpolate_features=False,
            section_dropout_prob=0.0,
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler= SequentialSampler(dataset),
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        y_true = []
        y_pred = []
        attn_dicts = []
        with torch.no_grad():
            for (x, x_patient, pos, _, y) in dataloader:
                # bring the data to the correct device
                x = x.to(self.device)
                x_patient = x_patient.to(self.device)
                pos = pos.to(self.device)

                # get model prediction and optionally the self-attention
                pred, attn = model(x=x, x_patient=x_patient, pos=pos, 
                                    return_last_self_attention=True)
                pred = torch.softmax(pred.to('cpu'), dim=1)
                
                # add true and predicted label to lists
                y_true.append(float(y[0, 1]))
                y_pred.append(float(pred[0, 1]))

                # save attention for last layer
                attn_dict = {}
                for i in range(pos.shape[1]):
                    position = str(tuple(pos[0, i, :].tolist()))
                    attn_values = [round(v, 3) for v in attn[0, :, 0, i+1].tolist()]
                    attn_dict[position] = attn_values
                attn_dicts.append(attn_dict)

        # save JSON with attention values
        specimen_indices = [item['specimen_index'] for item in feature_information.values()]
        attn_dicts = {i:a for i,a in zip(specimen_indices, attn_dicts)}
        with open(self.experiment_directory / f'attention_{variant}.json', 'w') as f:
            f.write(json.dumps(attn_dicts))

        # determine the scanner based on the datatype of the WSIs
        scanner = []
        for results_item in feature_information.values():
            for dataset_item in self.data[variant]:
                if results_item['specimen_index'] == dataset_item['specimen_index']:
                    datatypes = [os.path.splitext(path)[-1] for path in dataset_item['paths']]
                    types = []
                    if '.dcm' in datatypes:
                        types.append('Aperio')
                    if '.ndpi' in datatypes:
                        types.append('Hamamatsu')
                    scanner.append(' & '.join(types))

        # save prediction result per specimen
        predictions = {
            'specimen_index': [item['specimen_index'] for item in feature_information.values()],
            'scanner' : scanner,
            'patient': [item['patient'] for item in feature_information.values()],
            'sex': [item['sex'] for item in feature_information.values()],
            'age': [item['age'] for item in feature_information.values()],
            'location': [item['location'] for item in feature_information.values()],
            'specimen': [item['specimen'] for item in feature_information.values()],
            'diagnosis': [item['diagnosis'] for item in feature_information.values()],
            'secondary_findings': [item['secondary_findings'] for item in feature_information.values()],
            'y_true': y_true,
            'y_pred': y_pred,
        }

        # perform per class evaluation
        class_predictions = predict_per_class(y_true, y_pred, predictions['diagnosis'])
        # perform threshold-dependent evaluation
        threshold_results = evaluate_with_thresholds(y_true, y_pred, self.thresholds)
        if self.sensitivity_levels is not None:
            spec_at_sens_results = specificity_at_sensitivity(y_true, y_pred, scanner, self.sensitivity_levels)
        # perform threshold-independent evaluation 
        area_results = evaluate_without_thresholds(y_true, y_pred, scanner, 
            roc_figure_path=self.experiment_directory / f'ROC_curve_{variant}{self.image_extension}',
            pr_figure_path=self.experiment_directory / f'PR_curve_{variant}{self.image_extension}',
            cal_figure_path=self.experiment_directory / f'Calibration_{variant}{self.image_extension}',
            bins=self.calibration_bins,
        )

        # perform bootstrapping for evaluation
        if self.bootstrap_iterations is not None:
            bootstrap_spec_at_sens_results = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                scanner=scanner, 
                eval_func=partial(specificity_at_sensitivity, 
                                  sensitivity_levels=self.sensitivity_levels), 
                iterations=self.bootstrap_iterations, 
                stratified=self.bootstrap_stratified,
                confidence_level=self.confidence_level, 
                row_key='set', 
                seed=self.config['seed'],
            )
            bootstrap_area_results = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                scanner=scanner, 
                eval_func=partial(evaluate_without_thresholds, bins=self.calibration_bins), 
                iterations=self.bootstrap_iterations, 
                stratified=self.bootstrap_stratified,
                confidence_level=self.confidence_level, 
                row_key='set', 
                seed=self.config['seed'],
            )

        # convert dictionaries to dataframes
        predictions_df = pd.DataFrame.from_dict(predictions)
        class_predictions_df = pd.DataFrame.from_dict(class_predictions)
        threshold_results_df = pd.DataFrame.from_dict(threshold_results)
        if self.sensitivity_levels is not None:
            spec_at_sens_results_df = pd.DataFrame.from_dict(spec_at_sens_results)
        area_results_df = pd.DataFrame.from_dict(area_results)
        if self.bootstrap_iterations is not None:
            bootstrap_spec_at_sens_results_df =  pd.DataFrame.from_dict(bootstrap_spec_at_sens_results)
            bootstrap_area_results_df = pd.DataFrame.from_dict(bootstrap_area_results)

        # save evaluation results in spreadsheet
        with pd.ExcelWriter(self.experiment_directory / f'results_{variant}.xlsx') as writer:
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
            class_predictions_df.to_excel(writer, sheet_name='Class predictions', index=False)
            threshold_results_df.to_excel(writer, sheet_name='Results (threshold)', index=False)
            if self.sensitivity_levels is not None:
                spec_at_sens_results_df.to_excel(writer, sheet_name='Results (spec @ sens)', index=False)
            area_results_df.to_excel(writer, sheet_name='Results (area)', index=False)
            if self.bootstrap_iterations is not None:
                bootstrap_spec_at_sens_results_df.to_excel(writer, sheet_name='Bootstrap results (spec @ sens)', index=False)
                bootstrap_area_results_df.to_excel(writer, sheet_name='Bootstrap results (area)', index=False)
        
def predict_per_class(
    y_true: list[int], 
    y_pred: list[float], 
    diagnosis_codes: list[str],
) -> dict[str, list]:
    """ 
    Calculate statistics on predictions per class.

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        diagnosis_codes:  Diagnosis code(s) per case.

    Returns:
        class_predictions:  Predicted probability statistics per class.
    """
    # determine the mean and standard predicted probability for each class
    predictions_per_class = {}
    for diagnosis, label, pred in zip(diagnosis_codes, y_true, y_pred):
        # remove symbols and split diagnosis codes
        diagnosis = diagnosis.replace('(', '').replace(')', '').replace('+', '').replace('/', '').replace('&', '').replace('  ', ' ')
        codes = [code for code in list(set(diagnosis.split(' '))) if code != '']
        # assign the codes depending on the label
        for code in codes:
            key = (code, label)
            if key in predictions_per_class:
                predictions_per_class[key].append(pred)
            else:
                predictions_per_class[key] = [pred]
    
    # save prediction results per class
    class_predictions = {
        'diagnosis code': [key[0] for key in predictions_per_class.keys()],
        'y_true': [key[1] for key in predictions_per_class.keys()],
        'N cases': [len(preds) for preds in predictions_per_class.values()],
        'mean y_pred': [np.mean(preds) for preds in predictions_per_class.values()],
        'SD y_pred': [np.std(preds) for preds in predictions_per_class.values()],
    }

    return class_predictions


def evaluate_with_thresholds(
    y_true: list[int], 
    y_pred: list[float], 
    thresholds: Union[float, list[float]],
) -> dict[str, list]:
    """ 
    Evaluate predictive performance of model using threshold-dependent metrics.

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        thresholds:  One or more thresholds to use for evaluation.

    Returns:
        results:  Results of evaluation per threshold.
    """
    # define dictionary to save model performance in terms of several metrics 
    # for each threshold      
    results = {'threshold': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [],
                'sensitivity / recall': [], 'specificity': [], 'precision': [], 
                'f1_score': [], 'accuracy': [], 'balanced_accuracy': []}
    
    # if a single threshold was provided, put it in a list
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    # calculate performance metrics for each threshold
    for threshold in thresholds:
        binary_y_pred = [1 if pred > threshold else 0 for pred in y_pred]
        combinations = zip(binary_y_pred, y_true)
        counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for combination in combinations:
            if combination == (1, 1):
                counts['TP'] += 1
            elif combination == (0, 0):
                counts['TN'] += 1
            elif combination == (1, 0):
                counts['FP'] += 1
            elif combination == (0, 1):
                counts['FN'] += 1
        sensitivity = counts['TP'] / (counts['TP'] + counts['FN'])
        specificity = counts['TN'] / (counts['TN'] + counts['FP'])
        if (counts['TP'] + counts['FP']) == 0:
            precision = None
        else:
            precision = counts['TP'] / (counts['TP'] + counts['FP'])
        f1_score = 2*counts['TP'] / (2*counts['TP'] + counts['FP'] + counts['FN'])
        accuracy = ((counts['TP']+counts['TN']) 
                    / (counts['TP'] + counts['TN'] + counts['FP'] + counts['FN']))
        balanced_accuracy = (sensitivity + specificity) / 2

        # add values to results dictionary
        results['threshold'].append(threshold)
        results['TP'].append(counts['TP'])
        results['TN'].append(counts['TN'])
        results['FP'].append(counts['FP'])
        results['FN'].append(counts['FN'])
        results['sensitivity / recall'].append(sensitivity)
        results['specificity'].append(specificity)
        results['precision'].append(precision)
        results['f1_score'].append(f1_score)
        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)

    return results


def specificity_at_sensitivity(
    y_true: list[int],
    y_pred: list[float],
    scanner: list[str],
    sensitivity_levels: Union[float, list[float]],
    method: str = 'interpolate',
) -> dict[str, list]:
    """
    """    
    # determine what partitions and colors to use
    if scanner is None:
        partitions = ['All']
    else:
        partitions = ['Aperio', 'Hamamatsu', 'All']
    
    # if the sensitivity level is a single value, add it to a list
    if not isinstance(sensitivity_levels, (list, tuple)):
        sensitivity_levels = [sensitivity_levels]

    # pair the predictions and labels, and sort them 
    # based on the largest predicted probability
    pairs = list(zip(y_pred, y_true, scanner))

    # define dictionary to save model performance in terms of AUROC and AP 
    # per scanner type
    results = {'set': partitions}
    for sensitivity_level in sensitivity_levels:
        # add items to results dictionary
        results[f'sens @ {sensitivity_level} sens'] = []
        results[f'spec @ {sensitivity_level} sens'] = []
        # select pairs with the correct scanner type
        for partition in results['set']:
            if partition == 'All':
                selected_pairs = pairs
            else:
                selected_pairs = [pair for pair in pairs if pair[2] == partition]

            # determine the position of the prediction near the desired sensitivity level
            selected_y_pred, selected_y_true, _ = list(zip(*selected_pairs))
            selected_y_pred_positives = sorted([
                pair[0] for pair in selected_pairs if pair[1] == 1
            ], reverse=True)  

            # calculate onlt if there is atleast one positive case
            if len(selected_y_pred_positives):
                # determine the starting index
                position = len(selected_y_pred_positives)*sensitivity_level
                index = int(position)

                # binarize the predictions and calculate the sensitivity and specificity
                prev_sensitivity = None
                prev_specificity = None
                while True:
                    threshold = selected_y_pred_positives[index]
                    selected_y_pred_bin = [
                        1 if pred >= threshold else 0 for pred in selected_y_pred
                    ]
                    combinations = zip(selected_y_pred_bin, selected_y_true)
                    counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                    for combination in combinations:
                        if combination == (1, 1):
                            counts['TP'] += 1
                        elif combination == (0, 0):
                            counts['TN'] += 1
                        elif combination == (1, 0):
                            counts['FP'] += 1
                        elif combination == (0, 1):
                            counts['FN'] += 1
                    new_sensitivity = counts['TP'] / (counts['TP'] + counts['FN'])
                    new_specificity = counts['TN'] / (counts['TN'] + counts['FP'])                  
                    if new_sensitivity < sensitivity_level:
                        break
                    index -= 1
                    if index < 0:
                        break
                    prev_sensitivity = new_sensitivity
                    prev_specificity = new_specificity

                if method == 'interpolate':
                    # linearly interpolate between sensitivity points
                    a = (sensitivity_level - prev_sensitivity)/(new_sensitivity-prev_sensitivity)
                    sensitivity = a*new_sensitivity + (1-a)*prev_sensitivity
                    specificity = a*new_specificity + (1-a)*prev_specificity
                
                elif method == 'select_closest':
                    # select the closest point to the specified level
                    if (abs(sensitivity_level - prev_sensitivity) 
                        < abs(sensitivity_level - new_sensitivity)):
                        sensitivity = prev_sensitivity
                        specificity = prev_specificity
                    else:
                        sensitivity = new_sensitivity
                        specificity = new_specificity
            else:
                # add uninformative result if no positive case is present in the data
                sensitivity = None
                specificity = None
        
            # store results
            results[f'sens @ {sensitivity_level} sens'].append(sensitivity)
            results[f'spec @ {sensitivity_level} sens'].append(specificity)

    return results


def bootstrapper(
    y_true: list[int], 
    y_pred: list[float], 
    scanner: list[str],
    eval_func: Callable,
    iterations: int,
    stratified: bool,
    confidence_level: float,
    row_key: str,
    seed: int,
    show_progress: bool = True,
) -> dict[str, list]:
    """
    """
    # set seed
    random.seed(seed)

    # determine the number of positive cases
    N_positive = int(sum(y_true))
    # order all cases such that the positive cases are in front
    y_true, y_pred, scanner = zip(*sorted(zip(y_true, y_pred, scanner), reverse=True))

    # initialize iterator
    if show_progress:
        iterator = tqdm(range(iterations))
    else:
        iterator = range(iterations)

    # sample with replacement with or without stratification          
    sample_results = []
    for _ in iterator:
        # generate indices
        if stratified:
            sample_indices = [random.randint(0, N_positive-1) for _ in range(N_positive)]
            sample_indices.extend([
                random.randint(N_positive, len(scanner)-1) for _ in range(len(scanner)-N_positive)
            ]) 
        else:
            sample_indices = [random.randint(0, len(scanner)-1) for _ in range(len(scanner))]
        # select cases with replacement based on generated indices
        y_true_sampled = [y_true[i] for i in sample_indices]
        y_pred_sampled = [y_pred[i] for i in sample_indices]
        scanner_sampled = [scanner[i] for i in sample_indices]
        sample_results.append(eval_func(y_true_sampled, y_pred_sampled, scanner_sampled))

    # calculate mean and confidence intervals for bootstrap samples
    bootstrap_results = {}
    rows = sample_results[0][row_key]
    bootstrap_results[row_key] = rows
    metrics = list(sample_results[0].keys())
    metrics.remove(row_key)

    for metric in metrics:
        # add items to dictionary
        bootstrap_results[f'mean {metric}'] = []
        bootstrap_results[f'{confidence_level}% CI lower {metric}'] = []
        bootstrap_results[f'{confidence_level}% CI upper {metric}'] = []
        bootstrap_results[f'N {metric}'] = []

        # loop over rows
        for i in range(len(rows)): 
            # select the values for the iterations and compute the mean and 
            # confidence interval
            values = [sample[metric][i] for sample in sample_results]
            if None in values:
                values.remove(None)
            
            # calculate the mean and confidence interval
            mean = np.mean(values)
            lower = np.quantile(values, (1-confidence_level)/2)
            upper = np.quantile(values, 1-((1-confidence_level)/2))

            # store the results
            bootstrap_results[f'mean {metric}'].append(mean)
            bootstrap_results[f'{confidence_level}% CI lower {metric}'].append(lower)
            bootstrap_results[f'{confidence_level}% CI upper {metric}'].append(upper)
            bootstrap_results[f'N {metric}'].append(len(values))

    return bootstrap_results


def evaluate_without_thresholds(
    y_true: list[int], 
    y_pred: list[float], 
    scanner: list[str],
    bins: int,
    roc_figure_path: Optional[Union[str, Path]] = None,
    pr_figure_path: Optional[Union[str, Path]] = None,
    cal_figure_path: Optional[Union[str, Path]] = None,
) -> dict[str, list]:
    """
    Evaluate predictive performance of model using threshold-independent metrics
    (Area under ROC and PR curves).

    Args:
        y_true:  True label per case.
        y_pred:  Predicted probability by the model per case.
        scanner:  Name of scanner per case.
        bins:  Number of bins used in the calibration assessment.
        roc_figure_path:  Output path for saving ROC curve figure.
        pr_figure_path:  Output path for saving PR curve figure.
        cal_figure_path:  Output path for saving calibration figure.

    Returns:
        area_results:  Results of area under the curves evaluation.
    """
    # determine what partitions and colors to use
    if scanner is None:
        partitions = ['All']
        colors = ['mediumblue']
    else:
        partitions = ['Aperio', 'Hamamatsu', 'All']
        colors = ['lightskyblue', 'dodgerblue', 'mediumblue']
    
    # define dictionary to save model performance in terms of AUROC and AP 
    # per scanner type
    results = {'set': partitions, 'AUROC': [], 'AP': [], 'ECE': []}

    # plot ROC curve and calculate AUC-ROC
    if roc_figure_path is not None:
        fig, ax = plt.subplots(figsize=(5,5))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.3)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')

    # loop over data partitions
    for partition, color in zip(partitions, colors):
        # select data for partition
        if partition == 'All':
            selected_y_pred = y_pred
            selected_y_true = y_true
        else:
            selected_y_pred = []
            selected_y_true = []
            for i, scanner_name in enumerate(scanner):
                if partition in scanner_name:
                    selected_y_true.append(y_true[i])
                    selected_y_pred.append(y_pred[i])

        # prepare ROC curve and calculate AUC-ROC
        if len(set(selected_y_true)) == 2:
            fpr, tpr, _ = roc_curve(selected_y_true, selected_y_pred)
            auroc = roc_auc_score(selected_y_true, selected_y_pred)
            results['AUROC'].append(auroc)
            if roc_figure_path is not None:
                plt.plot(fpr, tpr, label=f'ROC curve {partition} (AUC: {auroc:0.3f})', 
                            lw=2, color=color)
        else:
            results['AUROC'].append(None)
    
    if roc_figure_path is not None:
        plt.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(roc_figure_path)
        plt.close()

    # plot precision-recall (PR) curve and calculate AUC-PR
    if pr_figure_path is not None:
        fig, ax = plt.subplots(figsize=(5,5))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        # format ticks
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", direction="in", length=5, width=1.3)
        ax.tick_params(which='minor', axis="both", direction="in", 
                        right='on', top='on', length=2.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_major_locator(MultipleLocator(0.200))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # format axes
        offset = 0.001
        plt.xlim([0-offset, 1+offset])
        plt.ylim([0-offset, 1+offset])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

    # loop over data partitions
    for partition, color in zip(partitions, colors):
        # select data for partition
        if partition == 'All':
            selected_y_pred = y_pred
            selected_y_true = y_true
        else:
            selected_y_pred = []
            selected_y_true = []
            for i, scanner_name in enumerate(scanner):
                if partition in scanner_name:
                    selected_y_true.append(y_true[i])
                    selected_y_pred.append(y_pred[i])

        # prepare PR curve and calculate AUC-PR
        if len(set(selected_y_true)) == 2:
            precision, recall, _ = precision_recall_curve(selected_y_true, selected_y_pred)
            ap = average_precision_score(selected_y_true, selected_y_pred)
            results['AP'].append(ap)
            if pr_figure_path is not None:
                plt.plot(recall, precision, label=f'PR curve {partition} (AUC: {ap:0.3f})', 
                        lw=2, color=color)
        else:
            results['AP'].append(None)

    if pr_figure_path is not None:
        plt.legend(loc=4, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
        plt.savefig(pr_figure_path)
        plt.close()


    # loop over data partitions
    for partition in partitions:
        # select data for partition
        if partition == 'All':
            selected_y_pred = y_pred
            selected_y_true = y_true
        else:
            selected_y_pred = []
            selected_y_true = []
            for i, scanner_name in enumerate(scanner):
                if partition in scanner_name:
                    selected_y_true.append(y_true[i])
                    selected_y_pred.append(y_pred[i])

        # assign each case to a predicted probability bin and store the label
        labels = {i: [] for i in range(bins)}
        probabilities = {i: [] for i in range(bins)}
        for pred, true in zip(selected_y_pred, selected_y_true):
            index = min(int(pred*bins), bins-1)
            probabilities[index].append(pred)
            labels[index].append(true)
        
        # calculate the faction of true labels for each bin
        count = [len(labels[i]) for i in range(bins)]
        valid = [True if count[i] > 0 else False for i in range(bins)]
        true_prob = [sum(labels[i])/len(labels[i]) if len(labels[i]) > 0 else None for i in range(bins)]
        pred_prob = [sum(probabilities[i])/len(probabilities[i]) if len(probabilities[i]) > 0 else None for i in range(bins)]
        ece = sum([(count[i]/sum(count))*abs(true_prob[i]-pred_prob[i]) for i in range(bins) if valid[i]])
        results['ECE'].append(ece)

        # plot confidence calibration
        if cal_figure_path is not None and partition == 'All':
            fig, ax = plt.subplots(1, 2, figsize=(10.5, 5))

            for i in range(2):
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[i].spines[axis].set_linewidth(1.5)

                # format ticks
                ax[i].tick_params(bottom=True, top=True, left=True, right=True)
                ax[i].tick_params(axis="both", direction="in", length=5, width=1.3)
            ax[0].xaxis.set_major_locator(MultipleLocator(0.200))
            ax[0].yaxis.set_major_locator(MultipleLocator(0.200))
            ax[1].xaxis.set_major_locator(MultipleLocator(0.200))

            # format axes
            offset = 0
            ax[0].set_xlim([0-offset, 1+offset])
            ax[0].set_ylim([0-offset, 1+offset])
            ax[0].set_xlabel('Confidence')
            ax[0].set_ylabel('Accuracy')
            ax[1].set_xlim([0-offset, 1+offset])
            ax[1].set_xlabel('Confidence')
            ax[1].set_ylabel('Count')

            ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], height=true_prob, 
                        width=[1/bins]*bins, edgecolor='black', linewidth=1.3, 
                        color='gray', label='Outputs')
            ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], 
                        height=[pred_prob[i]-true_prob[i] for i in range(bins)], 
                        width=[1/bins]*bins, bottom=true_prob, color='red', 
                        edgecolor='firebrick', linewidth=1.3, hatch='/', alpha=0.3,
                        label='Gap')
            ax[0].bar(x=[(i+0.5)/bins for i in range(bins)], 
                        height=[pred_prob[i]-true_prob[i] for i in range(bins)], 
                        width=[1/bins]*bins, bottom=true_prob, edgecolor='firebrick', 
                        linewidth=1.3, fill=False)
            ax[0].plot([0, 1], [0, 1], ls='--', color='black', lw=0.75)
            ax[0].set_title(f'Expected Calibration Error (ECE): {ece:0.2f}')
            ax[0].legend(loc=2, borderaxespad=1.2, fancybox=False, edgecolor='black', fontsize=10)
            
            ax[1].bar(x=[(i+0.5)/bins for i in range(bins)], height=count, 
                        width=[1/bins]*bins, edgecolor='black', linewidth=1.3, 
                        color='gray')
            plt.savefig(cal_figure_path)
            plt.close()

    return results