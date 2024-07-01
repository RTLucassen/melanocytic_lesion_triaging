"""
Evaluate performance of model ensemble based on spreadsheets with predictions
from individual models.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluation_service import (
    predict_per_class,
    evaluate_with_thresholds,
    evaluate_without_thresholds,
    specificity_at_sensitivity,
    bootstrapper,
)

# define experiment directory and variants
experiment_directory = ''
variants = ['test']

# define other settings
thresholds = np.arange(0, 1.001, 0.005)
sensitivity_levels = [0.99, 0.98, 0.95]
bootstrap_iterations = 10000
confidence_level = 0.95
method = 'interpolate'
stratified = True
bins = 10
seed = 1
image_extension = '.png'


# loop over variants
experiment_directory = Path(experiment_directory)
for variant in variants:
    # define path to excel sheet with ensembled model predictions
    path = experiment_directory / f'results_ensemble_{variant}.xlsx'

    # read the excel sheet with predictions
    predictions_df = pd.read_excel(path)
    
    # get the predictions and other information
    y_true = predictions_df['y_true']
    y_pred = predictions_df['y_pred_mean']
    diagnosis = predictions_df['diagnosis']
    scanner = predictions_df['scanner']

    # perform per class evaluation
    class_predictions = predict_per_class(y_true, y_pred, diagnosis)
    # perform threshold-dependent evaluation
    threshold_results = evaluate_with_thresholds(y_true, y_pred, thresholds)
    if sensitivity_levels is not None:
        spec_at_sens_results = specificity_at_sensitivity(y_true, y_pred, scanner, 
                                                          sensitivity_levels, method=method)
    # perform threshold-independent evaluation 
    area_results = evaluate_without_thresholds(y_true, y_pred, scanner, 
        roc_figure_path=experiment_directory / f'ROC_curve_{variant}{image_extension}',
        pr_figure_path=experiment_directory / f'PR_curve_{variant}{image_extension}',
        cal_figure_path=experiment_directory / f'Calibration_{variant}{image_extension}',
        bins=bins,
    )

    # perform bootstrapping for evaluation
    if bootstrap_iterations is not None:
        if sensitivity_levels is not None:
            bootstrap_spec_at_sens_results = bootstrapper(
                y_true=y_true, 
                y_pred=y_pred, 
                scanner=scanner, 
                eval_func=partial(
                    specificity_at_sensitivity, 
                    sensitivity_levels=sensitivity_levels,
                    method=method,
                ), 
                iterations=bootstrap_iterations, 
                stratified=stratified,
                confidence_level=confidence_level, 
                row_key='set', 
                seed=seed,
            )
        bootstrap_area_results = bootstrapper(
            y_true=y_true, 
            y_pred=y_pred, 
            scanner=scanner, 
            eval_func=partial(evaluate_without_thresholds, bins=bins), 
            iterations=bootstrap_iterations, 
            stratified=stratified,
            confidence_level=confidence_level, 
            row_key='set', 
            seed=seed,
        )

    # convert dictionaries to dataframes
    class_predictions_df = pd.DataFrame.from_dict(class_predictions)
    threshold_results_df = pd.DataFrame.from_dict(threshold_results)
    if sensitivity_levels is not None:
        spec_at_sens_results_df = pd.DataFrame.from_dict(spec_at_sens_results)
    area_results_df = pd.DataFrame.from_dict(area_results)
    if bootstrap_iterations is not None:
        if sensitivity_levels is not None:
            bootstrap_spec_at_sens_results_df = pd.DataFrame.from_dict(bootstrap_spec_at_sens_results)
        bootstrap_area_results_df = pd.DataFrame.from_dict(bootstrap_area_results)

    # save evaluation results in spreadsheet
    with pd.ExcelWriter(path) as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        class_predictions_df.to_excel(writer, sheet_name='Class predictions', index=False)
        threshold_results_df.to_excel(writer, sheet_name='Results (threshold)', index=False)
        if sensitivity_levels is not None:
            spec_at_sens_results_df.to_excel(writer, sheet_name='Results (spec @ sens)', index=False)
        area_results_df.to_excel(writer, sheet_name='Results (area)', index=False)
        if bootstrap_iterations is not None:
            if sensitivity_levels is not None:
                bootstrap_spec_at_sens_results_df.to_excel(writer, sheet_name='Bootstrap results (spec @ sens)', index=False)
            bootstrap_area_results_df.to_excel(writer, sheet_name='Bootstrap results (area)', index=False)