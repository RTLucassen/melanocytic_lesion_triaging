"""
Simulation experiment of how the implementation of a triaging system would affect 
the workflow of a pathology lab. 
"""

import random
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_indices(
    N: int, 
    a: int = 0, 
    b: Optional[int] = None, 
    replacement: bool = False,
) -> list[int]:
    """
    Sample N random indices in the range [a, b] with or without replacement. 
    """
    # set the maximum to the number of samples
    if b is None:
        b = N-1
    # check if the minimum is not larger than the maximum
    if a > b:
        raise ValueError('The argument for `a` must be equal to or larger than `b`.')
    # get the indices either with or without remplacement
    if replacement:
        indices = [random.randint(a, b) for _ in range(N)]
    else:
        if N > b+1-a:
            raise ValueError('The number of indices cannot be smaller than the '
                             'length of the range [a, b].')
        indices = list(range(a, b+1))
        random.shuffle(indices)
        indices = indices[:N]
    
    return indices


def baseline(df: pd.DataFrame, pathologists: list[tuple[int, bool]]) -> dict:
    """
    Random case assignment as baseline for the simulation experiment.
    
    Args:
        df:  Dataframe with predicted probability ('y_pred') 
            and true label ('y_true') per case.
        pathologists:  List with 2-tuple per pathologist, indicating the index 
            and whether the pathologist is an expert.

    Returns:
        results:  Dictionary with number of high and low complexity cases 
            assigned to each pathologist using random case assignment.
    """
    # initialize dictionary to store the result
    result = {pathologist: {'positive': 0, 'negative': 0} for pathologist in pathologists}
    # loop over all cases
    for i in range(len(df)):
        pathologist = pathologists[i % len(pathologists)]
        case = df_selected.iloc[i]
        # assign the case to the results
        if case['y_true'] == 0:
            result[pathologist]['negative'] += 1
        elif case['y_true'] == 1:
            result[pathologist]['positive'] += 1
        else:
            ValueError(f'Unexpected value for `y_true`.')
        
    return result


def AI_based_triaging(df: pd.DataFrame, pathologists: list[tuple[int, bool]]) -> dict:
    """
    AI-based triaging for case assignment in the simulation experiment.
    
    Args:
        df:  Dataframe with predicted probability ('y_pred_mean') 
            and true label ('y_true') per case.
        pathologists:  List with 2-tuple per pathologist, indicating the index 
            and whether the pathologist is an expert.

    Returns:
        results:  Dictionary with number of high and low complexity cases 
            assigned to each pathologist using AI-based triaging.
    """
    # initialize dictionary to store the result
    result = {pathologist: {'positive': 0, 'negative': 0} for pathologist in pathologists}
    # sort based on predicted probability
    df = df.sort_values(by=['y_pred_mean'], ascending=False)
    i_top = 0
    i_bottom = N_cases-1
    for _ in range(ceil(N_cases/len(pathologists))):
        for pathologist in pathologists:
            if i_top <= i_bottom:
                # check if the pathologist is an expert or not
                if pathologist[1] == True:
                    case = df.iloc[i_top]
                    i_top += 1
                else:
                    case = df.iloc[i_bottom]
                    i_bottom -= 1
                # assign the case to the result
                if case['y_true'] == 0:
                    result[pathologist]['negative'] += 1
                elif case['y_true'] == 1:
                    result[pathologist]['positive'] += 1
                else:
                    ValueError(f'Unexpected value for `y_true`.')

    return result


# define the path and sheet name
paths = [
    r"melanocytic_lesion_triaging\selected_models\exp068_ensemble\results_ensemble_test.xlsx",
    r"melanocytic_lesion_triaging\selected_models\exp068_ensemble\results_ensemble_test_uncertainty.xlsx",
]
sheet_name = 'Predictions'

# load spreadsheets, separate into positive and negative cases, and save the length
dfs = [pd.read_excel(path, sheet_name=sheet_name) for path in paths]
df = pd.concat(dfs)
df_pos = df[df['y_true'] == 1]
df_neg = df[df['y_true'] == 0]
N_total_cases = len(df)
N_total_pos_cases = len(df_pos)
N_total_neg_cases = len(df_neg)

# define case study settings
N_general_pathologists = 4
N_expert_pathologists = 1
N_cases = 500
neg_to_pos_ratio = None   # if None, use the ratio in the dataset 
sample_with_replacement = True
iterations = 10000
seed = 1

# other settings
confidence_level = 0.95

# define methods
methods = {
    'baseline': baseline, 
    'AI-based triaging': AI_based_triaging,
}

# ASSUMPTIONS:
# - each pathologist is assumed to get assigned the same number of cases
# - the expert pathologists are assigned the cases with the largest predicted probabilities

# CAVEATS:
# - a subset of the positive cases are consultation cases, which are generally
#   already directly assigned to the expert pathologist.


if __name__ == '__main__':

    comparison = {}
    for name, method in methods.items():
        # set seed
        random.seed(seed)

        # prepare pathologists
        pathologists = []
        samples = {}
        for times, expert in zip([N_general_pathologists, N_expert_pathologists], [False, True]):
            for _ in range(times):
                pathologists.append((len(pathologists), expert))
                samples[pathologists[-1]] = {'positive': [], 'negative': []}

        for j in tqdm(range(iterations)):        
            # randomize the order of pathologists
            random.shuffle(pathologists)

            # select the cases
            if neg_to_pos_ratio is None:
                # randomly select a set of cases from the dataset
                indices = get_indices(N_cases, 0, N_total_cases-1, sample_with_replacement)
                df_selected = df.iloc[indices]
            else:
                # determine the number of positive and negative cases
                N_neg_cases = round(N_cases*(neg_to_pos_ratio/(neg_to_pos_ratio+1)))
                N_pos_cases = N_cases-N_neg_cases
                # randomly select a set of positive cases from the dataset
                pos_indices = get_indices(N_pos_cases, 0, N_total_pos_cases-1, sample_with_replacement)
                df_pos_selected = df_pos.iloc[pos_indices]
                # randomly select a set of negative cases from the dataset
                neg_indices = get_indices(N_neg_cases, 0, N_total_neg_cases-1, sample_with_replacement)
                df_neg_selected = df_neg.iloc[neg_indices]
                # combine selected positive and negative cases and randomize the order
                df_selected = pd.concat([df_pos_selected, df_neg_selected])
                df_selected = df_selected.iloc[get_indices(N_cases)]
           
            # assign the cases to the pathologists (with or without triaging model)
            result = method(df_selected, pathologists)
            for pathologist, sample in result.items():
                for group in ['positive', 'negative']:
                    samples[pathologist][group].append(sample[group])

        # calculate the mean and confidence interval per pathologist
        pathologist_results = {}
        for key, values in samples.items():
            pathologist_results[key] = {}
            for category in ['positive', 'negative']:
                pathologist_results[key][category] = {
                    'mean': np.mean(values[category]),
                    f'{confidence_level*100:0.0f}% CI': (
                        np.quantile(values[category], (1-confidence_level)/2),
                        np.quantile(values[category], 1-((1-confidence_level)/2)),
                    ),
                }
        # calculate the mean and confidence interval per expertise level
        expertise_results = {}
        for expert in [True, False, 'all']:
            expertise_results[expert] = {'positive': [], 'negative': []}
            for key, values in samples.items():
                if expert == 'all' or key[1] == expert:
                    for category in ['positive', 'negative']:
                        if category not in expertise_results[expert]:
                            expertise_results[expert][category] = []
                        expertise_results[expert][category].extend(values[category])
        
        for expert in [True, False, 'all']:
            for category in ['positive', 'negative']:
                values = expertise_results[expert][category]
                expertise_results[expert][category] = {
                    'mean': np.mean(values),
                    f'{confidence_level*100:0.0f}% CI': (
                        np.quantile(values, (1-confidence_level)/2),
                        np.quantile(values, 1-((1-confidence_level)/2)),
                    ),
                }
        # save results
        comparison[name] = {
            'samples': samples,
            'pathologist_results': pathologist_results,
            'expertise_results': expertise_results,
        }        

    for name, results in comparison.items():
        print('\nMethod:', name)
        for key, value in results['pathologist_results'].items():
            if key[1] == True:
                print(f'{key[0]} - Expert pathologist:', value)
            else:
                print(f'{key[0]} - General pathologist:', value)
        print('')
        for key, value in results['expertise_results'].items():
            if key == 'all':
                print(f'All pathologist:', value)
            elif key == True:
                print(f'Expert pathologist:', value)
            else:
                print(f'General pathologists:', value)

    # determine the reduction in the number of referral cases
    referrals = {}
    for name, results in comparison.items():
        counts = [0]*iterations
        for pathologist, samples in results['samples'].items():
            if pathologist[1] == False:
                for i, sample in enumerate(samples['positive']):
                    counts[i] += sample
        referrals[name] = counts

    pairs = [('baseline', 'AI-based triaging')]
    for pair in pairs:
        reduction = [a-b for a, b in zip(referrals[pair[0]], referrals[pair[1]])]
        mean_reduction = np.mean(reduction)
        CI_lower_reduction = np.quantile(reduction, (1-confidence_level)/2)
        CI_upper_reduction = np.quantile(reduction, 1-((1-confidence_level)/2))
        print('Assuming all general pathologists identify all the remaining positive '
              'cases and refer them to the expert pathologists,\n the mean reduction '
              f'of case examinations by general pathologist equals {mean_reduction:0.2f} '
              f'[{confidence_level*100:0.0f}% CI: {CI_lower_reduction:0.2f}-{CI_upper_reduction:0.2f}] '
              f'per {N_cases} cases.'
        )