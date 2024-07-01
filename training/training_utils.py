"""
Utility functions for model training.
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        feature_information: dict[int, dict[str, Any]],
        patient_vector_func: Callable,
        label_func: Callable,
        weight_func: Optional[Callable] = None,
        length: Optional[int] = None,
        only_first_variant: bool = False,
        interpolate_features: bool = False,
        interpolation_sigma: float = 0.0,
        section_dropout_prob: float = 0.0,
    ) -> None:
        """
        Initializes dataset instance to load feature vectors.

        Args:
            feature_information:  Dictionary with feature information.
            patient_vector_func:  Function to create the vector with the encoded
                patient information.
            label_func:  Function to create label based on specimen information.
            weight_func:  Function to calculate weight based on specimen information.
            length:  Number of items in the dataset (can be set arbitrarily for
                training without specifying epochs).
            only_first_variant:  Indicates whether only the first from the list
                of feature variants should be selected. This is relevant in case
                feature variants based on augmented images are available, but
                only the features based on the original images should be selected
                (which are in the first position if `first_not_augmented` was equal
                to True in the augmentation configuration of the pipeline.)
            interpolate_features:  Indicates whether returned features are created 
                by interpolating between two sequences of feature variants.
            interpolation_sigma:  Standard deviation for normal distribution which
                is used for sampling the contribution factor of the augmented
                feature vector in the interpolation.
            section_dropout_prob:  Probability for randomly dropping out cross-section.
        """
        # define instance attributes
        self.feature_information = feature_information
        self.patient_vector_func = patient_vector_func
        self.label_func = label_func
        self.weight_func = weight_func
        self.length = length
        self.only_first_variant = only_first_variant
        self.interpolate_features = interpolate_features
        self.interpolation_sigma = interpolation_sigma
        self.section_dropout_prob = section_dropout_prob

        # get list with specimen indices
        self.indices = list(self.feature_information.keys()) 

        # check whether the arguments are valid
        if self.only_first_variant and self.interpolate_features:
            raise ValueError('Invalid combination of `only_first_variant` equals '
                             'True and `interpolate_features` equals True.')
     
    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        else:
            return len(self.indices)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        """
        Load a tile and prepare it.

        Args:
            index:  Index for selecting tile from the dataset.

        Returns:
            features:  Concatenated feature vectors.
            patient_vector:  Vector with patient information encoded.
            positions:  Contatenated position vectors corresponding 
                to the feature vectors.
            label:  Specimen label.
        """
        # get specimen information and number of feature variants
        specimen_information = self.feature_information[self.indices[index % len(self.indices)]]
        N_feature_variants = len(specimen_information['feature_paths'])

        # check if interpolation between feature variants should be performed
        if self.interpolate_features and not self.only_first_variant:
            # randomly select two variants
            feature_dicts = []
            feature_sets = []
            for variant_index in [0, random.randint(1, N_feature_variants-1)]:
                feature_dicts.append(torch.load(specimen_information['feature_paths'][variant_index]))

                # concatenate feature vectors
                feature_sets.append(torch.concat(
                    tensors=[item['feature'][None, ...] for item in feature_dicts[-1]], 
                    dim=0,
                ))
            # interpolate feature vectors
            if self.interpolation_sigma > 0.0:
                factor = min(abs(random.gauss(0, self.interpolation_sigma)), 1)
            else:
                factor = self.interpolation_sigma
            features = (feature_sets[0]*(1-factor)) + (feature_sets[1]*factor)

            # select a single feature dictionary for getting the positions,
            # patient information, and label
            feature_dict = feature_dicts[0]
        else:
            if self.only_first_variant:
                # select the first variant (possibly based on the original image 
                # without augmentation)
                feature_dict = torch.load(specimen_information['feature_paths'][0])
            else:
                # randomly select a single variant
                variant_index = random.randint(0, N_feature_variants-1)
                feature_dict = torch.load(specimen_information['feature_paths'][variant_index])
       
            # concatenate feature vectors
            features = torch.concat(
                tensors=[item['feature'][None, ...] for item in feature_dict], 
                dim=0,
            )

        # concatenate position vectors
        positions = torch.concat(
            tensors=[torch.tensor(item['position'])[None, ...] for item in feature_dict], 
            dim=0,
        )  

        # randomly drop out cross-sections
        if self.section_dropout_prob > 0.0:
            # select the cross-sections to drop out
            sections_indices = positions[:, 0].tolist()
            sections_dict = {
                i: random.random() > self.section_dropout_prob for i in set(sections_indices)
            }
            # set a random cross-section to True if all are False (i.e., dropped out)
            if True not in set(sections_dict.values()):
                sections_dict[random.randint(0, len(sections_dict)-1)] = True
            
            # get the indices for the features of the selected cross-sections
            selection = [i for i, index in enumerate(sections_indices) if sections_dict[index]]
            # apply the selection
            features = features[selection, :]
            positions = positions[selection, :]       

        # create a patient vector and define variables for the weight and label
        patient_vector = self.patient_vector_func(specimen_information)
        label = self.label_func(specimen_information)
        if self.weight_func is None:
            weight = 1
        else:
            weight = self.weight_func(specimen_information)

        return features, patient_vector, positions, weight, label


# helper function for reading feature information and creating patient vector
def read_feature_information(
    feature_information_path: Union[str, Path], 
    feature_directory: Optional[Union[str, Path]] = None, 
) -> dict[int, dict[str, Any]]:
    """
    Reads feature information

    Args:
        feature_information_path:  Path to feature with tile information.
        image_directory:  Path to folder where all images are stored.

    Returns:
        data_dict:  Dictionary with specimen information, feature paths,
            and corresponding tile position information.
    """
    # read feature information from file
    with open(feature_information_path, 'r') as f:
        lines = f.readlines()

    # check if the number of lines is a multiple of three
    if len(lines) % 3 != 0:
        raise ValueError(f'The number of lines in {feature_information_path} '
                         'must be a multiple of three.')

    # initialize dictionary for storing the feature information 
    data_dict = {}
    # loop over feature information per specimen
    for i in range(int(len(lines)/3)):
        image_filenames = eval(lines[i*3])
        specimen_information = json.loads(lines[i*3+1])
        feature_filenames = eval(lines[i*3+2])
        if feature_directory is not None:
            feature_paths = [(Path(feature_directory)/name) for name in feature_filenames]
        else:
            feature_paths = feature_filenames

        # add information to dictionary and indices to list
        specimen_index = specimen_information['specimen_index']
        if specimen_index in data_dict:
            data_dict[specimen_index]['feature_paths'].extend(feature_paths)
        else:
            data_dict[specimen_index] = {
                **specimen_information,
                'images': image_filenames,
                'feature_paths': feature_paths,
            }

    return data_dict


def get_patient_vector(specimen_information: dict[str, Any]) -> torch.Tensor:
    """
    Get vector with patient information encoded.

    Args:
        specimen_information:  Dictionary with specimen information.

    Returns:
        patient_vector:  Vector with patient information encoded.
    """
    # check if the required keys are available in the specimen information
    for key in ['sex', 'age', 'location']:
        if key not in specimen_information:
            raise ValueError(f'Patient information not specified for specimen: {key}.')
    
    patient_vector = []
    # add sex information
    if specimen_information['sex'].lower() in ['m', 'male']:
        patient_vector.append(0)
    else:
        patient_vector.append(1)
    
    # add age information
    patient_vector.append(specimen_information['age']/100)

    # add location information
    location = specimen_information['location'].lower()
    for group in ['head', 'neck', 'trunk', 'extremity', 'palm-sole', 'hand-foot']:
        if location == group:
            patient_vector.append(1)
        else:
            patient_vector.append(0)
    
    # convert list to vector
    patient_vector = torch.tensor(patient_vector)

    return patient_vector


def get_label(specimen_information: dict[str, Any]) -> torch.Tensor:
    """
    Get specimen label.

    Args:
        specimen_information:  Dictionary with specimen information.

    Returns:
        label: Specimen label.
    """
    if specimen_information['label'] == True:
        label = torch.tensor([0, 1])
    elif specimen_information['label'] == False:
        label = torch.tensor([1, 0])   
    else:
        raise ValueError('Invalid specimen label.')

    return label         


def get_scanner_weight(specimen_information: dict[str, Any], 
                       weight_dict: dict[str, float]) -> torch.Tensor:
    """
    Get weight depending on the scanner type inferred from the file extension.

    Args:
        specimen_information:  Dictionary with specimen information.

    Returns:
        weight: Scanner weight.
    """
    # get all unique WSI file extensions and determine the scanner type
    unique_extensions = {name.split('.')[-1] for name in specimen_information['images']}
    scanners = []
    if 'dcm' in unique_extensions:
        scanners.append('Aperio')
    if 'ndpi' in unique_extensions:
        scanners.append('Hamamatsu')     
    scanner = ' & '.join(scanners)   

    # get weight from weight dictionary based on scanner name
    weight = weight_dict[scanner]

    return weight


def get_diagnosis_weight(specimen_information: dict[str, Any], 
                         weight_dict: dict[str, float]) -> torch.Tensor:
    """
    Get weight depending on the lesion diagnosis.

    Args:
        specimen_information:  Dictionary with specimen information.

    Returns:
        weight: Diagnosis weight.
    """
    # get all unique WSI file extensions and determine the scanner type
    label = specimen_information['label']
    diagnosis = specimen_information['diagnosis']
    for replacement in [('(', ''), (')', ''), ('+', ' '), ('/', ' '), ('&', ' ')]:
        diagnosis = diagnosis.replace(*replacement)
    codes = [code for code in list(set(diagnosis.split(' '))) if code != '']
    keys = [f'{code}_{int(label)}' for code in codes]

    # loop over codes and check whether a weight is available
    weights = []
    for key in keys:
        if key in weight_dict:
            weights.append(weight_dict[key])
    # take the maximum weight in case more than one are available, 
    # otherwise use a weighting of 1
    weight = max(weights) if len(weights) else 1

    return weight


class FocalLoss(nn.Module):

    def __init__(
        self,
        sigmoid: bool = False, 
        gamma: float = 0.0,
        class_weights: Optional[Union[list[float], torch.Tensor]] = None,
    ) -> None:
        """
        Initialize focal loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the focal loss is equal to the 
                cross-entropy loss.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, 
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  Weighting factor for each item in the batch. 

        Returns:
            loss:  Focal loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # convert class weights from list to tensor if necessary
        if isinstance(self.class_weights, list):
            self.class_weights = torch.Tensor(self.class_weights)
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the pixel-wise predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
            log_y_pred = F.logsigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)
            log_y_pred = F.log_softmax(logit, dim=1)

        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        log_y_pred_flat = log_y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the pixelwise cross-entropy, focal weight, and pixelwise focal loss
        pixelwise_CE = -(log_y_pred_flat * y_true_flat)
        focal_weight = (1-(y_true_flat * y_pred_flat))**self.gamma
        pixelwise_focal_loss = focal_weight * pixelwise_CE

        # calculate the class-separated focal loss
        class_separated_focal_loss = torch.mean(pixelwise_focal_loss, dim=-1)
        
        # multiply the dice score per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_focal_loss[:, i] *= class_weight
        instance_loss = torch.sum(class_separated_focal_loss, dim=1)

        # if a weight was specified, multiple each item in the batch with
        if weight is not None:
            instance_loss *= weight

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss