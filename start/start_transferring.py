"""
Start transferring data.
"""

import json
from ast import literal_eval

import pandas as pd

from pipeline.transferring_service import TransferringService

# define paths
pipeline_config_path = 'pipeline.json'
dataset_path = 'dataset.xlsx'

if __name__ == '__main__':

    # load config
    with open(pipeline_config_path, 'r') as f:
        config = json.loads(f.read())

    # load dataset
    df = pd.read_excel(dataset_path)
    if 'weight' in df.columns:
        df['weight'] = [float(item) for item in list(df['weight'])]
    if 'paths' in df.columns:
        df['paths'] = [literal_eval(item) for item in list(df['paths'])]

    # configure transfer service and start the data transfer
    service = TransferringService(
        config=config,
        df=df,
    )
    service.establish_connection()
    service.start()