import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from scipy.stats import norm
from sdmetrics.visualization import get_column_pair_plot
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot, get_column_pair_plot
from sdv.metadata import SingleTableMetadata
from table_evaluator import TableEvaluator

# Create argument parser
parser = argparse.ArgumentParser(description='Visualise comparison of real and synthetic data and open in browser')

# Add arguments
# Files have to be in /file folder
parser.add_argument('--input', '-i', metavar="[FILENAME]", type=str, required=True, help='[REQUIRED] Provide the name of the input file that contains real data without the extension. The input file has to be present in the "/input" folder.')
parser.add_argument('--output', '-o', metavar="[FILENAME]", type=str, required=True, help='[REQUIRED] Provide the name of the output file that contains synthetic data without the extension. The output file has to be present in the "/output" folder.')
parser.add_argument('--losses', '-l', metavar="[FILENAME]", type=str, required=True, help='[REQUIRED] Provide the name of the losses file that contains the log of losses from training. The losses file has to be present in the "/logs" folder.')

args = parser.parse_args()

# ---Create folders
# Create evaluation folder if it does not exist
evaluation_folder = 'evaluation'
os.makedirs(evaluation_folder, exist_ok=True)

# Create a html folder if it does not exist (subfolder)
webpages_folder = 'html'
os.makedirs(os.path.join(evaluation_folder, webpages_folder), exist_ok=True)
html_file_path = os.path.join(evaluation_folder, webpages_folder)

# Create a jpg folder if it does not exist (subfolder)
images_folder = 'images'
os.makedirs(os.path.join(evaluation_folder, images_folder), exist_ok=True)
images_file_path = os.path.join(evaluation_folder, images_folder)


# ---Load real and synthetic data
real_data_folder = 'input'
real_data_file_name = args.input
real_data_file_name += '.csv'
real_data_file_path = os.path.join(real_data_folder, real_data_file_name)
real_data = pd.read_csv(real_data_file_path)

synthetic_data_folder = 'output'
synthetic_data_file_name = args.output
synthetic_data_file_name += '.csv'
synthetic_data_file_path = os.path.join(synthetic_data_folder, synthetic_data_file_name)
synthetic_data = pd.read_csv(synthetic_data_file_path)

# ---Load logs of losses
logs_folder = 'logs'
model_log_file_name = args.losses
model_log_file_name += '.csv'
loss_values_path  = os.path.join(logs_folder, model_log_file_name)
loss_values = pd.read_csv(loss_values_path)

# Create evaluation folder if it does not exist
evaluation_folder = 'evaluation'
os.makedirs(evaluation_folder, exist_ok=True)

# Create debugging folder for visualisations if it does not exist
debug_visualisation_name = 'debug'
os.makedirs(os.path.join(evaluation_folder, debug_visualisation_name), exist_ok=True)
debug_folder = os.path.join(evaluation_folder, debug_visualisation_name)

# Create a html folder if it does not exist (subfolder)
webpages_folder = 'html'
os.makedirs(os.path.join(debug_folder, webpages_folder), exist_ok=True)
html_file_path = os.path.join(debug_folder, webpages_folder)

# Create a jpg folder if it does not exist (subfolder)
images_folder = 'images'
os.makedirs(os.path.join(debug_folder, images_folder), exist_ok=True)
images_file_path = os.path.join(debug_folder, images_folder)

#---
# Rename columns
# Rename column 'Total.Backward.Packets' to 'Total.Bwd.Packets'
real_data = real_data.rename(columns={'Total.Backward.Packets': 'Total.Bwd.Packets'})
synthetic_data = synthetic_data.rename(columns={'Total.Backward.Packets': 'Total.Bwd.Packets'})

# Define metadata
metadata_dict = {
    "columns": {
        "Timestamp": {
            "sdtype": "datetime",
            "datetime_format": "%Y-%m-%d %H:%M:%S"
        },
        "Source.IP": {
            "sdtype": "categorical"
        },
        "Source.Port": {
            "sdtype": "categorical"
        },
        "Destination.IP": {
            "sdtype": "categorical"
        },
        "Destination.Port": {
            "sdtype": "categorical"
        },
        "Protocol": {
            "sdtype": "categorical"
        },
        "Flow.Duration": {
            "sdtype": "numerical"
        },
        "Total.Fwd.Packets": {
            "sdtype": "numerical"
        },
        "Total.Bwd.Packets": {
            "sdtype": "numerical"
        },
        "Total.Length.of.Fwd.Packets": {
            "sdtype": "numerical"
        },
        "Total.Length.of.Bwd.Packets": {
            "sdtype": "numerical"
        }
    },
    "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1"
}
metadata = SingleTableMetadata.load_from_dict(metadata_dict)
