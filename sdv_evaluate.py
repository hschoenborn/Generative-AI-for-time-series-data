import argparse
import datetime
import json
import os
import pandas as pd

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata

# Get current timestamp for unique identifiers
current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create argument parser
parser = argparse.ArgumentParser(description='Train CTGAN models and generate synthetic datasets that closely resemble real datasets.')

# Add arguments for argument parser
parser.add_argument('--input', '-i', metavar="[FILENAME]", type=str, required=True, help='[REQUIRED] Provide the name of the input file that contains real data without the extension. The input file has to be present in the "/input" folder.')
parser.add_argument('--output', '-o', metavar="[FILENAME]", type=str, required=True, help='[REQUIRED] Provide the name of the output file that contains synthetic data without the extension. The input file has to be present in the "/input" folder.')
args = parser.parse_args()

# Load real and synthetic data
real_data_folder = 'input'
real_data_file_name = args.input
real_data_file_name += '.csv'
real_data_file_path = os.path.join(real_data_folder, real_data_file_name)
real_data = pd.read_csv(real_data_file_path)

synth_data_folder = 'output'
synth_data_file_name = args.output
synth_data_file_name += '.csv'
synth_data_file_path = os.path.join(synth_data_folder, synth_data_file_name)
synth_data = pd.read_csv(synth_data_file_path)

# Manually create metadata that describes our real dataset
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
        "Total.Backward.Packets": {
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

# --- 4. Evaluation
# Diagnostic report (evaluates stucture)
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synth_data,
    metadata=metadata
)

# Data quality report (evaluates content)
quality_report = evaluate_quality(
    real_data,
    synth_data,
    metadata
)

# Save quality report to .txt file
evaluation_folder = 'evaluation'
os.makedirs(evaluation_folder, exist_ok=True)
evaluation_file_name = f"{real_data_file_name.split('.')[0]}_{synth_data_file_name.split('.')[0]}_{current_timestamp}.txt"
evaluation_file_path = os.path.join(evaluation_folder, evaluation_file_name)
with open(evaluation_file_path, 'w') as file:
    file.write(str(diagnostic.get_info()) + '\n')
    file.write(str(quality_report.get_info()) + '\n')
    file.write("Real data path: " + real_data_file_path + '\n')
    file.write("Synthetic data path: " + synth_data_file_path + '\n')
    file.write("Diagnostic score: " + str(float(diagnostic.get_score()) * 100) + '%\n')
    file.write("Quality score: " + str(float(quality_report.get_score()) * 100) + '%\n')
