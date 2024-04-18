import argparse
import datetime
import os

from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot, get_column_pair_plot
from sdv.datasets.local import load_csvs
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata

# Get current timestamp for unique identifiers
current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create argument parser
parser = argparse.ArgumentParser(description='Generate synthetic data and save it to a CSV file')

# Add arguments
# Files have to be in /file folder
parser.add_argument('--generate', '-g', type=int, required=True,
                    help='Specify how many rows of synthetic data to generate')
parser.add_argument('--model', '-mdl', type=str, required=True,
                    help='Specify which model to use for synthetic data generation')  # Model file has to be in /model folder
parser.add_argument('--batch', '-b', type=int, required=True,
                    help='Specify which batch size to use for sampling synthetic data')

args = parser.parse_args()

# --- 1. Data preparation
# Load existing model
model_folder = 'model'
model_name = args.model  # !!! CHANGE
model_name += '.pkl'
model_path = os.path.join(model_folder, model_name)
synthesizer = CTGANSynthesizer.load(filepath=model_path)

# Include metadata to run evaluations
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

# --- 3. Sampling
# Create output folder if it does not exist
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
output_name = f"synthetic_data_n{args.generate}_b{args.batch}_{model_name.split('.')[0]}_{current_timestamp}.csv"
output_path = os.path.join(output_folder, output_name)

# Create log folder if it does not exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
training_log_file_name = 'information.txt'
training_log_file_name = os.path.join(log_folder, training_log_file_name)
with open(training_log_file_name, 'a') as log:
    log.write(f"--- Synthetic data generation started at: {current_timestamp}\n")
    log.write(f"\tModel name: {model_name}\n")
    log.write(f"\tModel path: {model_path}\n")
    log.write(f"\tGenerating {args.generate} rows\n")
    log.write(f"\tBatch size: {args.batch}\n")
    log.write(f"\tOutput file path: {output_path}\n")

# Write synthetic data to output file
synthetic_data = synthesizer.sample(
    num_rows=args.generate if args.generate else 5000,  # Command-line argument, generated n=32 by default
    batch_size=args.batch if args.batch else 32,  # Command-line argument, batch size 32 by default
    output_file_path=output_path
)
