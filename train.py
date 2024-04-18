import argparse
import datetime
import os

from sdv.single_table import CTGANSynthesizer
from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata

# Get current timestamp for unique identifiers
current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create argument parser
parser = argparse.ArgumentParser(description='Generate synthetic data and save it to a CSV file')

# Add arguments
# Files have to be in /file folder
parser.add_argument('--input', '-i', type=str, required=True,
                    help='Specify name of input file containing real data without the extension')  # Input file has to be in /input folder
parser.add_argument('--load', '-l', metavar="FILE",
                    help='Specify an existing GAN model without the extension')  # Model file has to be in /model folder
parser.add_argument('--epochs', '-ep', type=int,
                    help='Specify the number of epochs for training')
parser.add_argument('--cuda', '-c', action="store_true",
                    help='Enable CUDA computing')
args = parser.parse_args()

# --- 1. Data preparation
# Load real data
datasets = load_csvs(folder_name='./input/')
input_file_name = 'filtered_new_data'  # !!! CHANGE
real_data = datasets[input_file_name]

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
# print(metadata)  # DEBUG, validate metadata

# --- 2. Modeling
if args.load:  # Load existing synthesizer if --load flag is present
    model_folder = 'model'
    model_name = args.load  # Command-line argument
    model_name += '.pkl'
    model_path = os.path.join(model_folder, model_name)
    synthesizer = CTGANSynthesizer.load(
        filepath=model_path
    )

else:  # Train new synthesizer if --load flag is not present
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        enforce_rounding=True,  # Control whether the synthetic data should have the same number of decimal digits as the real data
        enforce_min_max_values=True,  # Control whether the synthetic data should adhere to the same min/max boundaries set by the real data, default value is False
        epochs=args.epochs if args.epochs else 300,  # Command-line argument, 300 by default
        verbose=True,
        cuda=args.cuda  # Command-line argument, False by default
    )

model_folder = 'model'
os.makedirs(model_folder, exist_ok=True)
model_number = 1
model_base_name = f'_ctgan_i_{args.input}_ep{args.epochs}'
model_name = 'mdl' + str(model_number) + model_base_name + '.pkl'
model_path = os.path.join(model_folder, model_name)
print(model_path)

while os.path.exists(model_path):
    print('Model path already exists:', model_path)
    model_number += 1
    model_name = 'mdl' + str(model_number) + model_base_name + '.pkl'
    model_path = os.path.join(model_folder, model_name)

print("Final model path:", model_path)

# Create log folder if it does not exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
training_log_file_name = 'information.txt'
training_log_file_name = os.path.join(log_folder, training_log_file_name)
with open(training_log_file_name, 'a') as log:
    log.write(f"--- CTGAN model training started at: {current_timestamp}\n")
    log.write(f"\tModel name: {model_name}\n")
    log.write(f"\tModel path: {model_path}\n")
    log.write(f"\tModel was {'NOT' if not {args.cuda} else ''} trained using CUDA\n")
    log.write(f"\tTraining on {args.epochs} epochs\n")
    log.write(f"\tInput file path: {os.path.join('input', input_file_name+'.csv')}\n")

# --- 3. Model training
synthesizer.fit(real_data)

# Save trained model
synthesizer.save(filepath=model_path)

# Save model losses in log folder as CSV
losses = synthesizer.get_loss_values()
model_log_file_name = f'losses_{model_name}_{current_timestamp}.csv'
model_log_file_path = os.path.join(log_folder, model_log_file_name)
losses.to_csv(model_log_file_path, index=False)
