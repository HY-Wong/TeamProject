import mne
import os
import glob

def extract_channels(input_file_path, output_file_path):
    """
    Extract 'Snore', 'Tracheal', and 'Mic' from an EDF file and save it to a new FIF file.
    
    Parameters:
    - input_file_path: str, path to the input EDF file
    - output_file_path: str, path to the output FIF file
    """
    # Just load the EDF file with preload=False to minimize memory usage
    raw = mne.io.read_raw_edf(input_file_path, preload=False)
    
    # Select 'Mic' channel only
    raw = raw.pick_channels(['Snore', 'Tracheal', 'Mic'])
    
    # Save to a new FIF file
    raw.save(output_file_path, overwrite=True)
    
    print(f"Selected channels saved to {output_file_path}")  

    
def process_patient_data(base_directory):
    """
    Process all EDF files in the base directory, extracting channel 'Snore', 'Tracheal', and 'Mic'.
    
    Parameters:
    - base_directory: str, path to the base directory containing the patient folders
    """
    # Get all EDF files in the base directory (and subdirectories)
    edf_files = glob.glob(os.path.join(base_directory, '**', '*.edf'), recursive=True)
    
    for edf_file in edf_files:
        # Extract patient ID from the filename, the ID is in positions 6-10
        # For example, file name '00001016-100507[001].edf'
        file_name = os.path.basename(edf_file)
        patient_id = file_name[4:8]

         # Output file name: replace edf with fif
        output_file_path = os.path.join(base_directory, patient_id, file_name.replace('.edf', '.fif'))
        
        # Check if the output directory exists, create if not
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Extract and save the 'Mic' channel
        extract_channels(edf_file, output_file_path)

# Base directory where all the patient data is stored
base_directory = '/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF'

# Process data for all patients
process_patient_data(base_directory)
