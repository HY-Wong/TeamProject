


import os
import xml.etree.ElementTree as ET
import glob

# Define the directory containing the RML files
rml_directory = '/Users/ybys/Desktop/TP/PSG_Audio/APNEA_RMC'


# Note: I completely missed load more option before,
#       here the new code is slightly modified

'''
# List of RML files to process
rml_files = [
    '00000995-100507.rml', '00000999-100507.rml', '00001000-100507.rml',
    '00001006-100507.rml', '00001008-100507.rml', '00001010-100507.rml',
    '00001014-100507.rml', '00001016-100507.rml'
]
'''
# Use glob to find all RML files in the directory
rml_files = glob.glob(os.path.join(rml_directory, '*.rml'))

# Define the namespace (if applicable)
ns = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}

# Process each RML file
for rml_file in rml_files:
    rml_file_path = os.path.join(rml_directory, rml_file)
    
    # Load and parse the RML file
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    
    # Create a new XML structure to hold the extracted elements
    new_root = ET.Element('PatientStudy')

    # Extract the Acquisition element
    acquisition = root.find('ns:Acquisition', ns)
    if acquisition is not None:
        new_root.append(acquisition)

    # Extract the ScoringData element
    scoring_data = root.find('ns:ScoringData', ns)
    if scoring_data is not None:
        new_root.append(scoring_data)

    # Convert the new XML structure to a string or save it directly
    new_tree = ET.ElementTree(new_root)
    
    # Define the output path for the new XML
    new_rml_file_path = os.path.join(rml_directory, f'filtered_{rml_file}')
    
    # Save the new XML file
    new_tree.write(new_rml_file_path, encoding='utf-8', xml_declaration=True)

    print(f"Extracted elements from {rml_file} saved to: {new_rml_file_path}")
