import os
import json
import xml.etree.ElementTree as ET

data_dir = '../data'

for file_name in os.listdir(data_dir):
    if file_name.endswith('.rml'):
        patient_id = file_name.split('.')[0]
        # print(patient_id)
        
        # Path to your .xml (RML) file
        rml_path = os.path.join(data_dir, file_name)

        # Parse the XML file
        tree = ET.parse(rml_path)
        root = tree.getroot()
        # ET.dump(root)

        namespace = {'ns0': 'http://www.respironics.com/PatientStudy.xsd'}

        # Extract Nasal events
        nasal_events = root.findall(".//ns0:Event[@Family='Nasal']", namespace)
        nasal_data = []

        for event in nasal_events:
            event_info = {
                "type": event.get("Type"),
                "start": float(event.get("Start")),
                "duration": float(event.get("Duration"))
            }
            nasal_data.append(event_info)
        
        # Save the event data as a JSON file
        with open(os.path.join(data_dir, f'{patient_id}_nasal.json'), 'w') as json_file:
            json.dump(nasal_data, json_file, indent=4)

        # Extract Respiratory events
        respiratory_events = root.findall(".//ns0:Event[@Family='Respiratory']", namespace)
        respiratory_data = []

        for event in respiratory_events:
            event_info = {
                "type": event.get("Type"),
                "start": float(event.get("Start")),
                "duration": float(event.get("Duration"))
            }
            respiratory_data.append(event_info)

        # Save the event data as a JSON file
        with open(os.path.join(data_dir, f'{patient_id}_respiratory.json'), 'w') as json_file:
            json.dump(respiratory_data, json_file, indent=4)