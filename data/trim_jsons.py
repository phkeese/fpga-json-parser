import sys
import json
import re
import os

input_dir = 'raw'
output_dir = 'processed'

def convert_numbers_to_strings(json_object):
    if isinstance(json_object, dict):
        for key, value in json_object.items():
            json_object[key] = convert_numbers_to_strings(value)
    elif isinstance(json_object, list):
        for i in range(len(json_object)):
            json_object[i] = convert_numbers_to_strings(json_object[i])
    elif isinstance(json_object, str):
        # Remove unicode characters
        json_object = re.sub(r'[^\x20-\x7E]+', '', json_object)
    else:
        return str(json_object)
    return json_object

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        json_file_path = os.path.join(input_dir, filename)
        json_file_path_strings = os.path.join(output_dir, filename.replace('.json', '_trimmed.json'))

        try:
            with open(json_file_path) as file:
                data = json.load(file)
                data = convert_numbers_to_strings(data)
                #print(data)
            with open(json_file_path_strings, 'w') as file2:
                json.dump(data, file2, indent=2)
        except FileNotFoundError:
            print(f"File '{json_file_path}' not found.")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")