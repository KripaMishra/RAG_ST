
import json

def get_data(input_path, output_path):
    # Load the JSON data
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
        
    # Ensure data is a list of dictionaries
    if not isinstance(data, list):
        raise TypeError("The input JSON should be an array of objects.")

    # Open the output file for writing
    with open(output_path, 'w') as content_file:
        # Iterate over each item in the JSON array
        for item in data:
            if not isinstance(item, dict) or 'content' not in item:
                raise ValueError("Each item in the input JSON should be a dictionary with a 'content' key.")
            content = item['content']
            content_file.write(content + "\n")  # Add a newline to separate contents
        print(f"Contents written to {output_path}")

# input_path = '/home/ubuntu/Steps/nvidia_docs/nvidia_docs/spiders/output.json'
# output_path = '/home/ubuntu/Steps/text1.txt'

# get_data(input_path, output_path)