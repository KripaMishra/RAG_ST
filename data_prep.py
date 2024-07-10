import os
import json
import nltk
import re
import tempfile

nltk.download('punkt')

class DataPrep:
    def __init__(self, input_path, output_path):
        self.file_path = input_path
        self.output_path = output_path
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)  # Create a temporary file
        self.temp_path = self.temp_file.name

    def __del__(self):
        try:
            os.remove(self.temp_path)
            print(f"Deleted temporary file {self.temp_path}")
        except OSError as e:
            print(f"Error deleting temporary file {self.temp_path}: {e}")

    def get_data(self):
        # Load the JSON data
        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)
            
        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            raise TypeError("The input JSON should be an array of objects.")

        # Open the temp file for writing
        with open(self.temp_path, 'w') as content_file:
            # Iterate over each item in the JSON array
            for item in data:
                if not isinstance(item, dict) or 'content' not in item:
                    raise ValueError("Each item in the input JSON should be a dictionary with a 'content' key.")
                content = item['content']
                content_file.write(content + "\n")  # Add a newline to separate contents

    def split_sentences(self):
        # Use nltk's sent_tokenize to split sentences
        with open(self.temp_path, 'r') as f:
            content = f.read()
        sentences = nltk.sent_tokenize(content)
        
        # Join sentences that were split incorrectly
        formatted_sentences = []
        for sentence in sentences:
            if re.match(r'^[0-9]+(\.[0-9]+)+$', sentence):  # Check if the sentence is a number
                if formatted_sentences:
                    formatted_sentences[-1] += ' ' + sentence  # Append to the previous sentence
                else:
                    formatted_sentences.append(sentence)
            else:
                formatted_sentences.append(sentence)
        
        with open(self.output_path, 'w') as f:
            for sentence in formatted_sentences:
                f.write(sentence.strip() + '\n')
            print(f'The content file saved at {self.output_path}')

# # Example usage
# input_path = '/home/ubuntu/Steps/nvidia_docs/nvidia_docs/spiders/output.json'
# output_path = '/home/ubuntu/Steps/text1.txt'

# data_prep = DataPrep(input_path, output_path)
# data_prep.get_data()
# data_prep.split_sentences()
