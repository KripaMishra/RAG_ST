import os
import json
import re
from nltk.tokenize import sent_tokenize

class DataProcessor:
    def __init__(self, file_path, temp_path):
        self.file_path = file_path
        self.output_path = output_path
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_path = self.temp_file.name

    def __del__(self):
        try:
            os.remove(self.temp_path)
            print(f"Deleted temporary file {self.temp_path}")
        except OSError as e:
            print(f"Error deleting temporary file {self.temp_path}: {e}")


    def get_data(self):
        print(f"Attempting to read file: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)

        print(f"File size: {len(json.dumps(data))} characters")

        if not isinstance(data, list):
            print("Data is not a list. Converting to list.")
            data = [data]

        print(f"Extracted {len(data)} items")

        with open(self.temp_path, 'w') as content_file:
            items_written = 0
            for item in data:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    # Clean the content
                    content = self.clean_text(content)
                    content_file.write(content + "\n")
                    items_written += 1
                else:
                    print(f"Skipping invalid item: {item}")

        print(f"Wrote {items_written} items to temporary file")

        if items_written == 0:
            raise ValueError("No valid items were found in the data.")

        return data

    def clean_text(self, text):
        # Remove new line characters and special characters repeating more than 3 times
        text = re.sub(r'\\n|\n|\\r|\r|//', ' ', text)
        text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)
        return text.strip()

    def split_sentences(self):
        with open(self.temp_path, 'r') as f:
            content = f.read()
        
        # Split content on period followed by a space
        sentences = re.split(r'(?<=\.)\s+', content)
        
        formatted_sentences = []
        current_sentence = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            # Check if the sentence is only numerical or special characters
            if re.match(r'^[\d\W]+$', sentence):
                current_sentence += " " + sentence
            else:
                if current_sentence:
                    current_sentence += " " + sentence
                    # Check if the length of the sentence is between 90 and 850
                    if 90 <= len(current_sentence.strip()) <= 850:
                        formatted_sentences.append(current_sentence.strip())
                    current_sentence = ""
                else:
                    # Check if the next sentence is numerical, if so, merge it
                    if i + 1 < len(sentences) and re.match(r'^[\d\W]+$', sentences[i + 1].strip()):
                        current_sentence = sentence
                    else:
                        # Check if the length of the sentence is between 90 and 850
                        if 90 <= len(sentence.strip()) <= 850:
                            formatted_sentences.append(sentence.strip())
        
        if current_sentence and 90 <= len(current_sentence.strip()) <= 850:
            formatted_sentences.append(current_sentence.strip())
        with open(self.output_path, 'w') as f:
            for sentence in formatted_sentences:
                f.write(sentence + '\n')
            print(f'The content file saved at {self.output_path}')

# Example usage
file_path = '/home/ubuntu/project/Steps/nvidia_docs/nvidia_docs/spiders/output.json'

output_path = '/home/ubuntu/project/Steps/retrieved_result/cleaned_data/cleaned_data_170724.json'

processor = DataProcessor(file_path, output_path)
processor.get_data()
processor.split_sentences()