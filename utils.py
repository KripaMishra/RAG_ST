from pathlib import Path

def load_data(file_path):
    file_path=Path(file_path)
    with open(file_path) as f:
        text_lines=f.read()
    return text_lines

