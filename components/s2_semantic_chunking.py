from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_sentences(text):
    return [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

def split_large_chunk(chunk, max_length=300):
    words = chunk.split()
    sub_chunks = []
    current_sub_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            sub_chunks.append(' '.join(current_sub_chunk))
            current_sub_chunk = [word]
            current_length = len(word)
        else:
            current_sub_chunk.append(word)
            current_length += len(word) + 1

    if current_sub_chunk:
        sub_chunks.append(' '.join(current_sub_chunk))

    return sub_chunks

def semantic_chunking(text, threshold=0.3, max_chunk_length=300):
    sentences = split_into_sentences(text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    chunks = []
    current_chunk = [0]
    
    for i in range(1, len(sentences)):
        if np.max(similarity_matrix[i, current_chunk]) >= threshold:
            current_chunk.append(i)
        else:
            chunks.append(current_chunk)
            current_chunk = [i]
    
    if current_chunk:
        chunks.append(current_chunk)
    
    result = []
    for chunk in chunks:
        chunk_text = ' '.join([sentences[idx] for idx in chunk])
        if len(chunk_text) > max_chunk_length:
            result.extend(split_large_chunk(chunk_text, max_chunk_length))
        else:
            result.append(chunk_text)
    
    return result

def save_chunks_to_json(chunks, output_file, micro_file, micro_threshold=100):
    regular_chunks = []
    micro_chunks = []
    chunk_id = 1
    
    for chunk in chunks:
        chunk_dict = {"id": chunk_id, "content": chunk}
        if len(chunk) < micro_threshold:
            micro_chunks.append(chunk_dict)
        else:
            regular_chunks.append(chunk_dict)
        chunk_id += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(regular_chunks, f, ensure_ascii=False, indent=2)
    
    with open(micro_file, 'w', encoding='utf-8') as f:
        json.dump(micro_chunks, f, ensure_ascii=False, indent=2)
    
    return len(regular_chunks), len(micro_chunks)

# Example usage
file_path = '/home/ubuntu/project/Steps/result/cleaned_data/cleaned_data_170724.txt'
similarity_threshold = 0.15
max_chunk_length = 400
output_json_file = '/home/ubuntu/project/Steps/result/preprocessed_chunks.json'
micro_json_file = '/home/ubuntu/project/Steps/result/micro_chunks.json'
micro_threshold = 100

text = load_text(file_path)
chunks = semantic_chunking(text, similarity_threshold, max_chunk_length)

print(f"Total number of chunks: {len(chunks)}")

# Save chunks to JSON files
regular_count, micro_count = save_chunks_to_json(chunks, output_json_file, micro_json_file, micro_threshold)

print(f"\nRegular chunks saved to {output_json_file}: {regular_count}")
print(f"Micro chunks saved to {micro_json_file}: {micro_count}")

# Print some example chunks
print("\nExample chunks:")
for i, chunk in enumerate(chunks[:5], 1):  # Print first 5 chunks as examples
    print(f"\nChunk {i}:")
    print(f"Content: {chunk}")
    print(f"Length: {len(chunk)}")

# Print example of JSON format
print("\nExample of JSON format:")
example_chunks = [{"id": i, "content": chunk} for i, chunk in enumerate(chunks[:2], 1)]
print(json.dumps(example_chunks, indent=2))