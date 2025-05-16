#!/usr/bin/env python3
# build_kag_vnu.py
import os
import json
from pathlib import Path
from advanced_kag_builder import AdvancedKAGBuilder
import config

def read_text_file(file_path, encoding='utf-8'):
    """Read text file and return content."""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def process_data_dir(data_dir):
    """
    Process all text files in the data directory.
    Returns a list of processed data items ready for KAG builder.
    """
    processed_data_list = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(files)} text files to process...")
    
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        try:
            text_content = read_text_file(file_path)
            # Create a source_id from the filename without extension
            source_id = os.path.splitext(file_name)[0]
            
            # Add to processed data list
            processed_data_list.append({
                "source_id": source_id,
                "text_content": text_content,
                "original_url": f"data_processed/{file_name}"
            })
            print(f"Processed file: {file_name}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return processed_data_list

def main():
    """Main function to build KAG artifacts from processed data."""
    print("Starting KAG Builder process...")
    
    # Ensure artifacts directory exists
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # Process data files
    processed_data_list = process_data_dir(config.PROCESSED_DATA_DIR)
    print(f"Successfully processed {len(processed_data_list)} files.")
    
    # Initialize KAG Builder
    builder = AdvancedKAGBuilder(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    
    # Build KAG artifacts
    print("Building KAG artifacts (knowledge graph, vector index)...")
    builder.build_from_processed_data(processed_data_list)
    
    print("KAG Builder process completed successfully.")

if __name__ == "__main__":
    main() 