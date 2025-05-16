import os
import json
import re
import config

def process_regulation_file(file_path):
    """
    Process a VNU regulation text file and convert it to JSON format.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        dict: Dictionary with source_id, text_content
    """
    # Extract filename for source_id
    filename = os.path.basename(file_path)
    source_id = os.path.splitext(filename)[0]
    
    # Read the text file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except UnicodeDecodeError:
        # Try with another encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text_content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    # Basic cleaning - remove excessive whitespace and normalize newlines
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    text_content = re.sub(r' {2,}', ' ', text_content)
    
    # Create the processed data dictionary
    processed_data = {
        "source_id": source_id,
        "text_content": text_content,
        "document_type": "regulation",
        "original_url": ""  # Can be populated later if needed
    }
    
    return processed_data

def save_to_json(processed_data, output_path=None):
    """
    Save processed data to JSON file.
    
    Args:
        processed_data (dict): Processed document data
        output_path (str, optional): Path to save the JSON file.
            If None, uses config.PROCESSED_DATA_DIR/all_processed_texts.json
    """
    if not output_path:
        # Make sure the directory exists
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "all_processed_texts.json")
    
    # Check if the output file already exists
    if os.path.exists(output_path):
        # Load existing data
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            # If it's not a list, convert to list
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
                
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Could not read existing JSON file or file is corrupted. Creating new one.")
            existing_data = []
    else:
        existing_data = []
    
    # Check if this document is already in the data
    for i, doc in enumerate(existing_data):
        if doc.get("source_id") == processed_data.get("source_id"):
            # Replace existing entry
            existing_data[i] = processed_data
            print(f"Replacing existing document with source_id: {processed_data['source_id']}")
            break
    else:
        # Add new document if not found
        existing_data.append(processed_data)
        print(f"Added new document with source_id: {processed_data['source_id']}")
    
    # Save the data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved processed data to {output_path}")

if __name__ == "__main__":
    # Fix console encoding for Vietnamese characters
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Define the folder path to process
    folder_path = "D:\\KAG_VNUBot-trinhdz\\data_processed"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        exit(1)
        
    # Get all text files in the folder
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not text_files:
        print(f"No text files found in {folder_path}")
        exit(0)
        
    print(f"Found {len(text_files)} text files to process")
    
    # Process each file
    successful_files = 0
    failed_files = 0
    
    for i, filename in enumerate(text_files):
        file_path = os.path.join(folder_path, filename)
        try:
            # Use safer printing method
            print(f"\nProcessing file {i+1}/{len(text_files)}: {filename.encode('utf-8').decode('utf-8', 'ignore')}")
            
            # Process the regulation file
            processed_data = process_regulation_file(file_path)
            
            if processed_data:
                # Save the processed data
                save_to_json(processed_data)
                print(f"Successfully processed file")
                print(f"Text length: {len(processed_data['text_content'])} characters")
                successful_files += 1
            else:
                print(f"Failed to process file")
                failed_files += 1
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            failed_files += 1
    
    print(f"\nProcessing complete: {successful_files} files processed successfully, {failed_files} files failed")