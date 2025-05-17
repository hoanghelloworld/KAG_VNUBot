import os
import json

def create_json_from_text_files():
    processed_data_dir = "data_processed"
    output_file_path = os.path.join(processed_data_dir, "all_processed_texts.json")
    all_texts = []

    try:
        for filename in os.listdir(processed_data_dir):
            if filename.endswith(".txt"):
                source_id = os.path.splitext(filename)[0] 
                file_path = os.path.join(processed_data_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    all_texts.append({
                        "source_id": source_id,
                        "text_content": text_content
                    })
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        with open(output_file_path, 'w', encoding='utf-8') as json_f:
            json.dump(all_texts, json_f, ensure_ascii=False, indent=4)
            
        print(f"Successfully created {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_json_from_text_files()
