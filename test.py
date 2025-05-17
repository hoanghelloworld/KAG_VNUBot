import os
import json
import config

def create_all_processed_texts_json():
    """
    Tạo file all_processed_texts.json từ các file .txt trong thư mục data_processed.
    """
    processed_dir = config.PROCESSED_DATA_DIR
    output_file_path = os.path.join(processed_dir, "all_processed_texts.json")
    
    all_texts_data = []

    if not os.path.isdir(processed_dir):
        print(f"Lỗi: Thư mục '{processed_dir}' không tồn tại.")
        return

    print(f"Đang quét các file .txt trong thư mục: {processed_dir}")
    for filename in os.listdir(processed_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(processed_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # original_url có thể được thêm vào nếu bạn có thông tin này
                # Hiện tại, chúng ta sẽ để trống hoặc bỏ qua
                all_texts_data.append({
                    "source_id": filename,
                    "text_content": content,
                    "original_url": "" # Thêm URL gốc nếu có, nếu không để trống
                })
                print(f"Đã xử lý: {filename}")
            except Exception as e:
                print(f"Lỗi khi đọc file {filename}: {e}")

    if not all_texts_data:
        print("Không tìm thấy file .txt nào để xử lý.")
        return

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_texts_data, f, ensure_ascii=False, indent=4)
        print(f"Đã tạo thành công file '{output_file_path}' với {len(all_texts_data)} tài liệu.")
    except Exception as e:
        print(f"Lỗi khi ghi file JSON '{output_file_path}': {e}")

if __name__ == "__main__":
    # Tạo thư mục artifacts và data_processed nếu chưa có (để đảm bảo config hoạt động)
    if not os.path.exists(config.ARTIFACTS_DIR):
        os.makedirs(config.ARTIFACTS_DIR)
    if not os.path.exists(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)
        print(f"Đã tạo thư mục rỗng: {config.PROCESSED_DATA_DIR}. Hãy đặt các file .txt của bạn vào đây.")

    create_all_processed_texts_json()