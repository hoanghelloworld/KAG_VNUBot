# data_processor.py
import json
import os
import re
from bs4 import BeautifulSoup # Có thể cần lại nếu dữ liệu crawl vẫn còn HTML
import config

# TODO: (Người 1) Hoàn thiện các hàm làm sạch và chuẩn hóa dữ liệu.
#       - Loại bỏ HTML tags còn sót lại (nếu crawler chưa làm sạch hoàn toàn).
#       - Xử lý các ký tự đặc biệt, encoding.
#       - Loại bỏ các đoạn văn bản nhiễu (ví dụ: menu, footer, quảng cáo) dựa trên cấu trúc hoặc từ khóa.
#       - Có thể thực hiện thêm các bước như tách câu, lemmatization (nếu cần cho giai đoạn sau).

def clean_html_content(html_text):
    """Loại bỏ HTML tags và trả về văn bản thuần."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    # TODO: (Người 1) Tùy chỉnh logic loại bỏ thẻ không mong muốn
    # Ví dụ: loại bỏ script, style, navigation, footer
    for element_type in ["script", "style", "nav", "footer", "header", "aside"]:
        for element in soup.find_all(element_type):
            element.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    return text

def normalize_text(text):
    """Chuẩn hóa văn bản (ví dụ: loại bỏ khoảng trắng thừa)."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip() # Thay thế nhiều khoảng trắng bằng một
    # TODO: (Người 1) Thêm các bước chuẩn hóa khác nếu cần (ví dụ: lowercase, xử lý unicode)
    return text

def process_crawled_file(filepath):
    """Xử lý một file JSON đã crawl."""
    try:
        with open(filepath, 'r', encoding='utf-utf-8') as f:
            data = json.load(f)
        
        # Giả sử file JSON có dạng {"source_url": ..., "title": ..., "content": ...}
        # và content có thể vẫn còn HTML hoặc cần làm sạch.
        raw_content = data.get("content", "")
        
        # Nếu content là HTML, làm sạch nó
        # Nếu crawler đã trả về text thuần, bước này có thể không cần hoặc chỉ cần normalize
        cleaned_content_from_html = clean_html_content(raw_content) # Nếu content là html
        # Hoặc nếu content đã là text:
        # cleaned_content_from_html = raw_content 

        processed_text = normalize_text(cleaned_content_from_html)
        
        # Tạo source_id từ URL hoặc title (cần duy nhất)
        source_id = data.get("title", os.path.basename(filepath).replace(".json","")).replace(" ", "_").lower()
        # Hoặc source_id = hashlib.md5(data.get("source_url").encode()).hexdigest()

        if not processed_text: # Bỏ qua nếu không có nội dung
            return None
            
        return {"source_id": source_id, "text_content": processed_text, "original_url": data.get("source_url"), "title": data.get("title")}

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def process_all_crawled_data():
    """Xử lý tất cả các file trong thư mục crawled_data."""
    processed_data_list = []
    if not os.path.exists(config.CRAWLED_DATA_DIR):
        print(f"Crawled data directory not found: {config.CRAWLED_DATA_DIR}")
        return processed_data_list

    for filename in os.listdir(config.CRAWLED_DATA_DIR):
        if filename.endswith(".json"): # Giả sử crawler lưu file JSON
            filepath = os.path.join(config.CRAWLED_DATA_DIR, filename)
            print(f"Processing crawled file: {filepath}")
            processed_item = process_crawled_file(filepath)
            if processed_item:
                processed_data_list.append(processed_item)
    
    # Lưu trữ dữ liệu đã xử lý
    output_filepath = os.path.join(config.PROCESSED_DATA_DIR, "all_processed_texts.json")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_data_list, f, ensure_ascii=False, indent=4)
    print(f"All processed data saved to {output_filepath}")
    return processed_data_list

if __name__ == "__main__":
    # TODO: (Người 1) Đảm bảo rằng đã có dữ liệu trong config.CRAWLED_DATA_DIR từ bước crawler.
    #       Chạy hàm này sau khi crawler đã hoàn thành.
    
    # Tạo file dummy để test processor nếu chưa có crawler output
    dummy_crawled_file = os.path.join(config.CRAWLED_DATA_DIR, "dummy_page_1.json")
    if not os.path.exists(dummy_crawled_file) and not os.listdir(config.CRAWLED_DATA_DIR):
         with open(dummy_crawled_file, 'w', encoding='utf-8') as f:
            json.dump({
                "source_url": "http://example.com/dummy_course", 
                "title": "Dummy AI Course", 
                "content": "<h1>Welcome</h1><p>This is a <b>dummy</b> course about   AI.   Learn exciting things!</p><footer>Contact us</footer>"
            }, f, indent=4)
         print(f"Created dummy crawled file: {dummy_crawled_file} for testing processor.")

    processed_texts = process_all_crawled_data()
    if processed_texts:
        print(f"\nSuccessfully processed {len(processed_texts)} documents.")
        # print("First processed document (sample):")
        # print(json.dumps(processed_texts[0], indent=2, ensure_ascii=False))
    else:
        print("No data was processed. Check crawled_data directory or processor logic.")
