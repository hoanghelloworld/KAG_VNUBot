# data_crawler.py
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import config # Import cấu hình chung

# TODO: Hoàn thiện chức năng crawl dữ liệu từ một trang web cụ thể.
#       - Xác định cấu trúc trang web mục tiêu.
#       - Implement logic để trích xuất nội dung chính và metadata.
#       - Xử lý các trường hợp lỗi (ví dụ: trang không tồn tại, timeout).
#       - Lưu trữ dữ liệu đã crawl vào thư mục CRAWLED_DATA_DIR.

def fetch_page_content(url):
    """Lấy nội dung HTML của một trang."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Ném lỗi nếu status code là 4xx/5xx
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_course_page(html_content, url):
    """
    (Ví dụ đơn giản) Phân tích trang khóa học để lấy tiêu đề và nội dung.
    Cần tùy chỉnh rất nhiều cho trang web cụ thể.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # TODO: Điều chỉnh các selector này cho phù hợp với trang web mục tiêu
    title_tag = soup.find('h1') # Giả sử tiêu đề nằm trong thẻ <h1>
    title = title_tag.text.strip() if title_tag else "N/A"
    
    # Giả sử nội dung chính nằm trong một div có class 'course-content'
    content_div = soup.find('div', class_='course-content') 
    content = content_div.get_text(separator='\n', strip=True) if content_div else "N/A"
    
    if title == "N/A" and content == "N/A":
        return None # Không lấy được thông tin hữu ích
        
    return {"source_url": url, "title": title, "content": content}

def crawl_website(start_url, max_pages=5):
    """
    Thực hiện crawl website bắt đầu từ start_url.
    (Ví dụ rất đơn giản, không theo link sâu, chỉ crawl các link tìm thấy trên trang đầu)
    """
    print(f"Starting crawl from: {start_url}")
    crawled_data = []
    visited_urls = set()
    urls_to_visit = [start_url]
    
    # TODO: (Người 1) Implement logic duyệt link phức tạp hơn (ví dụ: BFS, DFS, giới hạn độ sâu)
    #       Xử lý robots.txt.
    #       Thêm user-agent.

    pages_crawled_count = 0
    while urls_to_visit and pages_crawled_count < max_pages:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue
        
        print(f"Crawling: {current_url}")
        html = fetch_page_content(current_url)
        visited_urls.add(current_url)
        
        if html:
            # Giả sử chúng ta muốn parse trang này như một trang khóa học
            page_data = parse_course_page(html, current_url)
            if page_data:
                crawled_data.append(page_data)
                pages_crawled_count += 1
                
                # Lưu trữ ngay khi crawl được (hoặc lưu định kỳ)
                filename = os.path.join(config.CRAWLED_DATA_DIR, f"crawled_page_{pages_crawled_count}.json")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=4)
                print(f"Saved data from {current_url} to {filename}")

            # TODO: Implement logic tìm link mới trên trang hiện tại và thêm vào urls_to_visit
            # soup = BeautifulSoup(html, 'html.parser')
            # for link in soup.find_all('a', href=True):
            #     abs_url = requests.compat.urljoin(current_url, link['href'])
            #     if abs_url.startswith(config.TARGET_WEBSITE_URL) and abs_url not in visited_urls: # Chỉ crawl trong domain
            #          urls_to_visit.append(abs_url)

        time.sleep(1) # Lịch sự, tránh request quá nhanh

    print(f"Crawling finished. Crawled {len(crawled_data)} pages.")
    return crawled_data

if __name__ == "__main__":
    # TODO: Cấu hình URL bắt đầu và các tham số khác từ config.py
    # dummy_start_url = "https://www.example.com/courses" # Thay bằng URL thực tế
    # print(f"Make sure to set TARGET_WEBSITE_URL in config.py")
    # if hasattr(config, 'TARGET_WEBSITE_URL'):
    #     crawled_content = crawl_website(config.TARGET_WEBSITE_URL, max_pages=config.MAX_PAGES_TO_CRAWL if hasattr(config, 'MAX_PAGES_TO_CRAWL') else 3)
    #     # Dữ liệu đã được lưu vào file trong quá trình crawl
    # else:
    #     print("Please set TARGET_WEBSITE_URL in config.py to run the crawler.")

    # Ví dụ chạy với URL giả định (cần server mock hoặc URL thật)
    print("Running crawler example (dummy, will likely fail without a real target or mock server)...")
    # crawl_website("http://localhost:8000/page1.html", max_pages=1) # Ví dụ nếu bạn có server local
    # Để chạy thực tế, bạn cần thay thế bằng URL của trang web đào tạo.
    # Hiện tại, file này chỉ là khung, cần Người 1 hoàn thiện.
