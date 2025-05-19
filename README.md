
<!-- filepath: /workspaces/KAG_VNUBot/README.md -->
# KAG_VNUBot: Chatbot Tra Cứu Thông Tin Quy Chế Đại Học Quốc Gia Hà Nội

<p align="center">
  <img src="https://www.vnu.edu.vn/upload/2015/01/17449/image/Logo-VNU-1995.png" alt="VNU Logo" width="150"/>
</p>

<p align="center">
  <strong>Một chatbot thông minh sử dụng Kiến thức Đồ thị (Knowledge Graph) và các mô hình ngôn ngữ lớn (LLMs) để trả lời các câu hỏi liên quan đến quy chế, chính sách và thủ tục học vụ tại Đại học Quốc gia Hà Nội (VNU).</strong>
</p>

<p align="center">
  <a href="#tính-năng">Tính năng</a> •
  <a href="#công-nghệ-sử-dụng">Công nghệ</a> •
  <a href="#cài-đặt">Cài đặt</a> •
  <a href="#sử-dụng">Sử dụng</a> •
  <a href="#kiến-trúc-hệ-thống">Kiến trúc</a> •
  <a href="#đóng-góp">Đóng góp</a>
</p>

## Giới thiệu

KAG_VNUBot là một dự án nhằm xây dựng một hệ thống chatbot tiên tiến, có khả năng hiểu và trả lời các câu hỏi phức tạp về các quy định, quy chế đào tạo, chính sách sinh viên, và các thủ tục hành chính khác tại Đại học Quốc gia Hà Nội. Dự án tận dụng sức mạnh của Knowledge Graph (KG) để biểu diễn tri thức một cách có cấu trúc và các Large Language Models (LLMs) để xử lý ngôn ngữ tự nhiên và tạo ra câu trả lời.

Mục tiêu chính là cung cấp một công cụ tra cứu thông tin nhanh chóng, chính xác và thân thiện cho sinh viên, giảng viên và cán bộ của VNU, giúp giảm thiểu thời gian tìm kiếm và nâng cao trải nghiệm người dùng.

## Tính năng

*   **Tra cứu thông tin thông minh**: Hiểu và trả lời các câu hỏi bằng ngôn ngữ tự nhiên về nhiều chủ đề liên quan đến VNU.
*   **Sử dụng Knowledge Graph**: Biểu diễn tri thức về các quy chế, văn bản, đơn vị, chương trình đào tạo dưới dạng đồ thị, cho phép suy luận và truy vấn hiệu quả.
*   **Tích hợp LLM**: Sử dụng các mô hình ngôn ngữ lớn để phân tích câu hỏi, trích xuất thực thể, quan hệ và sinh câu trả lời tự nhiên.
*   **Giao diện người dùng thân thiện**: Giao diện web trực quan (sử dụng Streamlit) để người dùng dễ dàng tương tác.
*   **Quá trình suy luận minh bạch**: Hiển thị các bước suy luận (scratchpad) của chatbot để người dùng hiểu cách câu trả lời được hình thành.
*   **Khả năng mở rộng**: Kiến trúc module cho phép dễ dàng cập nhật dữ liệu và cải tiến các thành phần.

## Công nghệ sử dụng

*   **Ngôn ngữ lập trình**: Python
*   **Xử lý ngôn ngữ tự nhiên (NLP) & LLMs**:
    *   Hugging Face Transformers (cho các mô hình LLM local như Qwen)
    *   Sentence Transformers (cho embedding)
    *   Together AI API (để sử dụng các LLM mạnh mẽ hơn như Llama)
*   **Knowledge Graph**:
    *   NetworkX (để xây dựng và quản lý đồ thị)
*   **Vector Database**:
    *   FAISS (để lưu trữ và tìm kiếm vector embedding hiệu quả)
*   **Giao diện người dùng**:
    *   Streamlit
*   **Thư viện khác**:
    *   Langchain (khung sườn xây dựng ứng dụng LLM)
    *   Pandas (xử lý dữ liệu)
    *   Playwright (thu thập dữ liệu web)
    *   Pydantic (quản lý cấu hình)

## Cài đặt

### Yêu cầu tiên quyết

*   Python 3.8+
*   Git
*   (Tùy chọn) CUDA Toolkit nếu bạn muốn sử dụng GPU cho các mô hình local.

### Các bước cài đặt

1.  **Clone repository:**
    ```bash
    git clone https://your-repository-url/KAG_VNUBot.git
    cd KAG_VNUBot
    ```

2.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\\Scripts\\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: File `requirements.txt` cần được tạo dựa trên các import trong dự án. Bạn có thể dùng `pip freeze > requirements.txt` sau khi cài đặt thủ công các thư viện chính).*

4.  **Cấu hình API Keys (nếu cần):**
    *   Tạo file `.env` trong thư mục gốc của dự án.
    *   Thêm các API key cần thiết, ví dụ:
        ```env
        TOGETHER_API_KEY="your_together_api_key_here"
        GEMINI_API_KEY="your_gemini_api_key_here" # Nếu sử dụng Gemini cho tiền xử lý dữ liệu
        ```
    *   Tham khảo file `config.py` để biết các biến môi trường có thể cấu hình.

5.  **Tải các mô hình LLM và Embedding (nếu chạy local):**
    Các mô hình sẽ tự động được tải về khi chạy lần đầu, nhưng quá trình này có thể mất thời gian. Đảm bảo bạn có kết nối internet ổn định.

6.  **Chuẩn bị dữ liệu và xây dựng Knowledge Graph:**
    *   **Thu thập dữ liệu**: Chạy notebook `data_crawler_and_preprocessed.ipynb` để thu thập dữ liệu từ các nguồn (ví dụ: website VNU, văn bản PDF) và tiền xử lý chúng thành định dạng text.
        *   Cell đầu tiên của notebook hướng dẫn cài đặt Playwright.
        *   Các cell tiếp theo thực hiện việc crawl dữ liệu web và chuyển đổi PDF sang text (sử dụng Gemini API nếu có).
    *   **Xử lý dữ liệu thô**: Chạy script `data_processed.py` để chuẩn hóa và cấu trúc lại dữ liệu text đã thu thập, lưu dưới dạng file JSON.
        ```bash
        python data_processed.py
        ```
    *   **Xây dựng Knowledge Graph và Vector Index**: Chạy script `advanced_kag_builder.py` để:
        *   Phân đoạn (chunking) văn bản.
        *   Trích xuất thực thể và quan hệ bằng LLM.
        *   Xây dựng đồ thị tri thức (Knowledge Graph) và lưu dưới dạng file GML.
        *   Tạo vector embeddings cho các chunk văn bản và xây dựng FAISS index.
        *   Lưu trữ docstore (mapping chunk_id với nội dung chunk).
        ```bash
        python advanced_kag_builder.py
        ```
        Quá trình này sẽ tạo ra các artifacts trong thư mục `artifacts/` (ví dụ: `my_faiss_index.index`, `my_knowledge_graph.gml`, `doc_store.json`).

## Sử dụng

Sau khi hoàn tất các bước cài đặt và chuẩn bị dữ liệu:

1.  **Chạy ứng dụng Streamlit:**
    ```bash
    streamlit run app_ui.py
    ```
    Hoặc nếu bạn muốn chạy qua `main_app.py`:
    ```bash
    python main_app.py
    ```

2.  Mở trình duyệt và truy cập vào địa chỉ được cung cấp (thường là `http://localhost:8501`).

3.  Nhập câu hỏi của bạn vào ô chat và nhấn Enter. Chatbot sẽ xử lý và đưa ra câu trả lời.

4.  Bạn có thể xem quá trình suy luận của chatbot bằng cách mở rộng mục "Show Reasoning Process".

5.  Trong sidebar, bạn có thể:
    *   Xóa lịch sử hội thoại.
    *   Hiển thị/ẩn các cài đặt nâng cao (top_k, max_steps).
    *   Xem thông tin cơ bản về Knowledge Graph (số lượng node, cạnh).

## Kiến trúc hệ thống

Hệ thống KAG_VNUBot bao gồm các thành phần chính sau:

1.  **Data Acquisition & Preprocessing (`data_crawler_and_preprocessed.ipynb`, `data_processed.py`):**
    *   Thu thập dữ liệu từ nhiều nguồn (web, PDF).
    *   Làm sạch, chuẩn hóa và chuyển đổi dữ liệu sang định dạng phù hợp.

2.  **Knowledge Graph Builder (`advanced_kag_builder.py`):**
    *   **Text Splitting**: Chia văn bản thành các đoạn nhỏ (chunks).
    *   **Entity Extraction**: Sử dụng LLM để xác định các thực thể (ví dụ: tên quy định, tên môn học, đơn vị) trong các chunks.
    *   **Relation Extraction**: Sử dụng LLM để xác định mối quan hệ giữa các thực thể.
    *   **Graph Construction**: Xây dựng đồ thị tri thức (NetworkX) từ các thực thể và quan hệ đã trích xuất.
    *   **Vector Embedding & Indexing**: Tạo vector embeddings cho các chunks văn bản (sử dụng Sentence Transformers) và xây dựng FAISS index để tìm kiếm nhanh.
    *   **Artifact Storage**: Lưu trữ KG (GML), FAISS index, và DocStore (JSON).

3.  **Reasoning Solver (`dynamic_reasoning_solver.py`):**
    *   **Query Understanding**: Phân tích câu hỏi của người dùng.
    *   **Tool Selection & Execution**:
        *   `search_vector_db`: Tìm kiếm các chunks văn bản liên quan trong FAISS index.
        *   `query_kg`: Truy vấn Knowledge Graph để tìm thực thể, quan hệ, hoặc thông tin chi tiết.
    *   **Iterative Reasoning (ReAct Pattern)**: Lặp đi lặp lại các bước suy nghĩ (Thought) và hành động (Action) để thu thập thông tin và xây dựng câu trả lời.
    *   **Response Generation**: Tổng hợp thông tin thu thập được và sinh câu trả lời cuối cùng bằng LLM.

4.  **User Interface (`app_ui.py`):**
    *   Giao diện chat dựa trên Streamlit.
    *   Hiển thị câu hỏi, câu trả lời, và quá trình suy luận.
    *   Cho phép người dùng cấu hình một số tham số.

5.  **LLM Utilities (`llm_utils.py`):**
    *   Cung cấp các hàm tiện ích để tương tác với các mô hình LLM (local và API), tạo embeddings.

6.  **Configuration (`config.py`, `prompt_manager`):**
    *   Quản lý các cài đặt của ứng dụng (đường dẫn, tên model, API keys).
    *   Quản lý các system prompts cho LLM.

<p align="center">
  <img src="link_den_so_do_kien_truc.png" alt="Sơ đồ kiến trúc KAG_VNUBot"/> 
  <!-- TODO: Thêm sơ đồ kiến trúc nếu có -->
</p>

## Luồng hoạt động (Workflow)

1.  Người dùng nhập câu hỏi vào giao diện chat.
2.  `app_ui.py` nhận câu hỏi và chuyển đến `DynamicKAGSolver`.
3.  `DynamicKAGSolver` bắt đầu quá trình suy luận:
    a.  LLM (với ReAct prompt) phân tích câu hỏi và quyết định hành động tiếp theo (ví dụ: `search_vector_db` hoặc `query_kg`).
    b.  Công cụ tương ứng được thực thi (tìm kiếm vector hoặc truy vấn KG).
    c.  Kết quả (Observation) từ công cụ được thêm vào "scratchpad".
    d.  LLM tiếp tục suy luận dựa trên scratchpad cập nhật, lặp lại cho đến khi có đủ thông tin hoặc đạt số bước tối đa.
4.  Khi LLM quyết định `finish(answer)`, câu trả lời cuối cùng được hình thành.
5.  Câu trả lời và scratchpad được hiển thị trên giao diện người dùng.

## Đóng góp

Chúng tôi hoan nghênh các đóng góp để cải thiện KAG_VNUBot! Vui lòng tham khảo hướng dẫn đóng góp (CONTRIBUTING.md - nếu có) hoặc tạo Pull Request/Issue.

Các lĩnh vực có thể đóng góp:

*   Cải thiện chất lượng và phạm vi của Knowledge Graph.
*   Tinh chỉnh các prompts cho LLM để tăng độ chính xác.
*   Thêm các nguồn dữ liệu mới.
*   Tối ưu hóa hiệu suất.
*   Cải thiện giao diện người dùng.

## Những việc cần làm (TODO)

*   [ ] Tạo file `requirements.txt` đầy đủ.
*   [ ] Thêm sơ đồ kiến trúc chi tiết vào README.
*   [ ] Viết thêm unit tests và integration tests.
*   [ ] Cải thiện cơ chế xử lý lỗi và logging.
*   [ ] Nghiên cứu các kỹ thuật tiên tiến hơn cho việc xây dựng và truy vấn KG.
*   [ ] Hỗ trợ đa ngôn ngữ (nếu cần).
*   [ ] Triển khai hệ thống lên một nền tảng cloud.

## Giấy phép

Dự án này được cấp phép theo [MIT License](LICENSE.md) (nếu có).

---

Hy vọng KAG_VNUBot sẽ là một công cụ hữu ích cho cộng đồng Đại học Quốc gia Hà Nội!
