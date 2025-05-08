# main_app.py
import streamlit.web.cli as stcli
import os
import sys

# TODO: File này dùng để chạy ứng dụng Streamlit.
#       Có thể không cần nếu logic chạy đã nằm trong app_ui.py và bạn chạy bằng `streamlit run app_ui.py`.
#       Nếu có các bước khởi tạo phức tạp hơn trước khi chạy UI, có thể đặt ở đây.

def run_streamlit_app():
    """Chạy ứng dụng Streamlit."""
    # Đảm bảo file app_ui.py nằm trong cùng thư mục hoặc đúng đường dẫn
    # Lấy đường dẫn tuyệt đối đến app_ui.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(current_dir, "app_ui.py")

    if not os.path.exists(app_file):
        print(f"Error: app_ui.py not found at {app_file}")
        return

    # Chuẩn bị các tham số cho streamlit run
    # Ví dụ: sys.argv = ["streamlit", "run", app_file, "--server.port", "8501"]
    sys.argv = ["streamlit", "run", app_file] 
    
    print(f"Attempting to run Streamlit app: {' '.join(sys.argv)}")
    stcli.main()


if __name__ == "__main__":
    print("Starting KAG Reasoning Chatbot Application...")
    # TODO: Thêm các bước kiểm tra hoặc khởi tạo cần thiết trước khi chạy UI.
    # Ví dụ: kiểm tra sự tồn tại của các model, artifacts (mặc dù app_ui cũng đã làm)
    
    # Kiểm tra cấu hình
    try:
        import config
        print(f"Config loaded. Using device: {config.DEVICE}")
        print(f"LLM Model: {config.LLM_MODEL_NAME}")
        print(f"Artifacts directory: {config.ARTIFACTS_DIR}")
    except ImportError:
        print("Error: config.py not found. Please ensure it exists in the project root.")
        exit()
    except Exception as e:
        print(f"Error loading config: {e}")
        exit()

    run_streamlit_app()
