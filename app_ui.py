# app_ui.py
import streamlit as st
import sys
import os

DEV_MODE = True # Chế độ phát triển, có thể tắt khi deploy

# Thêm project_root vào sys.path để import các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # Giả sử app_ui.py nằm trong project_root
# Nếu app_ui.py nằm trong thư mục con, ví dụ `ui/`, thì project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config # Để kiểm tra sự tồn tại của artifacts
from dynamic_reasoning_solver import DynamicKAGSolver # Import solver
# import llm_utils # Có thể cần để khởi tạo model nếu solver không tự làm

# TODO:  Thiết kế giao diện người dùng thân thiện.
#       - Hiển thị quá trình suy luận (scratchpad) một cách rõ ràng (ví dụ: trong expander).
#       - Xử lý trạng thái loading khi solver đang chạy.
#       - Cho phép người dùng cấu hình một số tham số cơ bản (ví dụ: top_k, max_steps) nếu muốn.

# --- Khởi tạo Solver (chỉ một lần) ---
@st.cache_resource # Quan trọng để không load lại solver mỗi lần interact
def load_solver():
    if DEV_MODE:
        # Trả về một solver giả cho phát triển UI
        from types import SimpleNamespace
        mock_solver = SimpleNamespace()
        mock_solver.solve = lambda query: f"Đây là câu trả lời mẫu cho: '{query}'\n\nCâu trả lời này chỉ dùng để phát triển giao diện."
        return mock_solver

    # Kiểm tra sự tồn tại của artifacts trước khi khởi tạo solver
    required_files = [config.FAISS_INDEX_PATH, config.GRAPH_PATH, config.DOC_STORE_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Application cannot start. Missing critical KAG artifact files: {', '.join(missing_files)}. "
                 "Please ensure the KAG data pipeline (crawler, processor, builder) has been run successfully.")
        return None
    try:
        # llm_utils.get_llm_model() # Pre-load models if not handled by solver's init
        # llm_utils.get_embedding_model()
        solver = DynamicKAGSolver()
        return solver
    except Exception as e:
        st.error(f"Error initializing KAG Solver: {e}")
        st.error("Please check if models are downloaded and artifacts are correctly built.")
        return None

# --- Giao diện Streamlit ---
st.set_page_config(page_title="KAG Reasoning Chatbot", layout="wide")

# Landing Page: Nếu chưa có tin nhắn thì logo+title nằm giữa màn hình
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown(
        """
        <style>
        .centered-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 70vh;
        }
        .chat-title {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>

        <div class="centered-container">
            <div class="chat-title">
                <img src="https://www.vnu.edu.vn/upload/2015/01/17449/image/Logo-VNU-1995.png" width="60" style="margin-right: 15px;">
                <h1 style="margin: 0; font-size: 40px;">VNU Chat</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Khi đã có tin nhắn: logo + VNU Chat hiển thị như header
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <img src="https://www.vnu.edu.vn/upload/2015/01/17449/image/Logo-VNU-1995.png" alt="VNU Logo" width="50" style="margin-right: 10px;">
            <h2 style="margin: 0;">VNU Chat</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# Load solver
solver_instance = load_solver()

# --- Thêm Sidebar cho cấu hình ---
with st.sidebar:
    st.header("Cấu hình")
    
    # Chỉ hiển thị cài đặt nâng cao khi người dùng bật tùy chọn
    show_advanced = st.checkbox("Hiển thị cài đặt nâng cao", False)
    
    # Thêm nút xóa lịch sử chat
    if st.button("Xóa lịch sử hội thoại"):
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.rerun()
    
    # Hiển thị cài đặt nâng cao nếu được chọn
    if show_advanced:
        st.subheader("Cài đặt tìm kiếm")
        
        # Cấu hình số lượng kết quả tìm kiếm (top_k)
        top_k = st.slider(
            "Số lượng kết quả tìm kiếm (top_k)", 
            min_value=1, 
            max_value=10, 
            value=config.TOP_K_RETRIEVAL,
            help="Số lượng đoạn văn bản liên quan nhất được truy xuất từ cơ sở tri thức"
        )
        
        # Cấu hình số bước suy luận tối đa (max_steps)
        max_steps = st.slider(
            "Số bước suy luận tối đa", 
            min_value=1, 
            max_value=5, 
            value=config.MAX_REASONING_STEPS,
            help="Số bước suy luận tối đa khi giải quyết câu hỏi"
        )
        
        # Hiển thị thông tin về Đồ thị tri thức (KG)
        st.subheader("Thông tin về KAG")
        if not DEV_MODE and solver_instance:
            # Hiển thị thông tin về đồ thị nếu có
            try:
                st.text(f"Số lượng node: {len(solver_instance.graph.nodes)}")
                st.text(f"Số lượng cạnh: {len(solver_instance.graph.edges)}")
                st.text(f"Số lượng chunk văn bản: {len(solver_instance.doc_store)}")
            except Exception as e:
                st.warning(f"Không thể truy xuất thông tin KAG {str(e)}")
        else:
            if DEV_MODE:
                st.info("Đang chạy ở chế độ phát triển. Thông tin KAG không khả dụng.")
            else:
                st.warning("Solver không được khởi tạo. Thông tin KAG không khả dụng.")



if solver_instance is None:
    st.warning("Solver could not be loaded. Please check the console for errors and ensure KAG artifacts are present.")
else:
    #giao diện landing
    

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Câu hỏi của bạn là gì?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            
            # TODO:  Hiển thị "Thinking..." hoặc spinner trong khi solver chạy.
            #       Cách tốt hơn là chạy solver trong một thread riêng để UI không bị block.
            #       Streamlit có cách để stream output từ generator.
            
            # Hiện tại, chạy đồng bộ (UI sẽ bị block)
            try:
                if DEV_MODE:
                    # Giả lập quá trình suy luận với độ trễ nhỏ
                    import time
                    with st.spinner("Đang suy luận và tìm kiếm..."):
                        time.sleep(1)  # Giả lập độ trễ 1 giây
                        if show_advanced:
                            st.info(f"Tìm kiếm với top_k={top_k}, max_steps={max_steps}")
                        
                        final_answer = solver_instance.solve(prompt)
                        scratchpad_log = f"""
            --- SOLVER: Reasoning Step 1/3 ---
            Phân tích câu hỏi: "{prompt}"
            Tôi cần tìm kiếm thông tin liên quan đến chủ đề này.

            --- SOLVER: Reasoning Step 2/3 ---
            Đã tìm thấy các thông tin sau:
            - Điểm 1: Đây là thông tin mẫu
            - Điểm 2: Thông tin này chỉ dùng để phát triển UI

            --- SOLVER: Reasoning Step 3/3 ---
            Tổng hợp câu trả lời dựa trên thông tin tìm được.
                        """
                else:
                    # Chạy solver thật
                    with st.spinner("Đang suy luận và tìm kiếm..."):
                        # Truyền tham số cấu hình cho solver nếu cần
                        solver_args = {}
                        if show_advanced:
                            solver_args = {
                                "top_k": top_k,
                                "max_steps": max_steps
                            }
                            st.info(f"Tìm kiếm với top_k={top_k}, max_steps={max_steps}")
                        
                        if solver_args:
                            final_answer = solver_instance.solve(prompt, **solver_args)
                        else:
                            final_answer = solver_instance.solve(prompt)
                        
                        scratchpad_log = "Scratchpad logging not fully implemented in UI for this version. Full log in console."
                
                full_response_text = final_answer
                message_placeholder.markdown(full_response_text)
                with st.expander("Show Reasoning Process"):
                    st.text_area("Scratchpad", value=scratchpad_log, height=300, disabled=True, key="current_scratchpad")
                
                
                # Thêm scratchpad vào message
                assistant_message = {"role": "assistant", "content": full_response_text, "scratchpad": scratchpad_log}

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response_text = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."
                message_placeholder.markdown(full_response_text)
                assistant_message = {"role": "assistant", "content": full_response_text, "scratchpad": f"Error: {e}"}
        
        st.session_state.messages.append(assistant_message)


# TODO:  Thêm các tính năng UI khác:
#       - Nút "Clear Chat".
#       - Hiển thị thông tin về KG (ví dụ: số lượng node, cạnh) - có thể ở sidebar.
#       - Cho phép upload tài liệu mới để build KG (nếu có chức năng đó).
