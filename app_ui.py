# app_ui.py
import streamlit as st
import sys
import os

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
st.title("📚 KAG Powered Reasoning Chatbot")
st.markdown("Hỏi những câu hỏi phức tạp về nội dung đào tạo đã được xử lý!")

# Load solver
solver_instance = load_solver()

if solver_instance is None:
    st.warning("Solver could not be loaded. Please check the console for errors and ensure KAG artifacts are present.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "scratchpad" in message and message["scratchpad"]:
                with st.expander("Show Reasoning Process"):
                    st.text_area("Scratchpad", value=message["scratchpad"], height=300, disabled=True)


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
                with st.spinner("Đang suy luận và tìm kiếm..."):
                    # Để lấy scratchpad, chúng ta cần sửa đổi solver.solve()
                    # hoặc thêm một cách để truy cập nó.
                    # Hiện tại, chúng ta không có scratchpad trực tiếp từ solve()
                    # Giả sử solver.solve() trả về (final_answer, scratchpad_history)
                    # Cần sửa đổi dynamic_reasoning_solver.py để trả về scratchpad
                    
                    # ---- GIẢ ĐỊNH CẦN SỬA SOLVER ----
                    # final_answer, scratchpad_log = solver_instance.solve_with_log(prompt) 
                    # ---- KẾT THÚC GIẢ ĐỊNH ----

                    # Cách hiện tại (chỉ có final_answer)
                    final_answer = solver_instance.solve(prompt)
                    scratchpad_log = "Scratchpad logging not fully implemented in UI for this version. Full log in console."
                    # Để có scratchpad ở đây, solver.solve cần trả về nó, hoặc có một callback
                    # Hoặc chúng ta có thể capture stdout của solver (phức tạp hơn)

                full_response_text = final_answer
                message_placeholder.markdown(full_response_text)
                
                # Thêm scratchpad vào message nếu có (cần sửa solver)
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
