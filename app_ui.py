# app_ui.py
import streamlit as st
import sys
import os
import time

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
        st.error(f"Ứng dụng không thể khởi động. Thiếu các file KAG cần thiết: {', '.join(missing_files)}. "
                 "Vui lòng chạy tiến trình xây dựng dữ liệu KAG (build_kag_vnu.py) trước.")
        return None
    try:
        # llm_utils.get_llm_model() # Pre-load models if not handled by solver's init
        # llm_utils.get_embedding_model()
        solver = DynamicKAGSolver()
        return solver
    except Exception as e:
        st.error(f"Lỗi khởi tạo KAG Solver: {e}")
        st.error("Vui lòng kiểm tra xem các model đã được tải xuống và artifacts đã được xây dựng chính xác chưa.")
        return None

# --- Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Hỏi Đáp VNU-KAG", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles ---
st.markdown("""
<style>
    .chat-title {
        text-align: center;
        color: #1E3A8A;
    }
    .source-link {
        font-size: 0.8em;
        color: #4B5563;
    }
    .source-content {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='chat-title'>🎓 Hệ thống Hỏi Đáp Đại học Quốc gia Hà Nội</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Trả lời các câu hỏi về quy chế, quy định học tập tại ĐHQGHN</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Thông tin hệ thống")
    st.markdown("Hệ thống sử dụng **KAG** (Knowledge Augmented Generation) để trả lời các câu hỏi về quy chế, quy định học tập tại ĐHQGHN.")
    
    st.subheader("Nguồn dữ liệu")
    st.markdown("""
    - Quy chế đào tạo đại học
    - Quy chế đào tạo thạc sĩ
    - Quy chế đào tạo tiến sĩ
    - Quy chế công tác sinh viên
    - Quy định về học phí, khen thưởng
    - Thông tin về trường ĐHCN và ĐHQGHN
    """)
    
    st.subheader("Gợi ý câu hỏi")
    st.markdown("""
    - Khi nào sinh viên bị buộc thôi học?
    - Thời gian đào tạo tiến sĩ là bao lâu?
    - Điều kiện để được công nhận tốt nghiệp đại học?
    - Sinh viên được đăng ký tối đa bao nhiêu tín chỉ một học kỳ?
    """)
    
    # Nút Clear Chat
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# --- Load solver ---
solver_instance = load_solver()

if solver_instance is None:
    st.warning("Không thể tải KAG Solver. Vui lòng kiểm tra console để xem lỗi và đảm bảo các artifacts KAG đã được tạo.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sources" in message:
                st.markdown(message["content"])
                with st.expander("📚 Xem nguồn tham khảo"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.markdown(f"<div class='source-content'>{source['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
            
            if "reasoning" in message and message["reasoning"]:
                with st.expander("🧠 Xem quá trình suy luận"):
                    st.markdown(message["reasoning"])

    # React to user input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            
            try:
                with st.spinner("🔍 Đang tìm kiếm và suy luận..."):
                    result = solver_instance.solve_with_sources(prompt)
                    
                    if isinstance(result, dict):
                        # New format with sources and reasoning
                        full_response_text = result.get("answer", "Không tìm thấy câu trả lời.")
                        sources = result.get("sources", [])
                        reasoning = result.get("reasoning", "")
                    else:
                        # Old format (just answer string)
                        full_response_text = result
                        sources = []
                        reasoning = ""
                
                # Display main answer first
                message_placeholder.markdown(full_response_text)
                
                # If we have sources, show them in an expander
                if sources:
                    with st.expander("📚 Xem nguồn tham khảo"):
                        for source in sources:
                            st.markdown(f"**{source['title']}**")
                            st.markdown(f"<div class='source-content'>{source['content']}</div>", unsafe_allow_html=True)
                
                # Store assistant message with all components
                assistant_message = {
                    "role": "assistant", 
                    "content": full_response_text,
                    "sources": sources if sources else [],
                    "reasoning": reasoning if reasoning else ""
                }

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                full_response_text = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."
                message_placeholder.markdown(full_response_text)
                assistant_message = {"role": "assistant", "content": full_response_text}
        
        st.session_state.messages.append(assistant_message)

    # --- Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>© 2024 VNU-KAG | Sử dụng KAG Framework</p>", unsafe_allow_html=True)

# TODO:  Thêm các tính năng UI khác:
#       - Nút "Clear Chat".
#       - Hiển thị thông tin về KG (ví dụ: số lượng node, cạnh) - có thể ở sidebar.
#       - Cho phép upload tài liệu mới để build KG (nếu có chức năng đó).
