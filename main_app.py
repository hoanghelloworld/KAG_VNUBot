import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
torch.classes.__path__ = []
import streamlit as st
import sys
import time
import threading
from threading import Thread
import queue

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

import config
from dynamic_reasoning_solver import DynamicKAGSolver
import llm_utils

DEV_MODE = False
@st.cache_resource
def load_solver():

    required_files = [config.settings.FAISS_INDEX_PATH, config.settings.GRAPH_PATH, config.settings.DOC_STORE_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Application cannot start. Missing critical KAG artifact files: {', '.join(missing_files)}. "
                 "Please ensure the KAG data pipeline (crawler, processor, builder) has been run successfully.")
        return None
    try:
        solver = DynamicKAGSolver()
        return solver
    except Exception as e:
        st.error(f"Error initializing KAG Solver: {e}")
        st.error("Please check if models are downloaded and artifacts are correctly built.")
        return None

def run_solver_in_thread(solver, prompt, top_k=None, max_steps=None, result_queue=None):
    try:
        solver_args = {}
        if top_k is not None:
            solver_args['top_k_retrieval'] = top_k
        if max_steps is not None:
            solver_args['max_reasoning_steps'] = max_steps
        
        if solver_args:
            scratchpad, answer = solver.solve(prompt, **solver_args)
        else:
            scratchpad, answer = solver.solve(prompt)
                    
        if result_queue:
            result_queue.put((answer, scratchpad))
    except Exception as e:
        if result_queue:
            result_queue.put((f"Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý: {str(e)}", f"Error details: {str(e)}"))

st.set_page_config(page_title="KAG Reasoning Chatbot", layout="wide")

st.markdown("""
<style>
    .chat-message-container {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .assistant-message {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #d9d9d9;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .thinking-animation {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 20px;
    }
    .thinking-animation div {
        position: absolute;
        top: 8px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #1890ff;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    .thinking-animation div:nth-child(1) {
        left: 8px;
        animation: thinking1 0.6s infinite;
    }
    .thinking-animation div:nth-child(2) {
        left: 8px;
        animation: thinking2 0.6s infinite;
    }
    .thinking-animation div:nth-child(3) {
        left: 32px;
        animation: thinking2 0.6s infinite;
    }
    .thinking-animation div:nth-child(4) {
        left: 56px;
        animation: thinking3 0.6s infinite;
    }
    @keyframes thinking1 {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    @keyframes thinking2 {
        0% { transform: translate(0, 0); }
        100% { transform: translate(24px, 0); }
    }
    @keyframes thinking3 {
        0% { transform: scale(1); }
        100% { transform: scale(0); }
    }
</style>
""", unsafe_allow_html=True)

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
            <p style="text-align: center; margin-top: 20px; max-width: 600px;">
                Chào mừng bạn đến với VNU Chat - trợ lý thông minh hỗ trợ tìm kiếm thông tin về 
                quy định, chính sách và thủ tục giáo dục tại Đại học Quốc gia Hà Nội.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <img src="https://www.vnu.edu.vn/upload/2015/01/17449/image/Logo-VNU-1995.png" alt="VNU Logo" width="50" style="margin-right: 10px;">
            <h2 style="margin: 0;">VNU Chat</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

solver_instance = load_solver()

with st.sidebar:
    st.header("Cấu hình")
    
    show_advanced = st.checkbox("Hiển thị cài đặt nâng cao", False)
    
    if st.button("Xóa lịch sử hội thoại", use_container_width=True):
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.rerun()
    
    if show_advanced:
        st.subheader("Cài đặt tìm kiếm")
        
        top_k = st.slider(
            "Số lượng kết quả tìm kiếm (top_k)", 
            min_value=1, 
            max_value=10, 
            value=config.settings.TOP_K_RETRIEVAL,
            help="Số lượng đoạn văn bản liên quan nhất được truy xuất từ cơ sở tri thức"
        )
        
        max_steps = st.slider(
            "Số bước suy luận tối đa", 
            min_value=1, 
            max_value=10, 
            value=config.settings.MAX_REASONING_STEPS,
            help="Số bước suy luận tối đa khi giải quyết câu hỏi"
        )
        
        st.subheader("Thông tin về KAG")
        if not DEV_MODE and solver_instance:
            try:
                st.text(f"Số lượng node: {len(solver_instance.graph.nodes)}")
                st.text(f"Số lượng cạnh: {len(solver_instance.graph.edges)}")
                st.text(f"Số lượng chunk văn bản: {len(solver_instance.doc_store)}")
                
                entity_count = 0
                entity_types = {}
                for node_id, attrs in solver_instance.graph.nodes(data=True):
                    if attrs.get('type') == 'entity':
                        entity_count += 1
                        entity_type = attrs.get('entity_type', 'UNKNOWN')
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                st.text(f"Số lượng thực thể: {entity_count}")
                with st.expander("Chi tiết phân loại thực thể"):
                    for entity_type, count in entity_types.items():
                        st.text(f"- {entity_type}: {count}")
                
            except Exception as e:
                st.warning(f"Không thể truy xuất thông tin KAG: {str(e)}")
        else:
            if DEV_MODE:
                st.info("Đang chạy ở chế độ phát triển. Thông tin KAG không khả dụng.")
            else:
                st.warning("Solver không được khởi tạo. Thông tin KAG không khả dụng.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Về VNU Chat")
    st.sidebar.info(
        """
        VNU Chat là ứng dụng chatbot thông minh sử dụng công nghệ Knowledge Augmented Generation (KAG) 
        để trả lời các câu hỏi về quy định, chính sách và thủ tục giáo dục tại Đại học Quốc gia Hà Nội.
        
        Ứng dụng này sử dụng mô hình trí tuệ nhân tạo kết hợp với dữ liệu từ các tài liệu chính thức 
        của ĐHQGHN để cung cấp thông tin chính xác và đáng tin cậy.
        """
    )

if solver_instance is None:
    st.warning("Không thể khởi tạo trợ lý. Vui lòng kiểm tra lỗi trong console và đảm bảo các file KAG đã được tạo.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
        
    if "current_scratchpad" not in st.session_state:
        st.session_state.current_scratchpad = ""

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "scratchpad" in message:
                with st.expander("Xem quá trình suy luận"):
                    st.markdown("### Quá trình suy luận")
                    st.code(message["scratchpad"], language="text")

    if prompt := st.chat_input("Câu hỏi của bạn là gì?", disabled=st.session_state.is_processing):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if st.session_state.is_processing:
                message_placeholder.warning("Đang xử lý câu hỏi trước, vui lòng đợi...")
            else:
                st.session_state.is_processing = True
                
                message_placeholder.markdown("""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div>Đang suy luận và tìm kiếm thông tin</div>
                    <div class="thinking-animation" style="margin-left: 10px;">
                        <div></div><div></div><div></div><div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    result_queue = queue.Queue()
                    
                
                    solver_args = {}
                    if show_advanced:
                        solver_args['top_k'] = top_k
                        solver_args['max_steps'] = max_steps
                        st.info(f"Tìm kiếm với top_k={top_k}, max_steps={max_steps}")
                        
                    solver_thread = Thread(
                        target=run_solver_in_thread, 
                        args=(solver_instance, prompt),
                        kwargs={
                            'top_k': solver_args.get('top_k'),
                            'max_steps': solver_args.get('max_steps'),
                            'result_queue': result_queue
                        }
                    )
                    solver_thread.start()
                    solver_thread.join()
                    final_answer, scratchpad_log = result_queue.get()

                    message_placeholder.markdown(final_answer)
                    
                    with st.expander("Xem quá trình suy luận", expanded=True):
                        st.session_state.current_scratchpad_content = scratchpad_log
                        st.markdown("### Quá trình suy luận")
                        st.code(scratchpad_log, language="text")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "scratchpad": scratchpad_log
                    })
                    
                except Exception as e:
                    error_message = f"Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý: {str(e)}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message,
                        "scratchpad": f"Error details: {str(e)}"
                    })
                
                finally:
                    st.session_state.is_processing = False

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <small>© 2025 - VNU Chat - Powered by Knowledge Augmented Generation (KAG) và LLM</small>
    </div>
    """, 
    unsafe_allow_html=True
)
