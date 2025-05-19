# dynamic_reasoning_solver.py
import faiss
import networkx as nx
import json
import os
import re

from config import settings, prompt_manager
import llm_utils # Sử dụng get_llm_response

class DynamicKAGSolver:
    def __init__(self, top_k_retrieval=settings.TOP_K_RETRIEVAL, max_reasoning_steps=settings.MAX_REASONING_STEPS):
        # --- Tải Artifacts ---
        print("SOLVER: Loading FAISS index...")
        self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
        
        print("SOLVER: Loading Knowledge Graph...")
        self.graph = nx.read_gml(settings.GRAPH_PATH)
        
        print("SOLVER: Loading Doc Store and FAISS ID Map...")
        with open(settings.DOC_STORE_PATH, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            self.doc_store = saved_data['doc_store']
            # faiss_id_map từ string (do JSON) sang int
            self.faiss_id_to_chunk_id = {int(k): v for k, v in saved_data['faiss_id_map'].items()}
        print("SOLVER: All artifacts loaded.")

        self.top_k = top_k_retrieval
        self.max_reasoning_steps = max_reasoning_steps

        # Tinh chỉnh prompt dựa trên thử nghiệm
        self.reason_act_prompt_template_str = """
Bạn là một agent lý luận chuyên trả lời các câu hỏi về quy chế, chính sách và thủ tục giáo dục tại ĐHQGHN (Đại học Quốc gia Hà Nội). Hãy phân tích kỹ Câu hỏi gốc để hiểu rõ thông tin được yêu cầu.

Bạn có quyền truy cập vào các công cụ sau:
1. `search_vector_db(search_query: str)`: Tìm kiếm trong cơ sở dữ liệu vector các đoạn văn bản (chunks) liên quan dựa trên `search_query`. Sử dụng công cụ này để tìm các quy định, định nghĩa hoặc thông tin theo ngữ cảnh. Hãy đặt câu hỏi tìm kiếm cụ thể và bằng tiếng Việt khi thích hợp.
   Ví dụ:
   - `search_vector_db("quy định về học phí đại học quốc gia Hà Nội")`
2. `query_kg(subject: str, relation: str, object: str, query_type: str)`: Truy vấn Đồ thị Tri thức (Knowledge Graph). Sử dụng công cụ này để tìm các thực thể cụ thể, thuộc tính của chúng hoặc mối quan hệ giữa chúng.
   Các giá trị `query_type` có sẵn:
   - `find_relation`: Cho chủ thể (subject) và đối tượng (object), tìm mối quan hệ của chúng. (ví dụ: query_kg("Học phần A", "?", "Chương trình B", query_type="find_relation"))
   - `find_object`: Cho chủ thể (subject) và mối quan hệ (relation), tìm (các) đối tượng. (ví dụ: query_kg("John McCarthy", "tạo ra", "?", query_type="find_object"))
   - `find_subject`: Cho mối quan hệ (relation) và đối tượng (object), tìm (các) chủ thể. (ví dụ: query_kg("?", "đặt tại", "Hà Nội", query_type="find_subject"))
   - `get_entity_details`: Cho một thực thể, lấy các thuộc tính và mối quan hệ trực tiếp của nó. (ví dụ: query_kg("Học phí", "?", "?", query_type="get_entity_details"))
   - `find_mentioning_chunks`: Cho một thực thể, tìm các đoạn văn bản đề cập đến nó. (ví dụ: query_kg("AI", "mentioned_in_chunk", "?", query_type="find_mentioning_chunks"))
   - `find_entity_by_type`: Cho một loại thực thể, liệt kê các thực thể thuộc loại đó. (ví dụ: query_kg("COURSE", "entity_type", "?", query_type="find_entity_by_type") hoặc query_kg(subject="?", relation="entity_type", object="COURSE", query_type="find_subject"))
   - `list_entities`: Liệt kê một mẫu các thực thể trong KG. (ví dụ: query_kg("?", "?", "?", query_type="list_entities"))
   Sử dụng '?' cho các phần chưa biết trong subject, relation, hoặc object cho các loại truy vấn phù hợp.
3. `finish(answer: str)`: Cung cấp câu trả lời cuối cùng cho Câu hỏi gốc. Chỉ sử dụng công cụ này khi bạn tự tin rằng mình đã có câu trả lời đầy đủ và chính xác.

4. Lưu ý về Knowledge Graph:
   - KG được xây dựng từ các văn bản quy chế và tài liệu giáo dục.
   - Các loại thực thể (ENTITY TYPES) thường gặp bao gồm:
     - PROGRAM: Các chương trình giáo dục (ví dụ: "Chương trình đào tạo tiên tiến").
     - COURSE: Tên môn học hoặc mã môn học (ví dụ: "Giải tích I").
     - REGULATION: Tên các quy định, quy chế, chính sách (ví dụ: "Quy chế đào tạo đại học").
     - DOCUMENT: Các định danh tài liệu (ví dụ: "Quyết định số 123/QĐ-ĐHQGHN").
     - REQUIREMENT: Các yêu cầu, chuẩn mực (ví dụ: "Chuẩn đầu ra tiếng Anh").
     - ACADEMIC_UNIT: Khoa, phòng, ban, viện, đơn vị học thuật (ví dụ: "Khoa Công nghệ Thông tin", "Đại học Quốc gia Hà Nội").
     - ROLE: Chức danh hoặc vai trò (ví dụ: "Hiệu trưởng", "Sinh viên").
     - CREDIT: Thông tin tín chỉ.
     - EVALUATION: Phương pháp đánh giá.
     - FEE: Học phí, lệ phí.
     - DEADLINE: Mốc thời gian, hạn chót.
   - Các loại mối quan hệ (RELATION TYPES) mô tả thường gặp bao gồm:
     - "thuộc_về": Thực thể là một phần của thực thể khác (ví dụ: COURSE "Giải tích I" thuộc_về PROGRAM "Chương trình Tiên tiến").
     - "yêu_cầu": Thực thể này yêu cầu thực thể kia (ví dụ: COURSE "Giải tích II" yêu_cầu COURSE "Giải tích I").
     - "được_quản_lý_bởi": Ví dụ: ACADEMIC_UNIT "Khoa CNTT" được_quản_lý_bởi ACADEMIC_UNIT "Trường ĐH Công nghệ".
     - "được_định_nghĩa_trong": Ví dụ: REQUIREMENT "Chuẩn đầu ra tiếng Anh" được_định_nghĩa_trong REGULATION "Quy chế đào tạo".
     - "bao_gồm_thành_phần": Ví dụ: PROGRAM "Cử nhân CNTT" bao_gồm_thành_phần COURSE "Lập trình cơ bản".
     - "áp_dụng_cho": Ví dụ: REGULATION "Quy chế học bổng" áp_dụng_cho ROLE "Sinh viên".
     - "có_liên_quan_đến": Mối liên hệ chung.
     - Và các mối quan hệ khác như: "được_đánh_giá_bằng", "tương_đương_với", "kế_thừa_bởi", "được_tạo_bởi", "được_sửa_đổi_bởi".
   
   - QUAN TRỌNG: KHÔNG TỰ TẠO RA các mối quan hệ mới khi sử dụng query_kg. Chỉ sử dụng các mối quan hệ đã liệt kê ở trên hoặc những mối quan hệ đã được xác nhận tồn tại trong KG.
   - Khi không chắc chắn về mối quan hệ chính xác, trước tiên hãy sử dụng query_type="get_entity_details" để tìm hiểu các mối quan hệ hiện có của thực thể đó.
   - Nếu không tìm thấy mối quan hệ cụ thể trong KG, hãy sử dụng search_vector_db để tìm kiếm thông tin trong văn bản.

Quy trình Lý luận:
1. Hiểu Câu hỏi gốc và chia nhỏ nếu phức tạp.
2. Sử dụng các công cụ để thu thập thông tin từng bước. Thêm các quan sát vào scratchpad của bạn.
   - Khi cần tìm kiếm thông tin về một thực thể trong KG, đầu tiên hãy sử dụng query_kg với query_type="get_entity_details" để xem thực thể đó có tồn tại và có những mối quan hệ nào.
   - Chỉ sử dụng những mối quan hệ đã được xác nhận tồn tại, tránh tự tạo ra các mối quan hệ không có trong hệ thống.
3. Phân tích thông tin để xác định những điểm còn thiếu hoặc mâu thuẫn.
4. Tiếp tục thu thập thông tin cho đến khi bạn có đủ để trả lời hoàn chỉnh.
5. Cung cấp một câu trả lời toàn diện bằng `finish()` khi sẵn sàng.

Định dạng phản hồi của bạn MỘT CÁCH NGHIÊM NGẶT như sau:
Thought: [Quá trình suy nghĩ của bạn]
Action: [Gọi một trong các công cụ, ví dụ: `search_vector_db("quy định học phí")` hoặc `finish("Theo quy định...")`]

Câu hỏi gốc: {original_query}

Scratchpad (lịch sử suy nghĩ, hành động và quan sát trước đó):
--- SCRATCHPAD START ---
{scratchpad}
--- SCRATCHPAD END ---

Thought:
"""
        self.stop_sequences_for_llm_action = ["Action:"] # Để LLM dừng sau khi tạo Thought

    def _retrieve_from_vector_db(self, search_query):
        """Truy xuất từ Vector DB."""
        print(f"  SOLVER Action: search_vector_db(\"{search_query}\")")
        if not search_query:
            return "Observation: Vector DB search query was empty. Please provide a specific query."
            
        query_embedding = llm_utils.get_embeddings([search_query]).cpu().numpy()
        distances, faiss_numeric_indices = self.faiss_index.search(query_embedding, self.top_k)
        
        retrieved_chunk_infos = []
        for i in range(len(faiss_numeric_indices[0])):
            numeric_id = faiss_numeric_indices[0][i]
            if (numeric_id != -1): # FAISS trả về -1 nếu không đủ k kết quả
                chunk_id_str = self.faiss_id_to_chunk_id.get(numeric_id)
                if chunk_id_str and chunk_id_str in self.doc_store:
                    # Rút gọn context hiển thị để tránh làm LLM bị quá tải
                    chunk_text_preview = self.doc_store[chunk_id_str][:300] + "..." \
                                         if len(self.doc_store[chunk_id_str]) > 300 else self.doc_store[chunk_id_str]
                    retrieved_chunk_infos.append(f"- Chunk ID: {chunk_id_str}\n  Content Preview: {chunk_text_preview}")
        
        if not retrieved_chunk_infos:
            return f"Observation: No relevant chunks found in vector DB for query: \"{search_query}\"."
        return f"Observation: Retrieved from vector DB for \"{search_query}\":\n" + "\n".join(retrieved_chunk_infos)

    def _query_kg_advanced(self, subject_str, relation_str, object_str, query_type_str):
        """
        Thực hiện truy vấn trên Knowledge Graph.
        """
        print(f"  SOLVER Action: query_kg(\"{subject_str}\", \"{relation_str}\", \"{object_str}\", query_type=\"{query_type_str}\")")
        
        subj = subject_str.lower().strip() if subject_str else "?"
        rel = relation_str.lower().strip() if relation_str else "?"
        obj = object_str.lower().strip() if object_str else "?"
        q_type = query_type_str.lower().strip()

        results = []
        
        if q_type == "find_mentioning_chunks":
            # Ví dụ: query_kg("AI", "mentioned_in_chunk", "?", query_type="find_mentioning_chunks")
            # Mong muốn subj là một thực thể
            if subj != "?" and self.graph.has_node(subj) and self.graph.nodes[subj].get('type') == 'entity':
                for neighbor_id, edge_data in self.graph.adj[subj].items():
                    if edge_data.get('type') == 'mentions_entity' and self.graph.nodes[neighbor_id].get('type') == 'chunk':
                         results.append(f"Entity '{subj}' is mentioned in Chunk ID: '{neighbor_id}' (Source Doc: {self.graph.nodes[neighbor_id].get('source_document_id','N/A')})")

        elif q_type == "find_relation":
            # Ví dụ: query_kg("John McCarthy", "?", "Artificial Intelligence", query_type="find_relation")
            # Tìm mối quan hệ giữa hai thực thể (nếu cả hai đều có trong KG)
            if subj != "?" and obj != "?" and self.graph.has_node(subj) and self.graph.has_node(obj):
                if self.graph.has_edge(subj, obj):
                    for edge_key, edge_data in self.graph.get_edge_data(subj, obj).items(): # Có thể có nhiều cạnh giữa 2 node
                        if edge_data.get('type') == 'kg_relation': # Chỉ lấy cạnh quan hệ KG
                            relation_label = edge_data.get('relation_label', 'related_to')
                            if rel == "?" or rel == relation_label.lower():
                                results.append(f"Found KG Relation: ('{subj}', '{relation_label}', '{obj}') from chunk '{edge_data.get('source_chunk_id','N/A')}'")
                
                # Tìm đường đi nếu không có cạnh trực tiếp
                if not results:
                    try:
                        # Tìm đường đi ngắn nhất giữa hai node
                        path = nx.shortest_path(self.graph, source=subj, target=obj)
                        if len(path) > 2:  # Đường đi có ít nhất một node trung gian
                            path_str = " -> ".join(path)
                            results.append(f"Indirect path found between '{subj}' and '{obj}': {path_str}")
                            # Thêm chi tiết về các cạnh trong đường đi
                            for i in range(len(path)-1):
                                source, target = path[i], path[i+1]
                                for edge_key, edge_data in self.graph.get_edge_data(source, target).items():
                                    edge_type = edge_data.get('type', 'unknown')
                                    relation = edge_data.get('relation_label', 'related_to') if edge_type == 'kg_relation' else edge_type
                                    results.append(f"  - Step {i+1}: ('{source}', '{relation}', '{target}')")
                    except nx.NetworkXNoPath:
                        results.append(f"No path found between '{subj}' and '{obj}' in the knowledge graph.")
        
        elif q_type == "find_object":
            # Ví dụ: query_kg("John McCarthy", "created", "?", query_type="find_object")
            # Tìm các đối tượng có quan hệ với chủ thể
            if subj != "?" and rel != "?" and self.graph.has_node(subj):
                # self.graph.adj[subj] trả về một dict {neighbor_id: {edge_key: edge_attributes_dict, ...}, ...}
                # cho MultiGraph/MultiDiGraph
                for neighbor_id, edges_to_neighbor in self.graph.adj[subj].items():
                    for edge_key, edge_data in edges_to_neighbor.items(): # Lặp qua từng cạnh cụ thể đến neighbor_id
                        if isinstance(edge_data, dict) and \
                           edge_data.get('type') == 'kg_relation' and \
                           edge_data.get('relation_label', '').lower() == rel:
                            results.append(f"Found object: '{neighbor_id}' related to '{subj}' via '{rel}' from chunk '{edge_data.get('source_chunk_id','N/A')}'")

        elif q_type == "find_subject":
            # Ví dụ: query_kg("?", "created", "Lisp", query_type="find_subject")
            # Tìm các chủ thể có quan hệ với đối tượng
            if obj != "?" and rel != "?" and self.graph.has_node(obj):
                # self.graph.pred[obj] (cho DiGraph/MultiDiGraph) hoặc self.graph.adj[obj] (cho Graph/MultiGraph không hướng)
                # Giả sử là đồ thị có hướng (DiGraph hoặc MultiDiGraph)
                # self.graph.pred[obj] trả về {predecessor_id: {edge_key: edge_attributes_dict, ...}, ...}
                # Nếu là đồ thị không hướng, bạn sẽ dùng self.graph.adj[obj]
                # và logic sẽ tương tự như find_object
                
                # Giả sử đồ thị có hướng (ví dụ: MultiDiGraph)
                if hasattr(self.graph, 'pred'): # Kiểm tra xem có phải là đồ thị có hướng không
                    predecessors_adj = self.graph.pred[obj]
                else: # Nếu là đồ thị không hướng, dùng adj
                    predecessors_adj = self.graph.adj[obj]

                for pred_id, edges_from_pred in predecessors_adj.items():
                    for edge_key, edge_data in edges_from_pred.items(): # Lặp qua từng cạnh cụ thể từ pred_id đến obj
                        if isinstance(edge_data, dict) and \
                           edge_data.get('type') == 'kg_relation' and \
                           edge_data.get('relation_label', '').lower() == rel:
                            # Kiểm tra xem cạnh có đúng hướng không nếu là đồ thị có hướng
                            # Trong trường hợp này, chúng ta đang tìm các cạnh đi VÀO obj từ pred_id
                            # self.graph.get_edge_data(pred_id, obj) sẽ cho chúng ta các cạnh từ pred_id đến obj
                            # Nếu bạn dùng self.graph.adj[obj] cho đồ thị không hướng, thì không cần kiểm tra hướng
                            
                            # Với self.graph.pred[obj], các cạnh đã đúng hướng (pred_id -> obj)
                            results.append(f"Found subject: '{pred_id}' related to '{obj}' via '{rel}' from chunk '{edge_data.get('source_chunk_id','N/A')}'")
 
        elif q_type == "get_entity_details":
            # Ví dụ: query_kg("John McCarthy", "?", "?", query_type="get_entity_details")
            # Lấy thuộc tính của một thực thể
            try:
                if subj != "?" and self.graph.has_node(subj):
                    node_attrs = self.graph.nodes[subj]
                    # Lọc ra các thuộc tính người dùng định nghĩa, không phải của networkx
                    custom_attrs = {k:v for k,v in node_attrs.items() if k not in ['type', 'text_preview', 'original_text_forms']}
                    if custom_attrs:
                        results.append(f"Attributes for '{subj}': {custom_attrs}")
                    
                    # Lấy tất cả các relation của entity này để hiển thị như attributes
                    for neighbor_id, edge_dict in self.graph.adj[subj].items():
                        for edge_key, edge_data in edge_dict.items():
                            print(f"edge_dict: {edge_dict}")
                            print(f"edge_key: {edge_key}")
                            print(f"edge_data: {edge_data}")
                            if edge_data.get('type') == 'kg_relation':
                                relation = edge_data.get('relation_label', 'related_to')
                                results.append(f"Relation: '{subj}' --[{relation}]--> '{neighbor_id}'")
                    
                    if not results:
                        results.append(f"No specific attributes or relations found for '{subj}' in KG.")
            except Exception as e: 
                return f"Observation: Error querying KG for entity '{subj}': {str(e)}"
                    
        elif q_type == "find_entity_by_type":
            # Ví dụ: query_kg("person", "entity_type", "?", query_type="find_entity_by_type")
            # Tìm các thực thể thuộc loại cụ thể
            entity_type = subj.lower()
            for node_id, node_attrs in self.graph.nodes(data=True):
                if (node_attrs.get('type') == 'entity' and 
                    node_attrs.get('entity_type', '').lower() == entity_type):
                    results.append(f"Found {entity_type}: '{node_id}'")
        
        elif q_type == "list_entities":
            # Liệt kê tất cả các thực thể trong KG
            entity_count = 0
            sample_entities = []
            for node_id, node_attrs in self.graph.nodes(data=True):
                if node_attrs.get('type') == 'entity':
                    entity_count += 1
                    if len(sample_entities) < 10:  # Lấy mẫu 10 entity đầu tiên
                        entity_type = node_attrs.get('entity_type', 'unknown')
                        sample_entities.append(f"'{node_id}' (type: {entity_type})")
            
            results.append(f"Found {entity_count} entities in the knowledge graph.")
            if sample_entities:
                results.append(f"Sample entities: {', '.join(sample_entities)}")
                if entity_count > 10:
                    results.append("Use more specific queries to explore other entities.")

        if not results:
            return f"Observation: No results found in KG for query_kg(\"{subject_str}\", \"{relation_str}\", \"{object_str}\", query_type=\"{q_type}\")."
        return f"Observation: KG query results for query_type='{q_type}':\n" + "\n".join([f"- {r}" for r in results])


    def _parse_llm_action_output(self, llm_output_str):
        """
        Phân tích output của LLM để lấy Thought và Action.
        Xử lý nhiều định dạng output khác nhau và xử lý các trường hợp lỗi.
        """
        thought = ""
        action_type = "error" # Mặc định là lỗi nếu không parse được
        action_input = "Could not parse LLM action."

        # Tìm Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", llm_output_str, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        else: # Nếu không có "Thought:", có thể toàn bộ là action hoặc lỗi
            thought = "No specific thought parsed."

        # Tìm Action
        action_full_str_match = re.search(r"Action:\s*(.+)", llm_output_str, re.DOTALL | re.IGNORECASE)
        if not action_full_str_match:
             # Nếu không có "Action:" và thought cũng không có, có thể LLM chỉ trả lời (coi như finish)
            if not thought_match and llm_output_str.strip(): # Nếu có text nhưng ko match pattern
                return llm_output_str.strip(), "finish", llm_output_str.strip() # Toàn bộ output là câu trả lời
            return thought, "error", "No 'Action:' directive found in LLM output."
        
        action_call_str = action_full_str_match.group(1).strip()

        # Phân tích các loại action
        # Hỗ trợ nhiều định dạng khác nhau mà LLM có thể tạo ra
        
        # 1. Xử lý finish() action
        finish_match = re.search(r'finish\s*\((.+?)(?:\)|$)', action_call_str, re.DOTALL)
        if finish_match:
            action_type = "finish"
            # Xử lý các kiểu quote khác nhau và chạy qua nhiều dòng
            finish_text = finish_match.group(1).strip()
            
            # Xử lý trường hợp text có quote
            if (finish_text.startswith('"') and finish_text.endswith('"')) or \
               (finish_text.startswith("'") and finish_text.endswith("'")):
                action_input = finish_text[1:-1]
            else:
                action_input = finish_text
                
            # Xử lý các ký tự thoát
            action_input = action_input.replace('\\"', '"').replace("\\'", "'")
            return thought, action_type, action_input
            
        # 2. Xử lý search_vector_db() action
        search_db_match = re.search(r'search_vector_db\s*\((.+?)(?:\)|$)', action_call_str, re.DOTALL)
        if search_db_match:
            action_type = "search_vector_db"
            search_query = search_db_match.group(1).strip()
            
            # Xử lý trường hợp query có quote
            if (search_query.startswith('"') and search_query.endswith('"')) or \
               (search_query.startswith("'") and search_query.endswith("'")):
                action_input = search_query[1:-1]
            else:
                action_input = search_query
                
            # Xử lý các ký tự thoát
            action_input = action_input.replace('\\"', '"').replace("\\'", "'")
            return thought, action_type, action_input
        
        # 3. Xử lý query_kg() action - phức tạp nhất vì có nhiều tham số
        query_kg_match = re.search(r'query_kg\s*\((.+?)(?:\)|$)', action_call_str, re.DOTALL)
        if query_kg_match:
            action_type = "query_kg"
            args_str = query_kg_match.group(1).strip()
            
            try:
                # Phương pháp 1: Xử lý trường hợp JSON format
                if args_str.startswith('[') and args_str.endswith(']'):
                    try:
                        json_str = args_str.replace("'", '"')  # Chuyển single quotes sang double quotes cho JSON
                        parsed_args = json.loads(json_str)
                        if len(parsed_args) == 4:
                            return thought, action_type, tuple(parsed_args)
                    except json.JSONDecodeError:
                        pass  # Nếu không phải JSON hợp lệ, thử phương pháp khác
                
                # Phương pháp 2: Xử lý trường hợp tham số có tên
                # Ví dụ: query_kg(subject="Entity", relation="rel", object="obj", query_type="find_relation")
                named_args = {}
                named_arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]*))'
                for match in re.finditer(named_arg_pattern, args_str):
                    arg_name = match.group(1)
                    # Lấy giá trị từ một trong các capturing group
                    arg_value = match.group(2) or match.group(3) or match.group(4) or ""
                    named_args[arg_name] = arg_value
                    
                if 'subject' in named_args and 'relation' in named_args and 'object' in named_args and 'query_type' in named_args:
                    return thought, action_type, (
                        named_args['subject'], 
                        named_args['relation'], 
                        named_args['object'], 
                        named_args['query_type']
                    )
                
                # Phương pháp 3: Sử dụng regex để tách các tham số có quote
                # Tìm các chuỗi được quote hoặc các token không có dấu phẩy
                args_pattern = r'(?:"([^"]*)"|\'([^\']*)\'|([^,\s][^,]*[^,\s]))'
                matches = re.findall(args_pattern, args_str)
                
                # Mỗi match sẽ là một tuple với các nhóm capture
                parsed_args = []
                for match_groups in matches:
                    # Lấy giá trị không rỗng đầu tiên trong nhóm
                    arg = next((group for group in match_groups if group), "")
                    parsed_args.append(arg)
                
                # Phương pháp 4: phương pháp đơn giản nhất, dùng khi các phương pháp khác thất bại
                if not parsed_args:
                    parsed_args = [arg.strip().strip('"').strip("'") 
                                  for arg in re.findall(r'(?:[^,"]|"[^"]*")+', args_str)]
                
                if len(parsed_args) == 4:  # subject, relation, object, query_type
                    return thought, action_type, tuple(parsed_args)
                else:
                    return thought, "error", f"Invalid number of arguments for query_kg. Expected 4, got {len(parsed_args)}. Args: {args_str}"
                    
            except Exception as e:
                return thought, "error", f"Error parsing query_kg arguments '{args_str}': {str(e)}"
        
        # Nếu không match với bất kỳ action pattern nào
        # Kiểm tra xem có phải LLM đã viết action dưới dạng text tự nhiên không
        # Ví dụ "I want to search for..." hoặc "Let's query the knowledge graph..."
        
        if "search" in action_call_str.lower() and "vector" in action_call_str.lower():
            # Tìm một chuỗi được quote - nếu có
            search_query_match = re.search(r'"([^"]+)"|\'([^\']+)\'', action_call_str)
            if search_query_match:
                search_query = search_query_match.group(1) or search_query_match.group(2)
                return thought, "search_vector_db", search_query
            else:
                # Nếu không có quote, dùng toàn bộ phần còn lại
                search_text = action_call_str.lower().replace("search", "").replace("vector_db", "").replace("vector db", "").strip()
                if search_text:
                    return thought, "search_vector_db", search_text
        
        if "query" in action_call_str.lower() and "kg" in action_call_str.lower():
            # Cố gắng trích xuất các phần tử từ text tự nhiên
            # Đây là phương pháp dự phòng và có thể không hoạt động trong tất cả các trường hợp
            try:
                # Tìm chuỗi query_type nếu có
                query_type_match = re.search(r'type\s*[=:]\s*["\']?(\w+)["\']?', action_call_str, re.IGNORECASE)
                query_type = query_type_match.group(1) if query_type_match else "find_relation"
                
                # Tìm các tham số khác
                subject_match = re.search(r'subject\s*[=:]\s*["\']?([^"\']+)["\']?', action_call_str, re.IGNORECASE)
                relation_match = re.search(r'relation\s*[=:]\s*["\']?([^"\']+)["\']?', action_call_str, re.IGNORECASE)
                object_match = re.search(r'object\s*[=:]\s*["\']?([^"\']+)["\']?', action_call_str, re.IGNORECASE)
                
                subject = subject_match.group(1).strip() if subject_match else "?"
                relation = relation_match.group(1).strip() if relation_match else "?"
                object_val = object_match.group(1).strip() if object_match else "?"
                
                return thought, "query_kg", (subject, relation, object_val, query_type)
            except Exception:
                pass  # Nếu cách này thất bại, tiếp tục xuống xử lý lỗi
        
        if "finish" in action_call_str.lower():
            # Trích xuất phần còn lại sau từ khóa finish
            answer_match = re.search(r'finish\s+(?:with|answer:?)?\s*["\']?(.+)', action_call_str, re.IGNORECASE)
            if answer_match:
                return thought, "finish", answer_match.group(1).strip().strip('"').strip("'")
        
        # Nếu tất cả các cách trên đều thất bại
        return thought, "error", f"Unknown or malformed action: {action_call_str}"


    def solve(self, original_query):
        scratchpad = f"Task: Answer the Original Query.\n" # Bắt đầu scratchpad
        
        for step in range(self.max_reasoning_steps):
            print(f"\n--- SOLVER: Reasoning Step {step + 1}/{self.max_reasoning_steps} ---")
            
            prompt_input = {
                "original_query": original_query,
                "scratchpad": scratchpad
            }
            current_prompt = self.reason_act_prompt_template_str.format(**prompt_input)
            
            # print(f"\nSOLVER Current Prompt to LLM:\n{current_prompt}\n") # DEBUG
            
            # Yêu cầu LLM tạo Thought trước, sau đó dừng để chúng ta parse
            # llm_thought_output = llm_utils.get_llm_response(current_prompt, max_new_tokens=150, stop_sequences=self.stop_sequences_for_llm_action, system_message="You are a reasoning agent. First provide your Thought, then stop before Action.")
            # print(f"LLM Thought Output: {llm_thought_output}")
            # # Sau đó, có thể tạo prompt mới chỉ để lấy Action, hoặc parse cả hai từ một response
            
            # Cách tiếp cận đơn giản hơn: lấy cả thought và action trong 1 lần gọi
            llm_full_output = llm_utils.get_llm_response(current_prompt, max_new_tokens=3000, system_message="You are a reasoning agent following instructions precisely.")
            print(f"LLM Full Output (Thought & Action):\n{llm_full_output}")

            thought, action_type, action_input = self._parse_llm_action_output(llm_full_output)
            
            scratchpad += f"\nStep {step + 1}:\nThought: {thought}\n"
            print(f"Thought: {thought}")

            if action_type == "finish":
                final_answer = action_input
                scratchpad += f"Action: finish(\"{final_answer}\")\n---\nFinal Answer Provided.\n"
                print(f"  SOLVER Action: finish(\"{final_answer}\")")
                print(f"\n--- SOLVER: Final Answer Determined ---")
                return final_answer
            
            elif action_type == "search_vector_db":
                scratchpad += f"Action: search_vector_db(\"{action_input}\")\n"
                observation = self._retrieve_from_vector_db(action_input)
                scratchpad += f"{observation}\n"
                print(observation)
            
            elif action_type == "query_kg":
                # action_input là tuple (subj, rel, obj, q_type)
                scratchpad += f"Action: query_kg(subject=\"{action_input[0]}\", relation=\"{action_input[1]}\", object=\"{action_input[2]}\", query_type=\"{action_input[3]}\")\n"
                observation = self._query_kg_advanced(*action_input)
                scratchpad += f"{observation}\n"
                print(observation)

            elif action_type == "error":
                error_message = f"Action Error: {action_input}\n"
                scratchpad += error_message
                print(f"  SOLVER {error_message.strip()}")
                # TODO: Có thể thêm logic để LLM thử lại với thông báo lỗi, hoặc dừng hẳn
                # For now, just stop.
                break 
            
            if len(scratchpad) > 3500: # Giới hạn độ dài scratchpad
                print("SOLVER: Scratchpad too long, truncating...")
                # Giữ lại phần đầu và phần cuối
                header = scratchpad[:500]
                footer = scratchpad[-2500:]
                scratchpad = header + "\n... (scratchpad truncated) ...\n" + footer


        # Nếu hết số bước mà chưa finish
        print("\n--- SOLVER: Max reasoning steps reached. Attempting to synthesize final answer. ---")
        # Tinh chỉnh prompt tổng hợp để tạo câu trả lời chất lượng cao
        final_synthesis_prompt = f"""
Câu hỏi gốc: {original_query}

Bạn đã trải qua một quá trình lý luận để trả lời câu hỏi này về các quy định, chính sách và thủ tục giáo dục tại ĐHQGHN (Đại học Quốc gia Hà Nội). Đây là scratchpad của bạn chứa đựng những suy nghĩ, hành động và quan sát:
--- SCRATCHPAD START ---
{scratchpad}
--- SCRATCHPAD END ---

Dựa trên tất cả thông tin đã thu thập được trong scratchpad của bạn:
1. Cung cấp một câu trả lời cuối cùng toàn diện, có cấu trúc tốt cho Câu hỏi gốc.
2. Chỉ bao gồm thông tin được hỗ trợ trực tiếp bởi bằng chứng trong scratchpad.
3. Hãy cụ thể và trích dẫn các nguồn thông tin nếu có thể (chunks, tên tài liệu).
4. Nếu có nhiều khía cạnh cho câu trả lời, hãy sắp xếp chúng thành các phần rõ ràng.
5. Nếu bạn gặp thông tin mâu thuẫn, hãy giải thích sự khác biệt.
6. Nếu bạn không thể trả lời các phần của câu hỏi, hãy nêu rõ thông tin nào còn thiếu.
7. Trả lời bằng ngôn ngữ của Câu hỏi gốc (Tiếng Việt hoặc Tiếng Anh).

Câu trả lời cuối cùng (ngắn gọn nhưng đầy đủ):
"""
        final_answer = llm_utils.get_llm_response(final_synthesis_prompt, max_new_tokens=3000, system_message="Bạn là một chuyên gia phân tích chính sách giáo dục chuyên về các quy định và thủ tục học vụ tại ĐHQGHN. Nhiệm vụ của bạn là tổng hợp thông tin một cách chính xác và toàn diện.")
        return final_answer

if __name__ == "__main__":
    # TODO:  Chạy file này sau khi Người 2 đã tạo artifacts (FAISS, GML, DocStore).
    #       Đảm bảo llm_utils.py và settings.py đã sẵn sàng.
    
    # Kiểm tra sự tồn tại của artifacts
    required_files = [settings.FAISS_INDEX_PATH, settings.GRAPH_PATH, settings.DOC_STORE_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Solver cannot start. Missing artifact files: {', '.join(missing_files)}")
        print("Please run the KAGBuilder (advanced_kag_builder.py) first.")
    else:
        print("All artifacts found. Initializing DynamicKAGSolver...")
        solver = DynamicKAGSolver()
        
        # Test queries
        # query1 = "What are the prerequisites for the 'Data Science Capstone' course and who teaches 'Introduction to AI'?"
        # print(f"\nSolving Query 1: {query1}")
        # answer1 = solver.solve(query1)
        # print(f"\n--- FINAL ANSWER (Query 1) ---\n{answer1}")

        # query2 = "Các trường đại học trực thuộc Đại học quốc gia hà Nội năm 2025"
        query2 = "Số yêu cầu để trở thành tiến sĩ tại Đại học quốc gia Hà Nội"
        
        print(f"\nSolving Query 2: {query2}")
        answer2 = solver.solve(query2)
        print(f"\n--- FINAL ANSWER (Query 2) ---\n{answer2}")