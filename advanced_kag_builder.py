import networkx as nx
import faiss
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter # Hoặc text splitter tự viết

from config import settings, prompt_manager
import llm_utils 

class AdvancedKAGBuilder:
    def __init__(self, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.graph = nx.Graph()
        self.doc_store = {}  # chunk_id -> text_chunk
        self.faiss_id_to_chunk_id = {}

    def _extract_entities_advanced(self, text_chunk, source_id_for_prompt=""):
        prompt = f"""
        Nhiệm vụ: Từ Text Chunk được cung cấp, hãy trích xuất tất cả các thực thể có tên (named entities). Với mỗi thực thể, hãy chỉ định rõ:
        1.  `text`: Nội dung văn bản của thực thể.
        2.  `type`: Loại của thực thể (ví dụ: PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT_NAME, EVENT).
        3.  `start_char`: Vị trí ký tự bắt đầu của thực thể trong Text Chunk (tính từ 0).
        4.  `end_char`: Vị trí ký tự kết thúc của thực thể trong Text Chunk (không bao gồm ký tự ở vị trí này).

        Context Source ID (tùy chọn để tham khảo): {source_id_for_prompt}
        Text Chunk:
        ---
        {text_chunk}
        ---

        Các loại thực thể cần đặc biệt chú ý đối với văn bản quy chế và tài liệu giáo dục (ưu tiên sử dụng các loại này nếu phù hợp, ngoài ra có thể dùng các loại chung khác nếu cần thiết):
        - PROGRAM: Các chương trình giáo dục (ví dụ: "Chương trình đào tạo tiên tiến", "Chương trình cử nhân chuẩn Quốc tế").
        - COURSE: Tên môn học hoặc mã môn học (ví dụ: "Giải tích I", "INT1001").
        - REGULATION: Tên các quy định, quy chế, chính sách (ví dụ: "Quy chế đào tạo đại học", "Nội quy học đường").
        - DOCUMENT: Các định danh tài liệu (ví dụ: "Quyết định số 123/QĐ-ĐHQGHN", "Thông tư 08/2021/TT-BGDĐT").
        - REQUIREMENT: Các yêu cầu, chuẩn mực (ví dụ: "Chuẩn đầu ra tiếng Anh", "Điều kiện xét tốt nghiệp").
        - ACADEMIC_UNIT: Tên khoa, phòng, ban, viện, trung tâm hoặc đơn vị học thuật (ví dụ: "Khoa Công nghệ Thông tin", "Phòng Đào tạo").
        - ROLE: Chức danh hoặc vai trò trong trường đại học (ví dụ: "Hiệu trưởng", "Sinh viên", "Giảng viên").
        - CREDIT: Thông tin liên quan đến tín chỉ (ví dụ: "120 tín chỉ", "3 tín chỉ học phí").
        - EVALUATION: Phương pháp đánh giá, hệ thống điểm (ví dụ: "Thi cuối kỳ", "Điểm tổng kết hệ 4").
        - FEE: Các loại học phí, lệ phí (ví dụ: "học phí học kỳ 1", "lệ phí xét tuyển").
        - DEADLINE: Các mốc thời gian, hạn chót (ví dụ: "hạn nộp hồ sơ", "ngày thi").

        QUAN TRỌNG:
        - Kết quả PHẢI trả về dưới dạng một danh sách JSON (JSON list) các đối tượng. TUYỆT ĐỐI tuân thủ định dạng JSON.
        - Mỗi đối tượng trong danh sách JSON phải có đủ 4 keys: "text", "type", "start_char", và "end_char".
        - Nếu không tìm thấy thực thể nào, hãy trả về một danh sách JSON rỗng: [].

        Ví dụ về định dạng JSON đầu ra mong muốn:
        [
            {{"text": "Trường Đại học Công nghệ", "type": "ACADEMIC_UNIT", "start_char": 50, "end_char": 75}},
            {{"text": "Quy chế học vụ", "type": "REGULATION", "start_char": 102, "end_char": 118}},
            {{"text": "sinh viên", "type": "ROLE", "start_char": 15, "end_char": 23}}
        ]

        Entities (JSON list):
        """
        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=500, system_message=prompt_manager.sys_prompt_entity_extractor)
        try:
            entities = json.loads(response_str)
            if not isinstance(entities, list): # Đảm bảo là list
                 print(f"Warning: LLM entity extraction did not return a list for chunk from {source_id_for_prompt}. Got: {response_str}")
                 return []
            # TODO: Validate thêm cấu trúc của từng object entity
            valid_entities = []
            for entity in entities:
                # Kiểm tra xem entity có cấu trúc hợp lệ không
                if (isinstance(entity, dict) and 
                    "text" in entity and isinstance(entity["text"], str) and len(entity["text"].strip()) > 0 and
                    "type" in entity and isinstance(entity["type"], str) and
                    "start_char" in entity and isinstance(entity["start_char"], int) and
                    "end_char" in entity and isinstance(entity["end_char"], int) and
                    entity["start_char"] < entity["end_char"]):
                    
                    # Chuẩn hóa entity trước khi thêm vào danh sách
                    entity["text"] = entity["text"].strip()
                    entity["type"] = entity["type"].upper()
                    valid_entities.append(entity)
                else:
                    print(f"Warning: Invalid entity structure: {entity}")
            
            return valid_entities
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (entities) for chunk from {source_id_for_prompt}: {response_str}")
            return [] # Trả về list rỗng nếu lỗi

    def _extract_relations(self, text_chunk, extracted_entities, source_id_for_prompt=""):
        """
        Trích xuất mối quan hệ giữa các thực thể bằng LLM.
        """
        if not extracted_entities:
            return []

        # Chuẩn bị danh sách thực thể cho prompt
        entities_for_prompt = [{"text": ent["text"], "type": ent["type"]} for ent in extracted_entities]

        prompt = f"""
        Nhiệm vụ: Từ Text Chunk và danh sách Extracted Entities (các thực thể đã được trích xuất) được cung cấp, hãy xác định các mối quan hệ có ý nghĩa giữa các thực thể đó.

        Context Source ID (tùy chọn để tham khảo): {source_id_for_prompt}
        Text Chunk:
        ---
        {text_chunk}
        ---
        Extracted Entities (Thực thể đã trích xuất, kèm theo loại):
        {json.dumps(entities_for_prompt, indent=2, ensure_ascii=False)} 
        ---

        Hướng dẫn xác định mối quan hệ:
        1.  Mỗi mối quan hệ phải được biểu diễn dưới dạng một bộ ba (triple) JSON: 
            `["subject_entity_text", "relation_description", "object_entity_text"]`
            Trong đó:
            -   `subject_entity_text`: PHẢI là nội dung văn bản (text) của một thực thể từ danh sách Extracted Entities đóng vai trò là chủ thể.
            -   `relation_description`: PHẢI là một cụm động từ ngắn gọn bằng Tiếng Việt mô tả chính xác mối quan hệ. Nên chuẩn hóa bằng cách viết thường và dùng dấu gạch dưới thay cho khoảng trắng (ví dụ: "là_điều_kiện_tiên_quyết_của", "được_quy_định_trong").
            -   `object_entity_text`: PHẢI là nội dung văn bản (text) của một thực thể từ danh sách Extracted Entities đóng vai trò là đối tượng.

        2.  Chỉ trích xuất các mối quan hệ được nêu rõ ràng hoặc có thể suy luận trực tiếp và mạnh mẽ từ Text Chunk. KHÔNG suy diễn hoặc thêm thông tin không có trong văn bản.
        3.  Ưu tiên sử dụng các loại mô tả quan hệ (relation_description) sau nếu phù hợp với ngữ cảnh văn bản quy chế và tài liệu giáo dục. Bạn cũng có thể sử dụng các mô tả khác nếu chúng diễn tả đúng mối quan hệ hơn:
            - "thuộc_về": Thực thể là một phần của thực thể khác (ví dụ: một môn học thuộc về một chương trình đào tạo).
            - "yêu_cầu": Thực thể này yêu cầu thực thể kia (ví dụ: một môn học yêu cầu môn tiên quyết).
            - "được_quản_lý_bởi": Thực thể được quản lý hoặc điều hành bởi thực thể khác.
            - "được_định_nghĩa_trong": Thực thể được định nghĩa hoặc mô tả trong một tài liệu/quy định.
            - "bao_gồm_thành_phần": Thực thể này có thực thể kia là một thành phần.
            - "được_đánh_giá_bằng": Thực thể được đánh giá bằng phương pháp/thực thể khác.
            - "tương_đương_với": Thực thể này tương đương với thực thể kia.
            - "kế_thừa_bởi": Thực thể này được kế thừa hoặc thay thế bởi thực thể kia.
            - "được_tạo_bởi": Thực thể được tạo ra bởi thực thể khác.
            - "được_sửa_đổi_bởi": Thực thể được sửa đổi bởi thực thể khác.
            - "áp_dụng_cho": Quy định/chính sách này áp dụng cho đối tượng/thực thể kia.
            - "có_liên_quan_đến": Hai thực thể có mối liên hệ chung chung.

        QUAN TRỌNG:
        - Kết quả PHẢI trả về dưới dạng một danh sách JSON (JSON list) các bộ ba. TUYỆT ĐỐI tuân thủ định dạng JSON.
        - Nếu không tìm thấy mối quan hệ rõ ràng nào, hãy trả về một danh sách JSON rỗng: [].

        Ví dụ về định dạng JSON đầu ra mong muốn:
        [
            ["Chương trình Tiên tiến ngành CNTT", "yêu cầu", "Chứng chỉ IELTS 6.0"],
            ["Quy chế tuyển sinh", "được_định_nghĩa_trong", "Thông tư số 05/2023/TT-BGDĐT"],
            ["Sinh viên K68", "thuộc_về", "Khoa Luật"]
        ]

        Relationships (JSON list of triples):
        """
        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=500, system_message=prompt_manager.sys_prompt_relation_extractor)
        try:
            relations = json.loads(response_str)
            if not isinstance(relations, list):
                 print(f"Warning: LLM relation extraction did not return a list for chunk from {source_id_for_prompt}. Got: {response_str}")
                 return []
            # TODO:  Validate thêm cấu trúc của từng triple
            valid_relations = []
            for relation in relations:
                # Kiểm tra xem relation có cấu trúc hợp lệ không (là list có 3 phần tử string)
                if (isinstance(relation, list) and len(relation) == 3 and
                    all(isinstance(item, str) for item in relation) and
                    all(len(item.strip()) > 0 for item in relation)):
                    
                    # Chuẩn hóa các phần tử trong relation trước khi thêm vào danh sách
                    subject = relation[0].strip()
                    relation_type = relation[1].strip().lower().replace(' ', '_')
                    object_text = relation[2].strip()
                    
                    valid_relations.append([subject, relation_type, object_text])
                else:
                    print(f"Warning: Invalid relation structure: {relation}")
            
            return valid_relations
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (relations) for chunk from {source_id_for_prompt}: {response_str}")
            return []

    def build_from_processed_data(self, processed_data_list):
        """
        processed_data_list: list of dicts, e.g., [{"source_id": ..., "text_content": ...}, ...]
        """
        all_chunks_for_embedding = []
        chunk_id_counter = 0

        if not processed_data_list:
            print("No processed data to build KG from.")
            return

        for item in processed_data_list:
            text_content = item.get("text_content", "")
            source_id = item.get("source_id", f"unknown_source_{chunk_id_counter}")
            original_url = item.get("original_url", "") # Lấy thêm thông tin nếu có

            if not text_content:
                continue

            # 1. Chia nhỏ văn bản thành các chunk
            chunks = self.text_splitter.split_text(text_content)
            
            for i, chunk_text in enumerate(chunks):
                current_chunk_id = f"{source_id}_chunk_{i}"
                self.doc_store[current_chunk_id] = chunk_text # Lưu text gốc của chunk
                all_chunks_for_embedding.append({"id": current_chunk_id, "text": chunk_text})

                # 2. Thêm node chunk vào đồ thị
                self.graph.add_node(current_chunk_id, type="chunk", 
                                    source_document_id=source_id, 
                                    original_url=original_url, # Lưu thêm URL gốc
                                    text_preview=chunk_text[:150]+"...") 

                # 3. Trích xuất thực thể nâng cao
                # print(f"Extracting entities from: {current_chunk_id}")
                extracted_entities = self._extract_entities_advanced(chunk_text, source_id_for_prompt=current_chunk_id)
                
                entity_nodes_in_chunk = {} # Lưu trữ node_id của thực thể đã chuẩn hóa trong chunk này

                for ent_data in extracted_entities:
                    ent_text = ent_data.get("text")
                    ent_type = ent_data.get("type", "UNKNOWN_ENTITY_TYPE")
                    # TODO:  Implement logic chuẩn hóa tên thực thể (quan trọng!)
                    # Ví dụ: viết thường, loại bỏ dấu câu không cần thiết, sử dụng stemming/lemmatization
                    normalized_ent_id = self._normalize_entity_text(ent_text)

                    if not self.graph.has_node(normalized_ent_id):
                        self.graph.add_node(normalized_ent_id, type="entity", entity_type=ent_type,
                                            # Lưu tên gốc nếu cần
                                            original_text_forms=set([ent_text])) 
                    else:
                        # Cập nhật danh sách tên gốc nếu đã tồn tại
                        if 'original_text_forms' in self.graph.nodes[normalized_ent_id]:
                            self.graph.nodes[normalized_ent_id]['original_text_forms'].add(ent_text)
                        else:
                             self.graph.nodes[normalized_ent_id]['original_text_forms'] = set([ent_text])
                        # Cập nhật entity_type nếu type mới cụ thể hơn (cần logic phức tạp hơn)
                        if ent_type != "UNKNOWN_ENTITY_TYPE" and self.graph.nodes[normalized_ent_id].get('entity_type') == "UNKNOWN_ENTITY_TYPE":
                            self.graph.nodes[normalized_ent_id]['entity_type'] = ent_type
                        
                    # Thêm cạnh từ chunk đến thực thể (ví dụ: "mentions_entity")
                    self.graph.add_edge(current_chunk_id, normalized_ent_id, type="mentions_entity",
                                        entity_text_in_chunk=ent_text,
                                        start_char=ent_data.get("start_char"), # Lưu vị trí nếu cần
                                        end_char=ent_data.get("end_char")
                                        )
                    entity_nodes_in_chunk[ent_text] = normalized_ent_id


                # 4. Trích xuất mối quan hệ
                # print(f"Extracting relations from: {current_chunk_id}")
                extracted_relations = self._extract_relations(chunk_text, extracted_entities, source_id_for_prompt=current_chunk_id)
                for subj_text, rel_type, obj_text in extracted_relations:
                    # TODO:  Map subj_text và obj_text về normalized_ent_id đã tạo ở trên
                    #       Điều này quan trọng để kết nối đúng các node trong KG.
                    #       Cần một cơ chế mapping hoặc tìm lại normalized_id từ text gốc.
                    
                    # Tìm kiếm entity node đã chuẩn hóa
                    subj_node_id = self._find_entity_node(subj_text, entity_nodes_in_chunk)
                    obj_node_id = self._find_entity_node(obj_text, entity_nodes_in_chunk)
                    
                    # Nếu không tìm thấy, tạo node mới
                    if not subj_node_id:
                        subj_node_id = self._normalize_entity_text(subj_text)
                        if not self.graph.has_node(subj_node_id): 
                            self.graph.add_node(subj_node_id, type="entity", entity_type="RELATION_IMPLIED", 
                                             original_text_forms=set([subj_text]))
                    
                    if not obj_node_id:
                        obj_node_id = self._normalize_entity_text(obj_text)
                        if not self.graph.has_node(obj_node_id): 
                            self.graph.add_node(obj_node_id, type="entity", entity_type="RELATION_IMPLIED",
                                             original_text_forms=set([obj_text]))

                    if subj_node_id and obj_node_id and subj_node_id != obj_node_id:
                         # Thêm cạnh mối quan hệ giữa các node thực thể
                        self.graph.add_edge(subj_node_id, obj_node_id, type="kg_relation", 
                                            relation_label=rel_type, 
                                            source_chunk_id=current_chunk_id) # Liên kết lại với chunk chứa bằng chứng
                    else:
                        print(f"Warning: Could not map entities for relation: [{subj_text}, {rel_type}, {obj_text}] in {current_chunk_id}")


                chunk_id_counter += 1
                if chunk_id_counter % 5 == 0: # Log thường xuyên hơn cho dev
                    print(f"KAG BUILDER: Processed {chunk_id_counter} chunks...")
        
        # --- Xây dựng Vector Index ---
        if not all_chunks_for_embedding:
            print("KAG BUILDER: No chunks to build vector index from.")
            return

        chunk_texts_for_embedding = [chunk['text'] for chunk in all_chunks_for_embedding]
        chunk_ids_for_index = [chunk['id'] for chunk in all_chunks_for_embedding]

        print("KAG BUILDER: Generating embeddings for chunks...")
        embeddings = llm_utils.get_embeddings(chunk_texts_for_embedding).cpu().numpy()

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        # Sử dụng IndexIDMap2 để có thể xóa ID nếu cần sau này (optional)
        # Hoặc giữ IndexIDMap nếu không cần xóa
        self.faiss_index_map = faiss.IndexIDMap(faiss_index) 

        numeric_ids_for_faiss = list(range(len(chunk_ids_for_index)))
        self.faiss_id_to_chunk_id = {i: chunk_id for i, chunk_id in enumerate(chunk_ids_for_index)}
        
        self.faiss_index_map.add_with_ids(embeddings, numeric_ids_for_faiss)
        print(f"KAG BUILDER: FAISS index built with {self.faiss_index_map.ntotal} vectors.")

        # --- Lưu trữ Artifacts ---
        faiss.write_index(self.faiss_index_map, settings.FAISS_INDEX_PATH)
        nx.write_gml(self.graph, settings.GRAPH_PATH) # Lưu đồ thị KG
        
        # Lưu doc_store (text của chunk) và mapping ID FAISS
        with open(settings.DOC_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"doc_store": self.doc_store, 
                       "faiss_id_map": self.faiss_id_to_chunk_id}, 
                      f, ensure_ascii=False, indent=4)
        print("KAG BUILDER: All artifacts (FAISS, KG, DocStore) saved.")

    def _normalize_entity_text(self, text):
        """
        Chuẩn hóa text của thực thể để sử dụng làm node ID.
        """
        import re
        # Bước 1: Chuyển sang chữ thường và loại bỏ khoảng trắng thừa
        normalized = text.lower().strip()
        
        # Bước 2: Loại bỏ dấu câu không cần thiết ở đầu và cuối
        normalized = re.sub(r'^[\s\.,;:\'\"!?\(\)\[\]\{\}]+|[\s\.,;:\'\"!?\(\)\[\]\{\}]+$', '', normalized)
        
        # Bước 3: Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Bước 4: Loại bỏ các từ chức năng nếu cần (có thể bỏ qua nếu muốn giữ nguyên nghĩa)
        # stop_words = {"và", "hoặc", "của", "là", "các", "những"}
        # words = normalized.split()
        # normalized = ' '.join([w for w in words if w not in stop_words])
        
        return normalized

    def _find_entity_node(self, entity_text, entity_nodes_in_chunk):
        """
        Tìm node ID cho một thực thể dựa trên text của nó.
        Đầu tiên tìm trong danh sách các thực thể của chunk hiện tại.
        Nếu không tìm thấy, thử chuẩn hóa và tìm kiếm trong toàn bộ đồ thị.
        """
        # Tìm kiếm chính xác trong danh sách entities của chunk
        if entity_text in entity_nodes_in_chunk:
            return entity_nodes_in_chunk[entity_text]
        
        # Nếu không tìm thấy, thử chuẩn hóa và so sánh
        normalized_text = self._normalize_entity_text(entity_text)
        
        # Tìm trong toàn bộ đồ thị xem có node entity nào phù hợp không
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'entity':
                # Kiểm tra các dạng text gốc
                original_forms = attrs.get('original_text_forms', set())
                if any(self._normalize_entity_text(form) == normalized_text for form in original_forms):
                    return node_id
                
                # Kiểm tra normalized node_id
                if self._normalize_entity_text(node_id) == normalized_text:
                    return node_id
        
        # Nếu không tìm thấy, trả về None
        return None

if __name__ == "__main__":
    # TODO:  Chạy file này sau khi Người 1 đã tạo ra "all_processed_texts.json".
    #       Cần có file llm_utils.py và config.py hoạt động.
    
    processed_data_path = os.path.join(settings.PROCESSED_DATA_DIR, "all_processed_texts.json")
    if not os.path.exists(processed_data_path):
        print(f"Processed data file not found: {processed_data_path}. Please run data_processor.py first.")
        # Tạo file dummy để test builder
        dummy_processed_data = [
            {"source_id": "dummy_course_ai", "text_content": "Introduction to Artificial Intelligence. John McCarthy coined the term AI. AI is a field of computer science. Machine Learning is a subfield of AI."},
            {"source_id": "dummy_course_python", "text_content": "Advanced Python programming. Python can be used for web development and data analysis. Guido van Rossum created Python."}
        ]
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_processed_data, f, indent=4)
        print(f"Created dummy processed data file for builder testing: {processed_data_path}")

    with open(processed_data_path, 'r', encoding='utf-8') as f:
        data_for_builder = json.load(f)

    print(f"Starting Advanced KAG Builder with {len(data_for_builder)} processed documents...")
    builder = AdvancedKAGBuilder()
    builder.build_from_processed_data(data_for_builder)
    print("Advanced KAG Builder process finished.")
