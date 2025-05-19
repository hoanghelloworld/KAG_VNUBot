import networkx as nx
import faiss
import json
import os
import re # Ensure re is imported
from langchain.text_splitter import RecursiveCharacterTextSplitter # Or custom text splitter

from config import settings, prompt_manager
import llm_utils 
from tqdm import tqdm

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
        
        if settings.CONTINUE_BUILDING_KAG:
            print("KAG BUILDER: Continue building KAG from existed artifacts...")
            self.load_existed_artifacts()
            print(f"KAG BUILDER: {len(self.doc_store)} chunks loaded.")
            print("KAG BUILDER: All artifacts loaded.")
        
    def load_existed_artifacts(self):
        print("SOLVER: Loading FAISS index...")
        self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
        
        print("SOLVER: Loading Knowledge Graph...")
        self.graph = nx.read_gml(settings.GRAPH_PATH)
        
        # Convert 'original_text_forms' back to set after loading
        for node_id, attrs in self.graph.nodes(data=True):
            if 'original_text_forms' in attrs and isinstance(attrs['original_text_forms'], str):
                attrs['original_text_forms'] = set(attrs['original_text_forms'].split('|'))
            # Ensure 'original_text_forms' exists as a set even if it was empty or not present in GML for some reason
            # or if it was an empty string from GML, which split('|') would result in {' '}
            elif 'original_text_forms' in attrs and attrs['original_text_forms'] == {''}: # handles case of empty string becoming {''}
                 attrs['original_text_forms'] = set()
            # If the node is an entity type but somehow misses 'original_text_forms', initialize it.
            # This is more of a safeguard.
            elif attrs.get('type') == 'entity' and 'original_text_forms' not in attrs:
                attrs['original_text_forms'] = set()

        print("SOLVER: Loading Doc Store and FAISS ID Map...")
        with open(settings.DOC_STORE_PATH, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            self.doc_store = saved_data['doc_store']
            self.faiss_id_to_chunk_id = {int(k): v for k, v in saved_data['faiss_id_map'].items()}
        print("SOLVER: All artifacts loaded.")
        
    

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
        - Toàn bộ danh sách JSON các đối tượng PHẢI được đặt bên trong một cặp thẻ <final_json_answer> và </final_json_answer>. 
          Ví dụ: <final_json_answer>[{{"text": "Thực thể A", "type": "PERSON", "start_char": 0, "end_char": 10}}]</final_json_answer>.
        - KHÔNG có bất kỳ văn bản nào khác bên ngoài cặp thẻ <final_json_answer></final_json_answer> hoặc bên trong cặp thẻ này ngoài chính danh sách JSON.
        - Nếu không tìm thấy thực thể nào, hãy trả về một danh sách JSON rỗng bên trong thẻ: <final_json_answer>[]</final_json_answer>.

        Ví dụ về định dạng JSON đầu ra mong muốn (NẰM TRONG THẺ <final_json_answer>):
        <final_json_answer>
        [
            {{"text": "Trường Đại học Công nghệ", "type": "ACADEMIC_UNIT", "start_char": 50, "end_char": 75}},
            {{"text": "Quy chế học vụ", "type": "REGULATION", "start_char": 102, "end_char": 118}},
            {{"text": "sinh viên", "type": "ROLE", "start_char": 15, "end_char": 23}}
        ]
        </final_json_answer>

        Entities (JSON list of objects inside <final_json_answer> tags):
        """
        # Giả sử config.prompt_manager.sys_prompt_entity_extractor tồn tại
        system_message_for_entity = getattr(prompt_manager, 'sys_prompt_entity_extractor', "Bạn là một trợ lý AI chuyên trích xuất thực thể dưới dạng JSON theo yêu cầu.")

        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=3000, system_message=system_message_for_entity) # Tăng max_new_tokens nếu cần
        
        json_to_parse = ""
        try:
            match = re.search(r"<final_json_answer>(.*?)</final_json_answer>", response_str, re.DOTALL)
            if match:
                json_to_parse = match.group(1).strip()
            else:
                print(f"Error: <final_json_answer> tag not found in LLM response for entity extraction from chunk {source_id_for_prompt}. Response: {response_str}")
                return []

            if not json_to_parse:
                print(f"Warning: <final_json_answer> tag was empty in LLM response for entity extraction from chunk {source_id_for_prompt}. Response: {response_str}")
                return []
            
            entities = json.loads(json_to_parse)
            if not isinstance(entities, list):
                 print(f"Warning: LLM entity extraction (inside tag) did not return a list for chunk from {source_id_for_prompt}. Got: {json_to_parse}")
                 return []

            valid_entities = []
            for entity in entities:
                if (isinstance(entity, dict) and 
                    "text" in entity and isinstance(entity["text"], str) and len(entity["text"].strip()) > 0 and
                    "type" in entity and isinstance(entity["type"], str) and
                    "start_char" in entity and isinstance(entity["start_char"], int) and
                    "end_char" in entity and isinstance(entity["end_char"], int) and
                    entity["start_char"] < entity["end_char"]):
                    
                    entity["text"] = entity["text"].strip()
                    entity["type"] = entity["type"].upper()
                    valid_entities.append(entity)
                else:
                    print(f"Warning: Invalid entity structure: {entity} in chunk {source_id_for_prompt}")
            
            return valid_entities
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (entities) for chunk {source_id_for_prompt}. Content parsed: '{json_to_parse}'. Original response: {response_str}")
            return []

    def _extract_relations(self, text_chunk, extracted_entities, source_id_for_prompt=""):
        """
        Extract relationships between entities using LLM.
        """
        if not extracted_entities:
            return []

        # Prepare entity list for the prompt
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
            - "tương_đương_với": Thực thể này tương đương với thực thể khác.
            - "kế_thừa_bởi": Thực thể này được kế thừa hoặc thay thế bởi thực thể khác.
            - "được_tạo_bởi": Thực thể được tạo ra bởi thực thể khác.
            - "được_sửa_đổi_bởi": Thực thể được sửa đổi bởi thực thể khác.
            - "áp_dụng_cho": Quy định/chính sách này áp dụng cho đối tượng/thực thể kia.
            - "có_liên_quan_đến": Hai thực thể có mối liên hệ chung chung.

        QUAN TRỌNG:
        - Kết quả PHẢI trả về dưới dạng một danh sách JSON (JSON list) các bộ ba. TUYỆT ĐỐI tuân thủ định dạng JSON.
        - Toàn bộ danh sách JSON các bộ ba PHẢI được đặt bên trong một cặp thẻ <final_json_answer> và </final_json_answer>. 
          Ví dụ: <final_json_answer>[["Thực thể A", "quan_hệ_1", "Thực thể B"]]</final_json_answer>.
        - KHÔNG có bất kỳ văn bản nào khác bên ngoài cặp thẻ <final_json_answer></final_json_answer> hoặc bên trong cặp thẻ này ngoài chính danh sách JSON.
        - Nếu không tìm thấy mối quan hệ rõ ràng nào, hãy trả về một danh sách JSON rỗng bên trong thẻ: <final_json_answer>[]</final_json_answer>.

        Ví dụ về định dạng JSON đầu ra mong muốn (NẰM TRONG THẺ <final_json_answer>):
        <final_json_answer>
        [
            ["Chương trình Tiên tiến ngành CNTT", "yêu_cầu", "Chứng chỉ IELTS 6.0"],
            ["Quy chế tuyển sinh", "được_định_nghĩa_trong", "Thông tư số 05/2023/TT-BGDĐT"],
            ["Sinh viên K68", "thuộc_về", "Khoa Luật"]
        ]
        </final_json_answer>

        Relationships (JSON list of triples inside <final_json_answer> tags):
        """
        system_message_for_relation = getattr(prompt_manager, 'sys_prompt_relation_extractor', "Bạn là một trợ lý AI chuyên trích xuất thông tin quan hệ dưới dạng JSON theo yêu cầu.")
        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=3000, system_message=system_message_for_relation)
        
        json_to_parse = ""
        try:
            match = re.search(r"<final_json_answer>(.*?)</final_json_answer>", response_str, re.DOTALL)
            if match:
                json_to_parse = match.group(1).strip()
            else:
                print(f"Error: <final_json_answer> tag not found in LLM response for chunk {source_id_for_prompt}. Response: {response_str}")
                return []

            if not json_to_parse: 
                print(f"Warning: <final_json_answer> tag was empty in LLM response for chunk {source_id_for_prompt}. Response: {response_str}")
                return []

            relations = json.loads(json_to_parse)
            if not isinstance(relations, list):
                 print(f"Warning: LLM relation extraction (inside tag) did not return a list for chunk from {source_id_for_prompt}. Got: {json_to_parse}")
                 return []
            
            valid_relations = []
            for relation in relations:
                if (isinstance(relation, list) and len(relation) == 3 and
                    all(isinstance(item, str) for item in relation) and
                    all(len(item.strip()) > 0 for item in relation)):
                    
                    subject = relation[0].strip()
                    relation_type = relation[1].strip().lower().replace(' ', '_')
                    object_text = relation[2].strip()
                    
                    valid_relations.append([subject, relation_type, object_text])
                else:
                    print(f"Warning: Invalid relation structure: {relation} in chunk {source_id_for_prompt}")
            
            return valid_relations
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (relations) for chunk {source_id_for_prompt}. Content parsed: '{json_to_parse}'. Original response: {response_str}")
            return []

    def build_from_processed_data(self, processed_data_list):
        """
        processed_data_list: list of dicts, e.g., [{"source_id": ..., "text_content": ...}, ...]
        """
        all_chunks_for_embedding = []
        chunk_id_counter = 0 # This counter is for the print log below, not for unknown source_id

        if not processed_data_list:
            print("No processed data to build KG from.")
            return

        for index, item in enumerate(processed_data_list):
            text_content = item.get("text_content", "")
            # Use a unique fallback for source_id if it's missing, based on the item's index
            source_id = item.get("source_id", f"unknown_source_doc_{index}") 
            original_url = item.get("original_url", "") # Get additional information if available

            if not text_content:
                continue

            # 1. Split text into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            pbar = tqdm(total=len(chunks), desc=f"KAG Builder: documents {index}/{len(processed_data_list)}")
            for i, chunk_text in enumerate(chunks):
                current_chunk_id = f"{source_id}_chunk_{i}"
                
                # continue building KAG from existed artifacts
                if current_chunk_id in self.doc_store:
                    pbar.update(1)
                    continue
                
                self.doc_store[current_chunk_id] = chunk_text # Store original text of the chunk
                all_chunks_for_embedding.append({"id": current_chunk_id, "text": chunk_text})

                # 2. Add chunk node to the graph
                self.graph.add_node(current_chunk_id, type="chunk", 
                                    source_document_id=source_id, 
                                    original_url=original_url, # Store original URL as well
                                    text_preview=chunk_text[:150]+"...") 

                # 3. Advanced entity extraction
                extracted_entities = self._extract_entities_advanced(chunk_text, source_id_for_prompt=current_chunk_id)
                
                entity_nodes_in_chunk = {} # Store normalized entity_id within this chunk

                for ent_data in extracted_entities:
                    ent_text = ent_data.get("text")
                    ent_type = ent_data.get("type", "UNKNOWN_ENTITY_TYPE")
                    normalized_ent_id = self._normalize_entity_text(ent_text)

                    if not self.graph.has_node(normalized_ent_id):
                        self.graph.add_node(normalized_ent_id, type="entity", entity_type=ent_type,
                                            # Store original forms if needed
                                            original_text_forms=set([ent_text])) 
                    else:
                        # Update list of original forms if it already exists
                        if 'original_text_forms' in self.graph.nodes[normalized_ent_id]:
                            self.graph.nodes[normalized_ent_id]['original_text_forms'].add(ent_text)
                        else:
                             self.graph.nodes[normalized_ent_id]['original_text_forms'] = set([ent_text])
                        # Update entity_type if the new type is more specific (more complex logic might be needed)
                        if ent_type != "UNKNOWN_ENTITY_TYPE" and self.graph.nodes[normalized_ent_id].get('entity_type') == "UNKNOWN_ENTITY_TYPE":
                            self.graph.nodes[normalized_ent_id]['entity_type'] = ent_type
                        
                    # Add edge from chunk to entity (e.g., "mentions_entity")
                    self.graph.add_edge(current_chunk_id, normalized_ent_id, type="mentions_entity",
                                        entity_text_in_chunk=ent_text,
                                        start_char=ent_data.get("start_char"), # Store position if needed
                                        end_char=ent_data.get("end_char")
                                        )
                    entity_nodes_in_chunk[ent_text] = normalized_ent_id


                # 4. Extract relationships
                # print(f"Extracting relations from: {current_chunk_id}")
                extracted_relations = self._extract_relations(chunk_text, extracted_entities, source_id_for_prompt=current_chunk_id)
                for subj_text, rel_type, obj_text in extracted_relations:
                    # Find normalized entity node
                    subj_node_id = self._find_entity_node(subj_text, entity_nodes_in_chunk)
                    obj_node_id = self._find_entity_node(obj_text, entity_nodes_in_chunk)
                    
                    # If not found, create a new node
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
                         # Add relationship edge between entity nodes
                        self.graph.add_edge(subj_node_id, obj_node_id, type="kg_relation", 
                                            relation_label=rel_type, 
                                            source_chunk_id=current_chunk_id) # Link back to the chunk containing the evidence
                    else:
                        print(f"Warning: Could not map entities for relation: [{subj_text}, {rel_type}, {obj_text}] in {current_chunk_id}")


                chunk_id_counter += 1
                # if chunk_id_counter % 5 == 0: # Log more frequently for development
                #     print(f"KAG BUILDER: Processed {chunk_id_counter} chunks...")
                pbar.update(1)
                
            #update vector index and save artifacts after each document
            self.build_vector_index_and_save_artifacts(all_chunks_for_embedding)
        
        
    def build_vector_index_and_save_artifacts(self, all_chunks_for_embedding):
        # ... (phần code xây dựng FAISS index giữ nguyên từ file của bạn) ...
        # --- Xây dựng Vector Index ---
        if not all_chunks_for_embedding:
            print("KAG BUILDER: No chunks to build vector index from.")
            # Quyết định xem có nên dừng ở đây hay vẫn lưu KG rỗng
            # Hiện tại, giả sử vẫn muốn lưu KG dù không có chunk nào cho vector index
            # Nếu bạn muốn dừng hẳn, thêm return ở đây

        if all_chunks_for_embedding: # Chỉ chạy nếu có chunk để tạo embedding
            chunk_texts_for_embedding = [chunk['text'] for chunk in all_chunks_for_embedding]
            chunk_ids_for_index = [chunk['id'] for chunk in all_chunks_for_embedding]

            print("KAG BUILDER: Generating embeddings for chunks...")
            embeddings = llm_utils.get_embeddings(chunk_texts_for_embedding).cpu().numpy()

            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index_map = faiss.IndexIDMap(faiss_index) 

            numeric_ids_for_faiss = list(range(len(chunk_ids_for_index)))
            self.faiss_id_to_chunk_id = {i: chunk_id for i, chunk_id in enumerate(chunk_ids_for_index)}
            
            self.faiss_index_map.add_with_ids(embeddings, numeric_ids_for_faiss)
            print(f"KAG BUILDER: FAISS index built with {self.faiss_index_map.ntotal} vectors.")
            faiss.write_index(self.faiss_index_map, settings.FAISS_INDEX_PATH)
        else:
            # Xử lý trường hợp không có chunk nào cho vector index
            # Có thể tạo một FAISS index rỗng hoặc không làm gì cả tùy theo logic của bạn
            # Hiện tại, nếu không có chunk, sẽ không có file FAISS index được tạo/ghi đè
            print("KAG BUILDER: No chunks provided for FAISS index, skipping FAISS index creation.")

        # --- Chuẩn bị và Lưu Đồ thị KG ---
        print("KAG BUILDER: Post-processing graph for GML compatibility...")
        processed_graph = self._post_process_graph_for_gml(self.graph)
        # processed_graph = self.graph
        
        print("KAG BUILDER: Saving Knowledge Graph to GML file...")
        nx.write_gml(processed_graph, settings.GRAPH_PATH) # Sử dụng config và đồ thị đã xử lý
        
        # Đảm bảo self.faiss_id_to_chunk_id được khởi tạo nếu không có embedding
        if not hasattr(self, 'faiss_id_to_chunk_id'):
            self.faiss_id_to_chunk_id = {}

        # --- Lưu Doc Store và FAISS ID Map ---
        print("KAG BUILDER: Saving DocStore and FAISS ID Map...")
        with open(settings.DOC_STORE_PATH, 'w', encoding='utf-8') as f: # Sử dụng config
            json.dump({"doc_store": self.doc_store, 
                       "faiss_id_map": self.faiss_id_to_chunk_id}, 
                      f, ensure_ascii=False, indent=4)
        print("KAG BUILDER: All artifacts (KG, DocStore) saved. FAISS index saved if chunks were provided.")

    def _post_process_graph_for_gml(self, graph_input):
        """
        Chuẩn hóa các thuộc tính của node và cạnh trong đồ thị để tương thích với định dạng GML.
        - Chuyển đổi 'original_text_forms' (set) thành chuỗi nối bằng '|'.
        - Chuyển đổi các thuộc tính là list, dict, set khác thành chuỗi JSON.
        - Chuyển đổi các thuộc tính có kiểu không chuẩn khác thành chuỗi.
        Trả về một bản sao của đồ thị đã được xử lý.
        """
        graph_to_save = graph_input.copy() # Làm việc trên bản sao

        # Xử lý thuộc tính của Nodes
        for node_id, attrs in graph_to_save.nodes(data=True):
            if 'original_text_forms' in attrs and isinstance(attrs['original_text_forms'], set):
                attrs['original_text_forms'] = "|".join(sorted(list(attrs['original_text_forms'])))
            
            for key, value in list(attrs.items()): # Duyệt qua bản sao của các items để có thể sửa dict
                if key == 'original_text_forms' and isinstance(value, str): # Bỏ qua nếu đã xử lý ở trên
                    continue
                if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
                    try:
                        attrs[key] = json.dumps(value, ensure_ascii=False)
                    except TypeError as e:
                        print(f"Warning: Could not JSON serialize attribute '{key}' for node '{node_id}'. Error: {e}. Converting to string.")
                        attrs[key] = str(value)
                elif not isinstance(value, (str, int, float, bool)) and value is not None:
                     print(f"Warning: Attribute '{key}' for node '{node_id}' has non-standard GML type {type(value)}. Converting to string.")
                     attrs[key] = str(value)

        # Xử lý thuộc tính của Edges
        for u, v, attrs in graph_to_save.edges(data=True):
            for key, value in list(attrs.items()): # Duyệt qua bản sao của các items
                if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
                    try:
                        attrs[key] = json.dumps(value, ensure_ascii=False)
                    except TypeError as e:
                        print(f"Warning: Could not JSON serialize attribute '{key}' for edge '({u}-{v})'. Error: {e}. Converting to string.")
                        attrs[key] = str(value)
                elif not isinstance(value, (str, int, float, bool)) and value is not None:
                     print(f"Warning: Attribute '{key}' for edge '({u}-{v})' has non-standard GML type {type(value)}. Converting to string.")
                     attrs[key] = str(value)
        
        return graph_to_save

    def _normalize_entity_text(self, text):
        """
        Normalize entity text to be used as a node ID.
        """
        # Step 1: Convert to lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Step 2: Remove unnecessary punctuation from start and end
        normalized = re.sub(r'^[\s\.,;:\'\"!?\(\)\[\]\{\}]+|[\s\.,;:\'\"!?\(\)\[\]\{\}]+$', '', normalized)
        
        # Step 3: Replace multiple consecutive spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Step 4: Remove function words if needed (can be skipped if meaning should be preserved)
        # stop_words = {"and", "or", "of", "is", "the", "a", "an"} # Example English stop words
        # words = normalized.split()
        # normalized = ' '.join([w for w in words if w.lower() not in stop_words])
        
        return normalized

    def _find_entity_node(self, entity_text, entity_nodes_in_chunk):
        """
        Find node ID for an entity based on its text.
        First, search in the current chunk's entity list.
        If not found, try normalizing and searching the entire graph.
        """
        # Exact search in the chunk's entity list
        if entity_text in entity_nodes_in_chunk:
            return entity_nodes_in_chunk[entity_text]
        
        # If not found, try normalizing and comparing
        normalized_text = self._normalize_entity_text(entity_text)
        
        # Search the entire graph for a matching entity node
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'entity':
                # Check original text forms
                original_forms = attrs.get('original_text_forms', set())
                if any(self._normalize_entity_text(form) == normalized_text for form in original_forms):
                    return node_id
                
                # Check normalized node_id
                if self._normalize_entity_text(node_id) == normalized_text: # Assuming node_id itself might be a non-normalized form
                    return node_id
        
        # If not found, return None
        return None

if __name__ == "__main__":
    # This script should be run after "all_processed_texts.json" has been generated.
    # Requires working llm_utils.py and config.py.
    
    processed_data_path = os.path.join(settings.PROCESSED_DATA_DIR, "all_processed_texts.json")
    if not os.path.exists(processed_data_path):
        print(f"Processed data file not found: {processed_data_path}. Please run the data processing script first.")
        # Create dummy file to test builder
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
