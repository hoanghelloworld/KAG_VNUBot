# advanced_kag_builder.py
import networkx as nx
import faiss
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter # Hoặc text splitter tự viết

import config
import llm_utils # Sử dụng hàm get_llm_response, get_embeddings

class AdvancedKAGBuilder:
    def __init__(self, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        # TODO: (Người 2) Có thể cần tùy chỉnh text_splitter nếu cần
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
        """
        Trích xuất thực thể nâng cao bằng LLM.
        """
        # TODO: (Người 2) Thiết kế prompt NER chi tiết.
        #       Yêu cầu LLM trả về JSON list: [{"text": "...", "type": "...", "start_char": ..., "end_char": ...}]
        #       Xử lý output JSON từ LLM, bao gồm cả trường hợp lỗi parsing.
        prompt = f"""
        Context Source ID (optional): {source_id_for_prompt}
        Text Chunk:
        ---
        {text_chunk}
        ---
        Extract all named entities from the Text Chunk. For each entity, specify its text, type (e.g., PERSON, ORGANIZATION, LOCATION, COURSE_NAME, TOPIC, TECHNOLOGY, DATE), and its character start and end offset within the Text Chunk.
        Return the result as a VALID JSON list of objects. Each object should have "text", "type", "start_char", and "end_char" keys.
        If no entities are found, return an empty JSON list [].

        Example Output:
        [
            {{"text": "John McCarthy", "type": "PERSON", "start_char": 30, "end_char": 42}},
            {{"text": "Artificial Intelligence", "type": "TOPIC", "start_char": 0, "end_char": 20}}
        ]

        Entities (JSON list):
        """
        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=500, system_message="You are an expert entity extraction system.")
        try:
            entities = json.loads(response_str)
            if not isinstance(entities, list): # Đảm bảo là list
                 print(f"Warning: LLM entity extraction did not return a list for chunk from {source_id_for_prompt}. Got: {response_str}")
                 return []
            # TODO: (Người 2) Validate thêm cấu trúc của từng object entity
            return entities
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (entities) for chunk from {source_id_for_prompt}: {response_str}")
            return [] # Trả về list rỗng nếu lỗi

    def _extract_relations(self, text_chunk, extracted_entities, source_id_for_prompt=""):
        """
        Trích xuất mối quan hệ giữa các thực thể bằng LLM.
        """
        if not extracted_entities:
            return []
            
        # TODO: (Người 2) Thiết kế prompt RE chi tiết.
        #       Yêu cầu LLM trả về JSON list các triple: [subject_text, relation_type, object_text]
        #       Sử dụng extracted_entities (có type) để giúp LLM.
        #       Định nghĩa một tập các relation_type mong muốn hoặc để LLM tự do hơn.
        
        # Chuẩn bị danh sách thực thể cho prompt
        entities_for_prompt = [{"text": ent["text"], "type": ent["type"]} for ent in extracted_entities]

        prompt = f"""
        Context Source ID (optional): {source_id_for_prompt}
        Text Chunk:
        ---
        {text_chunk}
        ---
        Extracted Entities (with types):
        {json.dumps(entities_for_prompt, indent=2)}
        ---
        Identify meaningful relationships (triples) between the Extracted Entities based on the Text Chunk.
        A triple should be in the format: ["subject_entity_text", "relation_description", "object_entity_text"].
        The "relation_description" should be a concise verb phrase (e.g., "teaches", "is_prerequisite_for", "mentions_topic", "developed_by").
        Focus on relationships explicitly stated or strongly implied in the text.
        Return the result as a VALID JSON list of triples.
        If no clear relationships are found, return an empty JSON list [].

        Example Output:
        [
            ["Introduction to AI", "mentions_topic", "Machine Learning"],
            ["Advanced Python", "is_prerequisite_for", "Data Science Capstone"]
        ]

        Relationships (JSON list of triples):
        """
        response_str = llm_utils.get_llm_response(prompt, max_new_tokens=500, system_message="You are an expert relation extraction system.")
        try:
            relations = json.loads(response_str)
            if not isinstance(relations, list):
                 print(f"Warning: LLM relation extraction did not return a list for chunk from {source_id_for_prompt}. Got: {response_str}")
                 return []
            # TODO: (Người 2) Validate thêm cấu trúc của từng triple
            return relations
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
                    # TODO: (Người 2) Implement logic chuẩn hóa tên thực thể (quan trọng!)
                    # Ví dụ: viết thường, loại bỏ dấu câu không cần thiết, sử dụng stemming/lemmatization
                    normalized_ent_id = ent_text.lower().strip() # Chuẩn hóa đơn giản

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
                        # if ent_type != "UNKNOWN_ENTITY_TYPE" and self.graph.nodes[normalized_ent_id].get('entity_type') == "UNKNOWN_ENTITY_TYPE":
                        # self.graph.nodes[normalized_ent_id]['entity_type'] = ent_type
                        
                    # Thêm cạnh từ chunk đến thực thể (ví dụ: "mentions_entity")
                    self.graph.add_edge(current_chunk_id, normalized_ent_id, type="mentions_entity",
                                        entity_text_in_chunk=ent_text,
                                        # start_char=ent_data.get("start_char"), # Lưu vị trí nếu cần
                                        # end_char=ent_data.get("end_char")
                                        )
                    entity_nodes_in_chunk[ent_text] = normalized_ent_id


                # 4. Trích xuất mối quan hệ
                # print(f"Extracting relations from: {current_chunk_id}")
                extracted_relations = self._extract_relations(chunk_text, extracted_entities, source_id_for_prompt=current_chunk_id)
                for subj_text, rel_type, obj_text in extracted_relations:
                    # TODO: (Người 2) Map subj_text và obj_text về normalized_ent_id đã tạo ở trên
                    #       Điều này quan trọng để kết nối đúng các node trong KG.
                    #       Cần một cơ chế mapping hoặc tìm lại normalized_id từ text gốc.
                    subj_node_id = entity_nodes_in_chunk.get(subj_text) # Hoặc tìm lại qua chuẩn hóa
                    obj_node_id = entity_nodes_in_chunk.get(obj_text) # Hoặc tìm lại qua chuẩn hóa
                    
                    # Nếu không tìm thấy (ví dụ LLM tạo ra thực thể không có trong list ban đầu), có thể tạo node mới hoặc bỏ qua
                    if not subj_node_id:
                        subj_node_id = subj_text.lower().strip()
                        if not self.graph.has_node(subj_node_id): self.graph.add_node(subj_node_id, type="entity", entity_type="RELATION_IMPLIED")
                    if not obj_node_id:
                        obj_node_id = obj_text.lower().strip()
                        if not self.graph.has_node(obj_node_id): self.graph.add_node(obj_node_id, type="entity", entity_type="RELATION_IMPLIED")

                    if subj_node_id and obj_node_id and subj_node_id != obj_node_id :
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
        faiss.write_index(self.faiss_index_map, config.FAISS_INDEX_PATH)
        nx.write_gml(self.graph, config.GRAPH_PATH) # Lưu đồ thị KG
        
        # Lưu doc_store (text của chunk) và mapping ID FAISS
        with open(config.DOC_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"doc_store": self.doc_store, 
                       "faiss_id_map": self.faiss_id_to_chunk_id}, 
                      f, ensure_ascii=False, indent=4)
        print("KAG BUILDER: All artifacts (FAISS, KG, DocStore) saved.")

if __name__ == "__main__":
    # TODO: (Người 2) Chạy file này sau khi Người 1 đã tạo ra "all_processed_texts.json".
    #       Cần có file llm_utils.py và config.py hoạt động.
    
    processed_data_path = os.path.join(config.PROCESSED_DATA_DIR, "all_processed_texts.json")
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
