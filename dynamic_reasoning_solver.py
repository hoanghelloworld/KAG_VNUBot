# dynamic_reasoning_solver.py
import faiss
import networkx as nx
import json
import os
import re

import config
import llm_utils # Sử dụng get_llm_response

class DynamicKAGSolver:
    def __init__(self, top_k_retrieval=config.TOP_K_RETRIEVAL, max_reasoning_steps=config.MAX_REASONING_STEPS):
        # --- Tải Artifacts ---
        print("SOLVER: Loading FAISS index...")
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
        
        print("SOLVER: Loading Knowledge Graph...")
        self.graph = nx.read_gml(config.GRAPH_PATH)
        
        print("SOLVER: Loading Doc Store and FAISS ID Map...")
        with open(config.DOC_STORE_PATH, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            self.doc_store = saved_data['doc_store']
            # faiss_id_map từ string (do JSON) sang int
            self.faiss_id_to_chunk_id = {int(k): v for k, v in saved_data['faiss_id_map'].items()}
        print("SOLVER: All artifacts loaded.")

        self.top_k = top_k_retrieval
        self.max_reasoning_steps = max_reasoning_steps

        # TODO: Tinh chỉnh prompt này dựa trên thử nghiệm.
        #       Có thể cần thêm ví dụ few-shot.
        #       Định nghĩa rõ ràng các công cụ và cách LLM nên sử dụng chúng.
        self.reason_act_prompt_template_str = """
        You are a reasoning agent trying to answer the Original Query.
        You have access to the following tools:
        1. `search_vector_db(search_query: str)`: Searches the vector database for relevant text chunks based on the `search_query`. Use this to find general information, definitions, or context.
        2. `query_kg(subject: str, relation: str, object: str, query_type: str)`: Queries the knowledge graph.
           - To find relations between two known entities: `query_kg("EntityA", "?", "EntityB", query_type="find_relation")`
           - To find entities related to one entity via a relation: `query_kg("EntityA", "relation_label", "?", query_type="find_object")` or `query_kg("?", "relation_label", "EntityB", query_type="find_subject")`
           - To get attributes of an entity (if KG stores them): `query_kg("EntityA", "?", "?", query_type="get_attributes")` (less common for simple graph)
           - To find chunks mentioning an entity: `query_kg("EntityA", "mentioned_in_chunk", "?", query_type="find_mentioning_chunks")`
        3. `finish(final_answer: str)`: If you have gathered enough information and can confidently answer the Original Query, use this tool to provide the final answer.

        Original Query: {original_query}

        Scratchpad (your thought process, actions taken, and observations):
        ---
        {scratchpad}
        ---

        Based on the Original Query and your Scratchpad, formulate your next Thought and the Action to take.
        Your Thought should briefly explain your reasoning for choosing the next action.
        Your Action MUST be one of the tools described above, with the correct arguments.

        Format your response STRICTLY as:
        Thought: [Your thought process]
        Action: [Call one of the tools, e.g., `search_vector_db("meaning of AI")` or `finish("The answer is...")`]
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
            if numeric_id != -1: # FAISS trả về -1 nếu không đủ k kết quả
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
        TODO: (Người 3) Implement logic truy vấn KG chi tiết hơn.
              - Xử lý các `query_type` khác nhau.
              - Chuẩn hóa input entity/relation (ví dụ: lowercase).
              - Tìm kiếm các node và cạnh trong `self.graph`.
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
                # TODO: (Người 3) Có thể tìm đường đi nếu không có cạnh trực tiếp
        
        # TODO: (Người 3) Implement thêm các query_type khác:
        # elif q_type == "find_object": ...
        # elif q_type == "find_subject": ...
        # elif q_type == "get_attributes": (nếu KAGBuilder lưu thuộc tính vào node) ...
        #     if subj != "?" and self.graph.has_node(subj):
        #         node_attrs = self.graph.nodes[subj]
        #         # Lọc ra các thuộc tính người dùng định nghĩa, không phải của networkx
        #         custom_attrs = {k:v for k,v in node_attrs.items() if k not in ['type', 'text_preview', 'original_text_forms']}
        #         if custom_attrs:
        #             results.append(f"Attributes for '{subj}': {custom_attrs}")
        #         else:
        #             results.append(f"No specific attributes found for '{subj}' in KG.")


        if not results:
            return f"Observation: No results found in KG for query_kg(\"{subject_str}\", \"{relation_str}\", \"{object_str}\", query_type=\"{q_type}\")."
        return f"Observation: KG query results for query_type='{q_type}':\n" + "\n".join([f"- {r}" for r in results])


    def _parse_llm_action_output(self, llm_output_str):
        """
        Phân tích output của LLM để lấy Thought và Action.
        TODO: (Người 3) Làm cho việc parsing này mạnh mẽ hơn, xử lý lỗi tốt hơn.
              Có thể yêu cầu LLM output JSON để dễ parse.
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
        finish_match = re.match(r"finish\((.*?)\)", action_call_str, re.IGNORECASE)
        search_db_match = re.match(r"search_vector_db\((.*?)\)", action_call_str, re.IGNORECASE)
        query_kg_match = re.match(r"query_kg\((.*?)\)", action_call_str, re.IGNORECASE)

        if finish_match:
            action_type = "finish"
            # Cẩn thận với quote trong input
            action_input = finish_match.group(1).strip().strip('"').strip("'")
        elif search_db_match:
            action_type = "search_vector_db"
            action_input = search_db_match.group(1).strip().strip('"').strip("'")
        elif query_kg_match:
            action_type = "query_kg"
            args_str = query_kg_match.group(1)
            # Tách args, cẩn thận với dấu phẩy trong chuỗi được quote
            # Ví dụ đơn giản: split by comma, rồi strip quote
            try:
                # Sử dụng regex để split by comma, nhưng bỏ qua comma trong quote
                # Hoặc cách đơn giản hơn là yêu cầu LLM không dùng comma trong string argument nếu được
                # args = [arg.strip().strip('"').strip("'") for arg in args_str.split(',')] 
                
                # Dùng json.loads nếu LLM có thể trả về một list các string
                # args_str_as_list = "[" + args_str + "]" # Biến nó thành một list JSON
                # parsed_args = json.loads(args_str_as_list.replace("'", "\"")) # Thay quote đơn bằng đôi cho JSON
                
                # Cách an toàn hơn: regex tìm các argument
                parsed_args = [a.strip().strip('"').strip("'") for a in re.findall(r'(?:[^,"]|"[^"]*")+', args_str)]

                if len(parsed_args) == 4: # subject, relation, object, query_type
                    action_input = tuple(parsed_args)
                else:
                    action_type = "error"
                    action_input = f"Invalid number of arguments for query_kg. Expected 4, got {len(parsed_args)}. Args: {args_str}"
            except Exception as e:
                action_type = "error"
                action_input = f"Error parsing query_kg arguments '{args_str}': {e}"
        else:
            action_type = "error"
            action_input = f"Unknown or malformed action: {action_call_str}"
            
        return thought, action_type, action_input


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
            llm_full_output = llm_utils.get_llm_response(current_prompt, max_new_tokens=300, system_message="You are a reasoning agent following instructions precisely.")
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
        # TODO: Tinh chỉnh prompt tổng hợp này.
        final_synthesis_prompt = f"""
        Original Query: {original_query}

        You have gone through a reasoning process. Here is your scratchpad containing your thoughts, actions, and observations:
        --- SCRATCHPAD START ---
        {scratchpad}
        --- SCRATCHPAD END ---

        Based on all the information gathered in your scratchpad, provide the best possible comprehensive final answer to the Original Query.
        If you cannot answer definitively, clearly state what information is still missing or what uncertainty remains.

        Final Answer:
        """
        final_answer = llm_utils.get_llm_response(final_synthesis_prompt, max_new_tokens=500, system_message="You are a summarization and final response generation expert.")
        return final_answer

if __name__ == "__main__":
    # TODO:  Chạy file này sau khi Người 2 đã tạo artifacts (FAISS, GML, DocStore).
    #       Đảm bảo llm_utils.py và config.py đã sẵn sàng.
    
    # Kiểm tra sự tồn tại của artifacts
    required_files = [config.FAISS_INDEX_PATH, config.GRAPH_PATH, config.DOC_STORE_PATH]
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

        query2 = "Tell me about John McCarthy's contributions related to AI and Lisp."
        print(f"\nSolving Query 2: {query2}")
        answer2 = solver.solve(query2)
        print(f"\n--- FINAL ANSWER (Query 2) ---\n{answer2}")
