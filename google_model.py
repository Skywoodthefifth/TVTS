import os
import re
from time import sleep
from typing import List, Dict

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from google.genai.errors import APIError
from google.genai.types import HarmCategory, HarmBlockThreshold

class GoogleModel():
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_keys_str = os.environ.get("GOOGLE_AI_API_KEY", "")
        self.api_keys = api_keys_str.split(",") if api_keys_str else []
        if not self.api_keys:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set or empty")
        print(f"Google API keys: {self.api_keys}")
        self.current_key_index = 0
        self.model_name = model_name
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))
        
        # Parse the bachkhoa_combined.txt file for RAG
        self.knowledge_base = self._parse_knowledge_base("bachkhoa_combined.txt")
        
        self.conversation_history: list[Content] = []  # Initialize conversation history

    def _parse_knowledge_base(self, file_path: str) -> List[Dict[str, str]]:
        """Parse the bachkhoa_combined.txt file to extract question-answer pairs."""
        knowledge_base = []
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Split by question blocks
        question_blocks = re.split(r'<start_question>', content)[1:]  # Skip the first empty element
        
        for block in question_blocks:
            if '<end_question>' not in block:
                continue
                
            # Extract question
            prompt_match = re.search(r'<start_prompt>(.*?)<end_prompt>', block, re.DOTALL)
            if not prompt_match:
                continue
            question = prompt_match.group(1).strip()
            
            # Extract answer
            answer_match = re.search(r'<start_answer>(.*?)<end_answer>', block, re.DOTALL)
            if not answer_match:
                continue
            answer = answer_match.group(1).strip()
            
            knowledge_base.append({
                "question": question,
                "answer": answer
            })
        
        print(f"Loaded {len(knowledge_base)} question-answer pairs from knowledge base")
        return knowledge_base
    
    def _search_relevant_content(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search for relevant content in the knowledge base using keyword matching."""
        query_lower = query.lower()
        query_words = re.findall(r'\w+', query_lower)
        
        scored_results = []
        
        for item in self.knowledge_base:
            question_lower = item["question"].lower()
            answer_lower = item["answer"].lower()
            
            # Calculate relevance score based on keyword matching
            score = 0
            
            # Check question relevance
            for word in query_words:
                if word in question_lower:
                    score += 2  # Higher weight for question matches
                if word in answer_lower:
                    score += 1
            
            if score > 0:
                scored_results.append((score, item))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_results[:max_results]]

    def generate_content(self, message: str) -> str:
        print(f"Starting chat completion with model: {self.model_name}")
        chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)  # Include conversation history
        
        # Search for relevant content using RAG
        relevant_content = self._search_relevant_content(message)
        
        # Build context from relevant content
        context = ""
        if relevant_content:
            context = "Thông tin liên quan từ cơ sở dữ liệu:\n\n"
            for i, item in enumerate(relevant_content, 1):
                context += f"{i}. Câu hỏi: {item['question']}\n"
                context += f"   Trả lời: {item['answer']}\n\n"
        
        # Create the enhanced prompt with RAG context
        enhanced_message = f"""Bạn là một phụ trách viên (tên là Chat Bot Tư Vấn Tuyển Sinh) hướng dẫn tư vấn tuyển sinh cho trường đại học Bách Khoa Đà Nẵng. Hãy trả lời các câu hỏi một cách đầy đủ các thông tin đã được cho và câu trả lời phải ngắn gọn, chuyên nghiệp, đầy khả năng thu hút các học sinh đang có nguyện vọng vào trường đại học Bách Khoa Đà Nẵng hoặc đang phân vân về nguyện vọng của mình.

{context}

Câu hỏi của học sinh: {message}

Hãy trả lời dựa trên thông tin liên quan ở trên (nếu có) trong vòng 100 từ."""

        # print("Question: ", message)
        print("Enhanced message: ", enhanced_message)

        try:
            chat_completion = chat.send_message(message=enhanced_message,
                                                config=types.GenerateContentConfig(
                                                    temperature=0.0,
                                                    safety_settings=[
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                    ]
                                                ))

            response = chat_completion.text.strip()
            # print("Answer: ", response)
            
            self.append_to_history(message, "user")  # Add original user message to history
            self.append_to_history(response, "model")  # Add model response to history

            return response
        except (APIError) as e:
            print(f"Gemini Model: {self.model_name} API error: {e}")
            # Switch to next API key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.client = genai.Client(
                api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))
            print(f"Switched to API key index {self.current_key_index}")
            sleep(3)
            return self.generate_content(message)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sleep(3)
            return self.generate_content(message)

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history: list[Content] = []
        print("Conversation history cleared.")

    def append_to_history(self, message: str, role: str):
        content = Content()
        part = Part()
        part.text = message
        content.parts = [part]
        content.role = role
        self.conversation_history.append(content)  # Add user message to history
        # print(f"Appended to history:\nRole: {role}\nMessage: {message}")

    def __str__(self):
        return f"Gemini(model_name={self.model_name})"
