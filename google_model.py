import os
import os.path
import re
from time import sleep
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
        
        # Initialize Vietnamese embedding model
        print("Loading Vietnamese embedding model...")
        try:
            self.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
            self.embeddings_available = True
            
            # Pre-compute embeddings for knowledge base
            print("Computing embeddings for knowledge base...")
            self._compute_knowledge_embeddings()
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            print("Falling back to keyword-only search")
            self.embedding_model = None
            self.embeddings_available = False
        
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
    
    def _load_embeddings_from_file(self, embeddings_file: str) -> bool:
        """Load pre-computed embeddings from file if available."""
        if not os.path.exists(embeddings_file):
            print(f"Embeddings file {embeddings_file} not found. Will compute new embeddings.")
            return False
        
        try:
            print(f"Loading embeddings from {embeddings_file}...")
            data = np.load(embeddings_file)
            self.question_embeddings = data['question_embeddings']
            self.answer_embeddings = data['answer_embeddings']
            print(f"Successfully loaded embeddings for {len(self.question_embeddings)} question-answer pairs")
            return True
        except Exception as e:
            print(f"Error loading embeddings from file: {e}")
            print("Will compute new embeddings.")
            return False
    
    def _save_embeddings_to_file(self, embeddings_file: str):
        """Save computed embeddings to file for future use."""
        try:
            print(f"Saving embeddings to {embeddings_file}...")
            np.savez_compressed(embeddings_file, 
                              question_embeddings=self.question_embeddings,
                              answer_embeddings=self.answer_embeddings)
            print(f"Successfully saved embeddings to {embeddings_file}")
        except Exception as e:
            print(f"Error saving embeddings to file: {e}")
    
    def _compute_knowledge_embeddings(self):
        """Pre-compute embeddings for all questions and answers in knowledge base."""
        if not self.knowledge_base:
            self.question_embeddings = np.array([])
            self.answer_embeddings = np.array([])
            return
        
        # Define embeddings file path
        embeddings_file = "knowledge_embeddings.npz"
        
        # Try to load existing embeddings first
        if self._load_embeddings_from_file(embeddings_file):
            return
        
        # If loading failed, compute new embeddings
        questions = [item['question'] for item in self.knowledge_base]
        answers = [item['answer'] for item in self.knowledge_base]
        
        # Compute embeddings
        print("Computing new embeddings...")
        self.question_embeddings = self.embedding_model.encode(questions, show_progress_bar=True)
        self.answer_embeddings = self.embedding_model.encode(answers, show_progress_bar=True)
        
        print(f"Computed embeddings for {len(questions)} question-answer pairs")
        
        # Save embeddings to file for future use
        self._save_embeddings_to_file(embeddings_file)
    
    def _keyword_search(self, query: str, max_results: int = 5) -> List[Tuple[float, Dict[str, str]]]:
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
        
        # Sort by score and return top results with scores
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:max_results]
    
    def _semantic_search(self, query: str, max_results: int = 5) -> List[Tuple[float, Dict[str, str]]]:
        """Search for relevant content using Vietnamese semantic embeddings."""
        if len(self.question_embeddings) == 0:
            return []
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate cosine similarity with questions
        question_similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        
        # Calculate cosine similarity with answers (lower weight)
        answer_similarities = cosine_similarity(query_embedding, self.answer_embeddings)[0]
        
        # Combine similarities (questions weighted higher)
        combined_similarities = question_similarities * 0.7 + answer_similarities * 0.3
        
        # Get top results
        top_indices = np.argsort(combined_similarities)[::-1][:max_results]
        
        results = []
        for idx in top_indices:
            if combined_similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append((float(combined_similarities[idx]), self.knowledge_base[idx]))
        
        return results
    
    def _hybrid_search(self, query: str, max_results: int = 5) -> List[Tuple[float, Dict[str, str]]]:
        """Combine keyword and semantic search results."""
        # Get keyword results
        keyword_results = self._keyword_search(query, max_results * 2)
        
        # Get semantic results if embeddings are available
        if self.embeddings_available:
            semantic_results = self._semantic_search(query, max_results * 2)
        else:
            print("Using keyword-only search (embeddings not available)")
            # Return keyword results only if no embeddings
            return keyword_results
        
        # Normalize scores and combine
        combined_scores = {}
        
        # Process keyword results (normalize to 0-1)
        if keyword_results:
            max_keyword_score = max(score for score, _ in keyword_results) if keyword_results else 1
            for score, item in keyword_results:
                key = (item['question'], item['answer'])
                normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                combined_scores[key] = combined_scores.get(key, 0) + normalized_score * 0.4
        
        # Process semantic results (already 0-1)
        for score, item in semantic_results:
            key = (item['question'], item['answer'])
            combined_scores[key] = combined_scores.get(key, 0) + score * 0.6
        
        # Convert back to list and sort
        final_results = []
        for (question, answer), score in combined_scores.items():
            final_results.append((score, {'question': question, 'answer': answer}))
        
        final_results.sort(key=lambda x: x[0], reverse=True)
        return final_results[:max_results]
    
    def _contextual_rerank(self, results: List[Tuple[float, Dict[str, str]]], 
                          query: str, conversation_history: List[Content]) -> List[Dict[str, str]]:
        """Re-rank results based on conversational context and relevance."""
        if not results:
            return []
        
        # Extract recent conversation context
        recent_context = ""
        if conversation_history:
            # Get last few exchanges for context
            recent_messages = conversation_history[-4:]  # Last 2 exchanges
            for content in recent_messages:
                if hasattr(content, 'parts') and content.parts:
                    recent_context += content.parts[0].text + " "
        
        reranked_results = []
        
        for score, item in results:
            # Calculate contextual relevance
            contextual_score = score
            
            # Boost score if answer mentions topics from recent conversation
            if recent_context:
                context_words = set(re.findall(r'\w+', recent_context.lower()))
                answer_words = set(re.findall(r'\w+', item['answer'].lower()))
                
                # Calculate overlap with recent context
                overlap = len(context_words.intersection(answer_words))
                if overlap > 0:
                    contextual_score *= (1 + overlap * 0.1)  # Boost by up to 50%
            
            # Boost score for shorter, more direct answers (better for chat)
            answer_length = len(item['answer'].split())
            if answer_length < 50:  # Prefer concise answers
                contextual_score *= 1.1
            elif answer_length > 200:  # Penalize very long answers
                contextual_score *= 0.9
            
            reranked_results.append((contextual_score, item))
        
        # Sort by new contextual score
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the items without scores
        return [item for score, item in reranked_results]
    
    def _search_relevant_content(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Main search method that uses hybrid search with contextual re-ranking."""
        # Use hybrid search
        hybrid_results = self._hybrid_search(query, max_results * 2)
        
        # Apply contextual re-ranking
        final_results = self._contextual_rerank(hybrid_results, query, self.conversation_history)
        
        return final_results[:max_results]

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
