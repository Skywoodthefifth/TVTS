import os
from time import sleep

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from google.genai.errors import APIError
from google.genai.types import HarmCategory, HarmBlockThreshold

class GoogleModel():
    def __init__(self, model_name: str = "gemini-2.0-flash-thinking-exp"):
        api_keys_str = os.environ.get("GOOGLE_AI_API_KEY", "")
        self.api_keys = api_keys_str.split(",") if api_keys_str else []
        if not self.api_keys:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set or empty")
        print(f"Google API keys: {self.api_keys}")
        self.current_key_index = 0
        self.model_name = model_name
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))
        
        # Read the content from 'bachkhoa_combined.txt' file
        with open("bachkhoa_combined.txt", "r", encoding="utf-8") as file:
            self.system_message = file.read()
        
        self.conversation_history: list[Content] = []  # Initialize conversation history

    def generate_content(self, message: str) -> str:
        print(f"Starting chat completion with model: {self.model_name}")
        chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)  # Include conversation history
        message = f"Bạn là một phụ trách viên về hướng dẫn tư vấn tuyển sinh cho đại học Bách Khoa Đà Nẵng. Hãy trả lời các câu hỏi đầy đủ thông tin và ngắn gọn, như đang nói chuyện với người bạn của mình.\n\nCâu hỏi: {message}\n\nTrả lời trong vòng 100 từ."
        # print("Question: ", message)

        try:
            chat_completion = chat.send_message(message=message,
                                                config=types.GenerateContentConfig(
                                                    system_instruction=self.system_message,
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
            
            self.append_to_history(message, "user")  # Add user message to history
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
        print(f"Appended to history:\nRole: {role}\nMessage: {message}")

    def __str__(self):
        return f"Gemini(model_name={self.model_name})"
