from google import genai
from google.genai import types
from config.config import Config
from typing import List

class GeminiLLM:
    """Gemini 2.5 Flash LLM for generating responses"""
    
    def __init__(self):
        """Initialize Gemini LLM client"""
        Config.validate()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = Config.LLM_MODEL
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate response using query and retrieved context
        
        Args:
            query: User's question
            context: List of relevant text chunks from vector search
            
        Returns:
            Generated response string
        """
        # Prepare context
        context_text = "\n\n".join([f"Context {i+1}: {text}" for i, text in enumerate(context)])
        
        # thêm điều kiện để trả lời khi nhận được ảnh rắn ví dụ nếu có ảnh rắn thì trả lời như vậy còn không thì trả lời như này 
        # Create prompt
        prompt = f"""Consider yourself a snake expert to give professional answers, answer users like an expert and not answer like you rely on this or that information to give results even though you have to get results from context to answer

Based on the following context information, please answer the question accurately and comprehensively.

Context Information: (But when answering, don't write that it is based on any context.)
{context_text}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question, please mention that.

Position yourself as a snake expert, give the user some more questions related to the current question so the user can build on that and then continue saying what question you want me to help you answer

With the question structure including the main content as follows, 3 to 5 questions can be randomly given to users for reference.
-Scientific name and common name
-Taxonomy
-Morphological characteristics
-Toxicology
-Predation behavior
-Behavior and ecology
-Geographic distribution and habitat
-Reproduction
-Conservation status
-Research value
-Human relevance
-Symptoms when bitten
-How to handle"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation with thinking disabled
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            final_response = response.candidates[0].content.parts[0].text
            return final_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def generate_simple_response(self, text: str) -> str:
        """
        Generate a simple response without context (for testing)
        
        Args:
            text: Input text
            
        Returns:
            Generated response string
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=text),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            final_response = response.candidates[0].content.parts[0].text
            return final_response
            
        except Exception as e:
            print(f"Error generating simple response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"