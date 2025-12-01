from typing import Any

class OpenAILLMClient:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def generate_text(self, prompt: str) -> str:
        """
        Generates text using the OpenAI client.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
