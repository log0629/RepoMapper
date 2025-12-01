from typing import Any

class RepoSummaryGenerator:
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def generate_summary(self, repo_map_content: str) -> str:
        """
        Generates a natural language summary of the repository based on the Repo Map.
        """
        if not repo_map_content:
            return ""
            
        prompt = f"Please summarize the following repository structure and content:\n\n{repo_map_content}"
        return self.llm_client.generate_text(prompt)
