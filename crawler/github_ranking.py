import requests
import re
from typing import List

class GithubRankingCrawler:
    BASE_URL = "https://api.github.com/repos/EvanLi/Github-Ranking/contents/Top100"
    RAW_BASE_URL = "https://raw.githubusercontent.com/EvanLi/Github-Ranking/master/Top100"

    def fetch_file_list(self) -> List[str]:
        """Fetch list of markdown files from the ranking repository using GitHub API."""
        try:
            response = requests.get(self.BASE_URL)
            response.raise_for_status()
            
            # API returns a list of file objects
            # [{"name": "Top-100-stars.md", ...}, ...]
            files = []
            for item in response.json():
                if item.get("name", "").endswith(".md"):
                    files.append(item["name"])
            
            return files
        except Exception as e:
            print(f"Error fetching file list: {e}")
            return []

    def parse_markdown(self, content: str) -> List[dict]:
        """Parse markdown content to extract full table data."""
        data = []
        lines = content.split('\n')
        
        # Table parsing logic
        # | Ranking | Project Name | Stars | Forks | Language | Open Issues | Description | Last Commit |
        
        for line in lines:
            line = line.strip()
            if not line.startswith('|'):
                continue
            
            # Skip separator lines | --- | --- |
            if '---' in line:
                continue
                
            parts = [p.strip() for p in line.split('|')]
            # Expected parts length: empty + 8 columns + empty = 10
            # But let's be flexible, at least enough for Project Name
            if len(parts) < 3:
                continue
                
            # parts[0] is empty string before first |
            # parts[1] is Ranking
            # parts[2] is Project Name
            
            try:
                # Extract Project Name and URL
                project_col = parts[2]
                project_name = project_col
                project_url = None
                
                match = re.search(r'\[(.*?)\]\((https://github\.com/[^)]+)\)', project_col)
                if match:
                    project_name = match.group(1)
                    project_url = match.group(2)
                
                # Construct row data
                # We handle potential missing columns gracefully
                row = {
                    "ranking": parts[1] if len(parts) > 1 else "",
                    "project_name": project_name,
                    "project_url": project_url,
                    "stars": parts[3] if len(parts) > 3 else "",
                    "forks": parts[4] if len(parts) > 4 else "",
                    "language": parts[5] if len(parts) > 5 else "",
                    "open_issues": parts[6] if len(parts) > 6 else "",
                    "description": parts[7] if len(parts) > 7 else "",
                    "last_commit": parts[8] if len(parts) > 8 else ""
                }
                
                # Only add if it looks like a valid data row (has ranking)
                if row["ranking"].isdigit():
                    data.append(row)
                    
            except Exception as e:
                print(f"Error parsing line: {line}. Error: {e}")
                continue
                
        return data

    def crawl(self) -> List[dict]:
        """Crawl the ranking repository and return a list of repository data."""
        all_data = []
        files = self.fetch_file_list()
        
        # No limit, process all files
        for filename in files:
            try:
                raw_url = f"{self.RAW_BASE_URL}/{filename}"
                response = requests.get(raw_url)
                response.raise_for_status()
                
                file_data = self.parse_markdown(response.text)
                all_data.extend(file_data)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        return all_data
