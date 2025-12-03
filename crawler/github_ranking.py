import requests
import re
from typing import List

class GithubRankingCrawler:
    BASE_URL = "https://github.com/EvanLi/Github-Ranking/tree/master/Top100"
    RAW_BASE_URL = "https://raw.githubusercontent.com/EvanLi/Github-Ranking/master/Top100"

    def fetch_file_list(self) -> List[str]:
        """Fetch list of markdown files from the ranking repository."""
        try:
            response = requests.get(self.BASE_URL)
            response.raise_for_status()
            
            # Simple regex to find links to markdown files in the Top100 directory
            # Pattern looks for hrefs ending in .md
            # Note: GitHub HTML structure might change, but this is a basic approach
            # Looking for href="/EvanLi/Github-Ranking/blob/master/Top100/Top-100-stars.md"
            pattern = r'href="/EvanLi/Github-Ranking/blob/master/Top100/([^"]+\.md)"'
            files = re.findall(pattern, response.text)
            
            return list(set(files)) # Remove duplicates
        except Exception as e:
            print(f"Error fetching file list: {e}")
            return []

    def parse_markdown(self, content: str) -> List[str]:
        """Parse markdown content to extract GitHub repository URLs from the 'Project Name' column."""
        urls = []
        lines = content.split('\n')
        
        # Table parsing logic
        # We look for lines starting with | and containing a link in the 2nd column
        # | Ranking | Project Name | ...
        # | 1 | [name](url) | ...
        
        for line in lines:
            line = line.strip()
            if not line.startswith('|'):
                continue
                
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
                
            # 2nd column (index 2 because split creates empty string at start)
            # | 1 | [name](url) | ... -> ['', '1', '[name](url)', ...]
            project_col = parts[2]
            
            # Extract URL from markdown link [name](url)
            match = re.search(r'\[.*?\]\((https://github\.com/[^)]+)\)', project_col)
            if match:
                urls.append(match.group(1))
                
        return urls

    def crawl(self, limit: int = None) -> List[str]:
        """Crawl the ranking repository and return a list of GitHub repository URLs."""
        all_urls = []
        files = self.fetch_file_list()
        
        if limit:
            files = files[:limit]
            
        for filename in files:
            try:
                raw_url = f"{self.RAW_BASE_URL}/{filename}"
                response = requests.get(raw_url)
                response.raise_for_status()
                
                urls = self.parse_markdown(response.text)
                all_urls.extend(urls)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        return list(dict.fromkeys(all_urls)) # Unique URLs preserving order
