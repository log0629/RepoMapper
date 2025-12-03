import requests
import subprocess
import os
import schedule
import time

SERVER_URL = "http://localhost:8000"
DATA_DIR = "data/repos"

def run_job():
    print("Starting daily job...")
    try:
        # 1. Crawl
        print("Crawling GitHub ranking...")
        crawl_response = requests.post(f"{SERVER_URL}/crawl/github-ranking")
        crawl_response.raise_for_status()
        repos = crawl_response.json().get("data", [])
        print(f"Found {len(repos)} repositories.")

        for repo in repos:
            project_url = repo.get("project_url")
            if not project_url:
                continue
                
            # Extract org and repo name
            # https://github.com/org/repo -> org, repo
            parts = project_url.strip("/").split("/")
            if len(parts) < 2:
                continue
            
            org_name = parts[-2]
            repo_name = parts[-1]
            folder_name = f"{org_name}_{repo_name}"
            repo_path = os.path.abspath(os.path.join(DATA_DIR, folder_name))
            
            # 2. Clone
            if not os.path.exists(repo_path):
                print(f"Cloning {project_url} to {repo_path}...")
                try:
                    subprocess.run(["git", "clone", project_url, repo_path], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to clone {project_url}: {e}")
                    continue
            else:
                print(f"Repository {folder_name} already exists. Updating...")
                try:
                    subprocess.run(["git", "-C", repo_path, "pull"], check=True)
                    print(f"Updated {folder_name} successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to update {folder_name}: {e}")
                    # Continue to index even if update fails? Or skip?
                    # Let's continue to index the current state.
                    pass
            
            # 3. Index
            print(f"Indexing {folder_name}...")
            try:
                index_response = requests.post(
                    f"{SERVER_URL}/index",
                    json={
                        "root_path": repo_path
                    }
                )
                index_response.raise_for_status()
                print(f"Indexed {folder_name} successfully.")
            except Exception as e:
                print(f"Failed to index {folder_name}: {e}")

    except Exception as e:
        print(f"Job failed: {e}")

if __name__ == "__main__":
    # Schedule to run every day at 00:00
    schedule.every().day.at("00:00").do(run_job)
    
    print("Scheduler started. Waiting for jobs...")
    # Run once immediately for demonstration/testing if needed, or just loop
    # run_job() # Uncomment to run immediately on start
    
    while True:
        schedule.run_pending()
        time.sleep(60)
