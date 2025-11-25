import os
import subprocess
import sys

# The specific repo we are wrapping
REPO_URL = "https://github.com/MCG-NJU/SAM2-Plus"
# The directory where we will clone it (inside our custom node folder)
REPO_DIR = os.path.join(os.path.dirname(__file__), "sam2_plus_repo")

def git_clone(url, target_dir):
    if os.path.exists(target_dir):
        return
    
    print(f"Downloading SAM2-Plus repository to {target_dir}...")
    try:
        subprocess.check_call(["git", "clone", url, target_dir])
        print("SAM2-Plus repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

def install():
    git_clone(REPO_URL, REPO_DIR)
    
    # Optional: Install the repo requirements if they differ
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(REPO_DIR, "requirements.txt")])

if __name__ == "__main__":
    install()