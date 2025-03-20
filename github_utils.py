import base64
import requests
import os

def upload_file_to_github(file_path, folder_path, repo_name, github_token):
    """
    Uploads a file to a GitHub repository.
    
    Args:
        file_path (str): Path to the file to upload.
        folder_path (str): Folder in the repository where the file will be uploaded.
        repo_name (str): Name of the GitHub repository (e.g., "username/repo").
        github_token (str): GitHub personal access token.
    
    Returns:
        str: Success or error message.
    """
    try:
        with open(file_path, "rb") as file:
            file_content = file.read()
        
        # Encode the file content in base64
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # GitHub API URL
        url = f"https://api.github.com/repos/{repo_name}/contents/{folder_path}/{file_name}"
        
        # Prepare the request headers and payload
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "message": f"Upload {file_name}",
            "content": encoded_content
        }
        
        # Make the request to upload the file
        response = requests.put(url, headers=headers, json=payload)
        
        if response.status_code == 201:
            return f"File {file_name} uploaded successfully to GitHub."
        else:
            return f"Failed to upload {file_name} to GitHub. Status code: {response.status_code}, Response: {response.text}"
    except Exception as e:
        return f"Error uploading file to GitHub: {e}"
