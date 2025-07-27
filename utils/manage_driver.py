import os
import io
import tempfile
import shutil
import streamlit as st
from io import BytesIO
import pandas as pd
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


class DriveManager:
    def __init__(self, scopes=None):
        """
        Initialize the DriveManager with credentials.

        Args:
            credentials_path (str): Path to your service account JSON file.
            scopes (list, optional): List of API scopes. Defaults to full Drive access.
        """
        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/drive']
        self.credentials = service_account.Credentials.from_service_account_file(
            'utils/credentials.json', scopes=scopes)
        self.service = build('drive', 'v3', credentials=self.credentials)

    def create_folder(self, folder_name, parent_folder_id=None):
        """
        Create a folder in Google Drive if it doesn't already exist.
        
        Args:
            folder_name (str): Name of the folder to create.
            parent_folder_id (str, optional): ID of the parent folder.
                If None, the folder is created in the root directory.
        
        Returns:
            str: The ID of the folder (either existing or newly created).
        """
        # Build the query to check if folder exists
        if parent_folder_id:
            query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        else:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false"
        
        # Execute the query
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = results.get('files', [])
        
        # If folder exists, return its ID
        if folders:
            print(f"Folder '{folder_name}' already exists.")
            return True, folders[0]['id']
        
        # Otherwise, create the folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
            
        folder = self.service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        print(f"Created new folder '{folder_name}'.")
        return False, folder.get('id')

    def upload_file(self, folder_id, resume_path, replace_if_exists):
        """
        Uploads a resume (file) to a specified folder in Google Drive.
        
        Args:
            folder_id (str): The ID of the folder where the file will be uploaded.
            resume_path (str or UploadedFile): The local file path or Streamlit UploadedFile object.
            replace_if_exists (bool): Whether to replace the existing file if it exists.
                                    If False, the function will skip uploading.
        
        Returns:
            str: The ID of the uploaded file, or the ID of the existing file if skipped.
            None: If the file exists and replace_if_exists is False.
        """

        # Handle both file paths and Streamlit UploadedFile
        if isinstance(resume_path, str):
            file_name = os.path.basename(resume_path)
            temp_file_path = resume_path  # Use as is
        elif isinstance(resume_path, st.runtime.uploaded_file_manager.UploadedFile):
            file_name = resume_path.name
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(resume_path.getbuffer())
                temp_file_path = temp_file.name
        else:
            raise ValueError("resume_path must be a file path (str) or a Streamlit UploadedFile object.")

        try:
            # Check if file already exists in the folder
            existing_files = self.list_files_in_folder(folder_id)
            existing_file = next((file for file in existing_files if file['name'] == file_name), None)

            if existing_file:
                if replace_if_exists:
                    # Delete the existing file
                    self.service.files().delete(fileId=existing_file['id']).execute()
                    print(f"Replaced existing file: {file_name}")
                else:
                    # Skip uploading
                    print(f"Skipped existing file: {file_name}")
                    return existing_file['id']

            # Upload the new file
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }

            media = MediaFileUpload(temp_file_path, resumable=True)

            file = self.service.files().create(
                body=file_metadata, 
                media_body=media,
                fields='id'
            ).execute()

            return file.get('id')
        
        finally:
            # Remove temporary file if used
            if isinstance(resume_path, st.runtime.uploaded_file_manager.UploadedFile):
                os.remove(temp_file_path)

    def download_files(self, folder_id, destination_path):
        """
        Downloads all resumes (files) from a specified folder in Google Drive.

        Args:
            folder_id (str): The ID of the folder to download files from.
            destination_path (str): The local directory where files should be saved.

        Returns:
            list: A list of file names that were downloaded.
        """
        # Ensure the destination directory exists
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        files = self.list_files_in_folder(folder_id)
        downloaded_files = []
        for file in files:
            file_id = file['id']
            file_name = file['name']
            request = self.service.files().get_media(fileId=file_id)
            dest_file_path = os.path.join(destination_path, file_name)
            with io.FileIO(dest_file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
            downloaded_files.append(file_name)
        return downloaded_files

    def list_subfolders(self, parent_folder_id):
        """
        List all subfolders under a given parent folder.

        Args:
            parent_folder_id (str): The ID of the parent folder.

        Returns:
            list: A list of subfolder dictionaries with 'id' and 'name' keys.
        """
        query = f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        return folders

    def list_files_in_folder(self, folder_id):
        """
        List all files within a specified folder.

        Args:
            folder_id (str): The ID of the folder.

        Returns:
            list: A list of file dictionaries with 'id', 'name', and 'mimeType' keys.
        """
        query = f"'{folder_id}' in parents and trashed = false"
        results = self.service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])
        return files
    
    def get_shareable_links(self, folder_id):
        """
        Get the names and view-only shareable links for all files in a folder.
        
        Args:
            folder_id (str): The ID of the folder to get files from.
            
        Returns:
            list: A list of dictionaries containing file names and their view-only shareable links.
        """
        files = self.list_files_in_folder(folder_id)
        file_links = []
        
        for file in files:
            try:
                file_id = file['id']
                file_name = file['name']
                
                # Create permission for anyone with the link to view
                permission = {
                    'type': 'anyone',
                    'role': 'reader',
                    'allowFileDiscovery': False
                }
                
                # Apply the permission to the file
                self.service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    fields='id'
                ).execute()
                
                # Get the shareable link
                # Format: https://drive.google.com/file/d/{file_id}/view
                shareable_link = f"https://drive.google.com/file/d/{file_id}/view"
                
                file_links.append({
                    'Name': file_name,
                    'Link': shareable_link
                })
                
                
            except Exception as e:
                print(f"Error generating link for {file.get('name', 'unknown file')}: {str(e)}")
        
        return file_links
    
    def process_drive_resumes(self, category_id):
        files = self.list_files_in_folder(category_id)
        data = []
        
        for file in files:
            try:
                file_id = file['id']
                file_name = file['name']
                
                # Skip non-PDF files
                if not file_name.lower().endswith('.pdf'):
                    print(f"Skipping non-PDF file: {file_name}")
                    continue
                
                # Create permission for anyone with the link to view
                permission = {
                    'type': 'anyone',
                    'role': 'reader',
                    'allowFileDiscovery': False
                }
                
                # Apply the permission to the file
                self.service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    fields='id'
                ).execute()
                
                # Download file into memory
                request = self.service.files().get_media(fileId=file_id)
                file_bytes = BytesIO()
                downloader = MediaIoBaseDownload(file_bytes, request)
                
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Processing {file_name}: {int(status.progress() * 100)}%")
                
                # Reset buffer position
                file_bytes.seek(0)
                
                # Process PDF
                reader = PdfReader(file_bytes)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
                # Add to data list
                data.append({
                    'Name': file_name,
                    'Link': f"https://drive.google.com/file/d/{file_id}/view",
                    'Resume': text
                })
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
        
        return pd.DataFrame(data)
    

