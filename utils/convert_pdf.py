import pandas as pd
from PyPDF2 import PdfReader
from fastapi import UploadFile
import io
from io import BytesIO
from googleapiclient.http import MediaIoBaseDownload


async def process_pdf_files_fastapi(uploaded_files: list[UploadFile]) -> pd.DataFrame:
    """
    Process uploaded PDF files and return a DataFrame with extracted text
    
    Args:
        uploaded_files (list[UploadFile]): List of FastAPI UploadFile objects
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Name', 'Resume']
    
    Raises:
        Exception: If there's an error processing any PDF file
    """
    data = []
    
    for uploaded_file in uploaded_files:
        print(uploaded_file)
        try:
            # Read the content of the uploaded file into memory
            content = await uploaded_file.read()
            
            # Create a BytesIO object from the content
            pdf_file = io.BytesIO(content)
            
            # Read PDF and extract text
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
            # Add to data list with filename
            data.append({
                'Name': uploaded_file.filename,  # Use filename from UploadFile
                'Link': '',
                'Resume': text
            })
            
            # Reset file pointer for potential future reads
            await uploaded_file.seek(0)
            
        except Exception as e:
            print(f"Error processing PDF {uploaded_file.filename}: {str(e)}")
            raise Exception(f"Failed to process {uploaded_file.filename}: {str(e)}")
    
    return pd.DataFrame(data)

def process_pdf_files(uploaded_files):
    """
    Process uploaded PDF files and return a DataFrame with extracted text
    
    Args:
        uploaded_files (list): List of uploaded PDF files from st.file_uploader
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Name', 'Resume']
    """
    data = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read PDF and extract text
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
            # Add to data list with filename
            data.append({
                'Name': uploaded_file.name,  # Store the original filename
                'Link': '',
                'Resume': text
            })
            
        except Exception as e:
            print("Error processing PDF")
    
    return pd.DataFrame(data)
