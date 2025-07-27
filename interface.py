import sys
import os
import configparser
import time
import tempfile
import shutil
import torch
import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.manage_driver import DriveManager
from utils.llm_agent import ChatBot
from utils.retriever import SelfQueryRetriever
import utils.chatbot_verbosity as chatbot_verbosity
from utils.convert_pdf import process_pdf_files
from utils.configuration import read_config, write_config

sys.dont_write_bytecode = True
load_dotenv()
config = configparser.ConfigParser()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
api_key = os.getenv("OPENAI_API_KEY")
gpt_selection = os.getenv("LLM_MODEL")

drive_manager = DriveManager()
device = "cuda" if torch.cuda.is_available() else "cpu"

welcome_message = """
### 📌 Introduction
Recruitment processes generate vast amounts of unstructured data—CVs, job descriptions, and candidate information. Our **AI-powered Retrieval-Augmented Generation (RAG) system** revolutionizes how hiring managers interact with this data. Simply upload PDFs, integrate with Google Drive, and **chat intelligently** to extract the insights you need in seconds!

With multiple **Chat Modes**, you can:
✔ **Search Candidates** based on Job Descriptions
✔ **Interact with Specific CVs** using Candidate IDs
✔ **Have General Hiring Conversations**
✔ **Chat with Local Files**
✔ **Sync with Google Drive** (upload new CVs, access existing vector databases)
✔ **Maintain Clean Conversations** for seamless interactions

No more endless document searching—just **ask, and get precise answers instantly!**

### 🛠️ How to Use

1️⃣ **Select RAG Mode** – Choose **Generic Chat** or **RAG Fusion** for enhanced insights.
2️⃣ **Select File Source** – Choose **Local Files** or **Google Drive**.
3️⃣ **Local File** – If you choose Local Files, upload PDFs and chat with them.
4️⃣ **Google Drive** – If you select Google Drive as the source, choose a team category and chat with existing vector databases by clicking Start Chat.
5️⃣ **Upload More Files** – Upload new CVs to Google Drive and sync them with the vector database.
6️⃣ **Keep Conversations Clean** – Reset or switch context as needed.

Your **intelligent hiring assistant** is ready to **simplify recruitment like never before!**
"""

about_message = """
# About

This project leverages **AI, NLP, and vector databases** to streamline hiring workflows. Designed with a **Recruitment Manager’s needs in mind**, it transforms unstructured text data into actionable insights.
"""

st.set_page_config(page_title="SOL Resume Chat", page_icon="📄")
st.title("SOL Resume Chat")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device})

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

if "rag_pipeline" not in st.session_state:
  st.session_state.rag_pipeline = None

if "upload_option" not in st.session_state:
  st.session_state.upload_option = None

if "show_team_creation" not in st.session_state:
  st.session_state.show_team_creation = False

if "category_selection" not in st.session_state:
  st.session_state.category_selection = None

if "uploaded_files_to_drive" not in st.session_state:
  st.session_state.uploaded_files_to_drive = False

embedding_model = HuggingFaceEmbeddings(
  model_name=EMBEDDING_MODEL,
  model_kwargs={"device": device}
)

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1024,
  chunk_overlap=500
)

def handle_upload():
  """
  Callback function to handle file upload changes and update session state
  """
  if st.session_state.uploaded_files:
    try:
      # Process the PDFs
      df = process_pdf_files(st.session_state.uploaded_files)
      
      # Update the session state DataFrame
      loader = DataFrameLoader(df, page_content_column="Resume")
      documents = loader.load()
      document_chunks = text_splitter.split_documents(documents)

      start = time.time()
      vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
      end = time.time()
      print(f"{end - start} seconds")

      st.session_state.rag_pipeline = SelfQueryRetriever(vectorstore_db, df)
      st.success(f"Successfully processed {len(df)} PDF files")
      
    except Exception as e:
      st.error(f"Error processing files: {str(e)}")

def upload_new_files_to_drive():
  """
  Callback function to handle Google Drive file processing
  """
  if st.session_state.uploaded_files:
    try:
      temp_dir = tempfile.mkdtemp()
      folder_id = read_config("drive", st.session_state.category_selection)
      for file in st.session_state.uploaded_files:
        drive_manager.upload_file(folder_id, file, replace_if_exists=False)
        print(f"Uploaded {file.name}")

      category_directory_id = read_config("drive", st.session_state.category_selection)
      temp_save_path = os.path.join(temp_dir, 'vectorstore_db')

      os.makedirs(temp_save_path, exist_ok=True)

      df = drive_manager.process_drive_resumes(category_directory_id)
      df.to_csv(os.path.join(temp_save_path, 'resumes.csv'), index=False)
      
      # Update the session state DataFrame
      loader = DataFrameLoader(df, page_content_column="Resume")
      documents = loader.load()
      document_chunks = text_splitter.split_documents(documents)
      
      start = time.time()
      vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
      end = time.time()
      print(f"{end - start} seconds time taken to build vectorstore_db")

      # save the vectorstore_db to drive
      vectorstore_db.save_local(temp_save_path)

      _, folder_id = drive_manager.create_folder('vectorstore_db', category_directory_id)
      for file in os.listdir(temp_save_path):
        drive_manager.upload_file(folder_id, os.path.join(temp_save_path, file), replace_if_exists=True)
      
      st.session_state.rag_pipeline = SelfQueryRetriever(vectorstore_db, df)
      st.success(f"Successfully processed {len(df)} PDF files")
      shutil.rmtree(temp_save_path)
      
    except Exception as e:
      st.error(f"Error processing files: {str(e)}")

def toggle_team_creation():
  st.session_state.show_team_creation = True

def create_team():
  if st.session_state.new_team_name:
    parent_folder_id = read_config("drive", "parent_folder_id")
    exist, new_folder_id = drive_manager.create_folder(st.session_state.new_team_name, parent_folder_id)
    
    if not exist:
      write_config('drive', st.session_state.new_team_name, new_folder_id)
      st.success(f"Team '{st.session_state.new_team_name}' created successfully")
      # Reset the form
      st.session_state.show_team_creation = False
      st.session_state.new_team_name = ""
    else:
      st.error(f"Team '{st.session_state.new_team_name}' already exists")
  else:
    st.error("Please enter a team name")

def cancel_team_creation():
  st.session_state.show_team_creation = False
  if 'new_team_name' in st.session_state:
    st.session_state.new_team_name = ""

def from_drive_vectorstore():
  """
  Callback function to handle Google Drive file processing
  """
  if st.session_state.category_selection:
    try:
      temp_dir = tempfile.mkdtemp()
      temp_path = os.path.join(temp_dir, 'vectorstore_db')
      os.makedirs(temp_path, exist_ok=True)

      folder_id = read_config("drive", st.session_state.category_selection)
      subfolders = drive_manager.list_subfolders(folder_id)

      # Find the folder named "vectorstore_db"
      vectorstore_db_folder = next((folder for folder in subfolders if folder['name'] == 'vectorstore_db'), None)
      
      if vectorstore_db_folder:
        vectorstore_db_id = vectorstore_db_folder['id']
        drive_manager.download_files(vectorstore_db_id, temp_path)
        df = pd.read_csv(os.path.join(temp_path, 'resumes.csv'))
        os.remove(os.path.join(temp_path, 'resumes.csv'))

        vectordb = FAISS.load_local(temp_path, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)

        st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, df)
        st.success(f"Successfully processed {len(df)} PDF files")

      else:
        st.error("No prebuild vectorstore found in the selected category")

      shutil.rmtree(temp_path)
      
    except Exception as e:
      st.error(f"Error processing files: {str(e)}")
  else:
    st.error("Please select a team category and chat option first")

def check_model_name(model_name: str, api_key: str):
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  return True if model_name in model_list else False

def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  # Initialize session state to track selection
  if "rag_selection" not in st.session_state:
    st.session_state.rag_selection = "Generic RAG"  # Default selection

  # Set a title
  st.subheader("Select RAG Mode")

  # Create two buttons side by side
  col1, col2 = st.columns(2)

  with col1:
    if st.button("Generic RAG"):
      st.session_state.rag_selection = "Generic RAG"

  with col2:
    if st.button("RAG Fusion"):
      st.session_state.rag_selection = "RAG Fusion"

  # Show the selected mode
  st.write(f"**Selected Mode:** {st.session_state.rag_selection}")

  st.subheader("Select File Source")
  col1, col2 = st.columns(2)

  with col1:
    if st.button("Local File"):
      st.session_state.upload_option = "Local File"

  with col2:
    if st.button("Google Drive"):
      st.session_state.upload_option = "Google Drive"

  # Show the appropriate input field based on the selection
  if st.session_state.upload_option == "Local File":
    uploaded_files = st.file_uploader(
      "Upload PDF files",
      type=["pdf"],
      accept_multiple_files=True,
      key="uploaded_files",
      on_change=handle_upload
    )

  elif st.session_state.upload_option == "Google Drive":
    st.markdown("Select Team")
    parent_folder_id = read_config("drive", "parent_folder_id")

    # Category selection buttons
    categories = drive_manager.list_subfolders(parent_folder_id)
    category_names = [item['name'] for item in categories]

    # Display buttons in a grid layout
    cols = st.columns(3)  # Adjust the number for layout preferences

    for idx, category in enumerate(category_names):
      with cols[idx % 3]:  # Distribute buttons evenly across columns
        if st.button(category):
          st.session_state.category_selection = category

    st.write(f"**Selected Category:** {st.session_state.category_selection}")

    # Create two buttons side by side
    col1, col2 = st.columns(2)

    with col1:
      st.session_state.uploaded_files_to_drive = "Start Chat"
      st.button("Start Chat", 
            key="from_drive_vectorstore_button", 
            on_click=from_drive_vectorstore
      )

    with col2:
      if st.button("Upload Files"):
        st.session_state.uploaded_files_to_drive = "File"
    
    if st.session_state.uploaded_files_to_drive == "File":
      if st.session_state.category_selection:
        try:
          uploaded_files = st.file_uploader(
            "Upload New PDF Files to Drive",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=upload_new_files_to_drive
          )
        except:
          st.error(f"Error uploading files.")
      else:
        st.error("Please select a team category first")

    if not st.session_state.show_team_creation:
      st.button("Create New Team", on_click=toggle_team_creation)
    else:
      # Use a form to enable Enter key submission
      with st.form(key="team_creation_form", clear_on_submit=False):
        st.text_input("Enter new team name:", key="new_team_name")
        # Hidden submit button that will be triggered by Enter key
        submitted = st.form_submit_button("Submit", on_click=create_team)
      
      # Add a cancel button outside the form
      st.button("Cancel", on_click=cancel_team_creation)
  
  st.subheader("Conversation Controls")
  st.button("Clear conversation", on_click=clear_message)

  st.divider()
  st.markdown(about_message)

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])

retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=api_key,
  model=gpt_selection,
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))
