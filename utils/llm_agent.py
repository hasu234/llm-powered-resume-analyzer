import sys, httpx, os
sys.dont_write_bytecode = True

from dotenv import load_dotenv

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RAG_K_THRESHOLD = 5
LLM_MODEL = os.getenv("LLM_MODEL")


class ChatBot():
  def __init__(self, api_key: str, model: str):
    self.llm = ChatOpenAI(
      model=model, 
      api_key=api_key, 
      temperature=0.1
    )

  def generate_subquestions(self, question: str):
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition. Break down this job description into 3-4 key aspects to improve resume retrieval.
      Ensure all critical job requirements are covered while removing irrelevant details like job ID or contract duration.
      Only use the information provided in the initial query. Do not make up or hallucinate any information.
      Provide each result in a structured format, one per line.
      """)
    
    user_message = HumanMessage(content=f"""
      Generate 3 to 4 sub-queries based on this job description: 
      {question}
    """)

    oneshot_example = HumanMessage(content="""
      Generate 3 to 4 sub-queries based on this job description:

      Wordpress Developer
      We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
      To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
      WordPress Developer Responsibilities:
      Meeting with clients to discuss website design and function.
      Designing and building the website front-end.
      Creating the website architecture.
      Designing and managing the website back-end including database and server integration.
      Generating WordPress themes and plugins.
      Conducting website performance tests.
      Troubleshooting content issues.
      Conducting WordPress training with the client.
      Monitoring the performance of the live website.
      WordPress Developer Requirements:
      Bachelors degree in Computer Science or a similar field.
      Proven work experience as a WordPress Developer.
      Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
      Knowledge of code versioning tools including Git, Mercurial, and SVN.
      Experience working with debugging tools such as Chrome Inspector and Firebug.
      Good understanding of website architecture and aesthetics.
      Ability to project manage.
      Good communication skills.
    """)

    oneshot_response = AIMessage(content="""
      1. Required Technical Skills:
         - Proficiency in WordPress development, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), and version control (Git, Mercurial, SVN).
         - Experience required: 3 years in WordPress, 2 years in web designing.
      
      2. Key Responsibilities:
         - Meeting with clients to define website needs.
         - Developing the front-end, back-end, and architecture.
         - Managing database and server integration.
         - Creating and maintaining WordPress themes and plugins.
         - Conducting performance tests and troubleshooting issues.
      
      3. Required Qualifications & Experience:
         - Bachelor's degree in Computer Science or a related field.
         - Strong understanding of website architecture and aesthetics.
         - Project management and communication skills.
    """)

    response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message])
    result = response.content.split("\n\n")
    return result

  def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str):
    """
    Generate a stream of messages based on the context and history.

    Args:
        question (str): The question to generate a message stream for.
        docs (list): The list of documents to use as context.
        history (list): The list of chat messages to use as chat history.
        prompt_cls (str): The class of the prompt, either "retrieve_applicant_jd" or "retrieve_applicant_id".

    Returns:
        A stream of messages, where each message is a string.
    """
    context = "\n\n".join(doc for doc in docs)
    
    if prompt_cls == "retrieve_applicant_jd":
      # Generate a system message for the "retrieve_applicant_jd" prompt
      system_message = SystemMessage(content="""
        You are an AI assistant specializing in talent acquisition. Your goal is to analyze resumes and determine the best candidate based on the job description.
        Use the provided context to evaluate resumes, highlighting key qualifications, skills, and experiences that match the job requirements.
        Provide a structured response with clear reasoning for the best selection.
        When referring to candidates, use their applicant ID to avoid confusion.
        If the information is insufficient, state that explicitly instead of making assumptions.
        Link the retreived resume (drive link or file path) to the application ID or name using markdown link format for each applicant.
      """)

      # Generate a user message for the "retrieve_applicant_jd" prompt
      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Context: {context}
        Question: {question}
      """)
    
    else:
      # Generate a system message for the "retrieve_applicant_id" prompt
      system_message = SystemMessage(content="""
        You are an AI assistant that helps HR professionals analyze and extract key information from resumes.
        Use the provided resume context to answer questions accurately and concisely.
        Structure responses with clearly defined sections such as "Experience", "Skills", "Education", and "Certifications" for clarity.
        If the resume lacks specific information, acknowledge that instead of making up details.
        Link the retreived resume (drive link or file path) to the application ID or name using markdown link format for each applicant.
      """)

      # Generate a user message for the "retrieve_applicant_id" prompt
      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Question: {question}
        Context: {context}
      """)

    # Generate a stream of messages using the system message and user message
    stream = self.llm.stream([system_message, user_message])
    return stream
