"""
This module contains a class for a retriever that can retrieve documents
from a vector store based on a query. The retriever also supports
reranking the retrieved documents using reciprocal rank fusion.
"""

import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
    """
    List of IDs of the applicants to retrieve resumes for
    """
    id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")


class JobDescription(BaseModel):
    """
    Descriptions of a job to retrieve similar resumes for
    """
    job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for")


class RAGRetriever:
    def __init__(self, vectorstore_db, df):
        """
        Initialize the retriever with a vector store and a DataFrame.

        Args:
            vectorstore_db (VectorStore): The vector store to use.
            df (pd.DataFrame): The DataFrame containing the resumes.
        """
        self.vectorstore = vectorstore_db
        self.df = df

    def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
        """
        Rerank the documents using reciprocal rank fusion.

        Args:
            document_rank_list (list[dict]): A list of dictionaries containing the documents and their scores.
            k (int): The number of documents to consider for reranking.

        Returns:
            dict: A dictionary with the documents and their reranked scores.
        """
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results

    def __retrieve_docs_id__(self, question: str, k=50):
        """
        Retrieve documents from the vector store based on a query.

        Args:
            question (str): The query to use for retrieval.
            k (int): The number of documents to retrieve.

        Returns:
            dict: A dictionary with the documents and their scores.
        """
        docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
        docs_score = {str(doc.metadata["Name"]): score for doc, score in docs_score}
        return docs_score

    def retrieve_id_and_rerank(self, subquestion_list: list):
        """
        Retrieve documents from the vector store based on a list of subquestions and rerank them.

        Args:
            subquestion_list (list): A list of subquestions to use for retrieval.

        Returns:
            dict: A dictionary with the documents and their reranked scores.
        """
        document_rank_list = []
        for subquestion in subquestion_list:
            document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
        reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
        return reranked_documents

    def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
        """
        Retrieve the documents from the DataFrame based on the IDs and scores.

        Args:
            doc_id_with_score (dict): A dictionary with the IDs and scores.
            threshold (int): The number of documents to retrieve.

        Returns:
            list: A list of strings containing the retrieved documents.
        """
        id_resume_dict = dict(zip(self.df["Name"].astype(str), self.df["Resume"]))
        id_link_dict = dict(zip(self.df["Name"].astype(str), self.df["Link"]))

        retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
        retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
        
        for i in range(len(retrieved_documents)):
            resume_link = id_link_dict.get(retrieved_ids[i], "No link available")
            retrieved_documents[i] = f"Applicant Name: {retrieved_ids[i]} ;\nResume Link: {resume_link} \n\nResume Contents: {retrieved_documents[i]}"

        return retrieved_documents


class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db, df):
        """
        Initialize the retriever with a vector store and a DataFrame.

        Args:
            vectorstore_db (VectorStore): The vector store to use.
            df (pd.DataFrame): The DataFrame containing the resumes.
        """
        super().__init__(vectorstore_db, df)
        # Add file link mapping
        self.id_to_link = dict(zip(self.df["Name"].astype(str), self.df["Link"].astype(str)))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in talent acquisition."),
            ("user", "{input}")
        ])
        self.meta_data = {
            "rag_mode": "",
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subquestion_list": [],
            "retrieved_docs_with_scores": [],
            "retrieved_file_links": []  # Add tracking for file paths
        }

    def retrieve_docs(self, question: str, llm, rag_mode: str):
        @tool(args_schema=ApplicantID)
        def retrieve_applicant_id(id_list: list):
            """Retrieve resumes for applicants in the id_list"""
            retrieved_resumes = []
            if not id_list:
                print("Warning: Empty ID list provided")
                return []
            
            for id in id_list:
                try:
                    # we can Implement llm for better results
                    # ----------------------------------------
                    # Convert both to string and strip whitespace for robust comparison
                    matching_rows = self.df[self.df["Name"].astype(str).str.contains(str(id).strip(), case=False, na=False)]
                    # matching_rows = self.df[self.df["Name"].astype(str).str.strip() == str(id).strip()]
                    
                    if matching_rows.empty:
                        print(f"Warning: No match found for ID {id}")
                        continue

                    resume_df = matching_rows.iloc[0][["Name", "Link", "Resume"]]
                        
                    # resume_with_id = "Applicant Name " + str(resume_df["Name"]) + "\n" + str(resume_df["Resume"])
                    resume_with_id = f"Applicant Name: {str(resume_df["Name"])} ;\nResume Link: {str(resume_df["Link"])} \n\n Contents: {str(resume_df["Resume"])}"
                    retrieved_resumes.append(resume_with_id)
                    
                except Exception as e:
                    print(f"Error processing ID {id}: {str(e)}")
                    continue
                    
            # If no resumes were retrieved, try to get some default results
            if not retrieved_resumes:
                print("No resumes retrieved, attempting to get default results")
                try:
                    # Get first 3 resumes as fallback
                    fallback_df = self.df.head(3)[["Name", "Link", "Resume"]]
                    for _, row in fallback_df.iterrows():
                        # resume_with_id = "Applicant Name " + str(row["Name"]) + "\n" + str(row["Resume"])
                        resume_with_id = f"Applicant Name: {str(row["Name"])} ;\nResume Link: {str(resume_df["Link"])} \n\n Contents: {str(row["Resume"])}"
                        retrieved_resumes.append(resume_with_id)
                except Exception as e:
                    print(f"Fallback retrieval failed: {str(e)}")
                    
            return retrieved_resumes

        @tool(args_schema=JobDescription)
        def retrieve_applicant_jd(job_description: str):
            """Retrieve similar resumes given a job description"""
            if not job_description:
                print("Warning: Empty job description provided")
                fallback_resumes = retrieve_applicant_id({"id_list": self.df["Name"].head(3).tolist()})
                return fallback_resumes
                
            subquestion_list = [job_description]
            
            if rag_mode == "RAG Fusion":
                try:
                    additional_questions = llm.generate_subquestions(question)
                    if additional_questions:
                        subquestion_list.extend(additional_questions)
                except Exception as e:
                    print(f"Error generating subquestions: {str(e)}")
                    
            self.meta_data["subquestion_list"] = subquestion_list
            
            try:
                retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
                if not retrieved_ids:
                    print("No IDs retrieved, using fallback method")
                    retrieved_ids = [(id, 1.0) for id in self.df["Name"].head(3).tolist()]
            except Exception as e:
                print(f"Error in ID retrieval: {str(e)}")
                retrieved_ids = [(id, 1.0) for id in self.df["Name"].head(3).tolist()]
                
            self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
            retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
            
            # # Get file paths for retrieved resumes
            # retrieved_paths = []
            # for resume in retrieved_resumes:
            #     applicant_name = resume.split("\n")[0].replace("Applicant Name ", "")
            #     file_link = self.id_to_link.get(applicant_name, "NA")
            #     retrieved_paths.append(file_link)
            
            return retrieved_resumes

        def router(response):
            if isinstance(response, AgentFinish):
                return response.return_values["output"]
            else:
                toolbox = {
                    "retrieve_applicant_id": retrieve_applicant_id,
                    "retrieve_applicant_jd": retrieve_applicant_jd
                }
                self.meta_data["query_type"] = response.tool
                self.meta_data["extracted_input"] = response.tool_input
                
                try:
                    result = toolbox[response.tool].run(response.tool_input)
                    if not result:  # If empty result
                        print("Empty result from tool, using fallback")
                        result = retrieve_applicant_id(self.df["Name"].head(3).tolist())
                    return result
                except Exception as e:
                    print(f"Error in router: {str(e)}")
                    return retrieve_applicant_id(self.df["Name"].head(3).tolist())

        self.meta_data["rag_mode"] = rag_mode
        
        try:
            llm_func_call = llm.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in [retrieve_applicant_id, retrieve_applicant_jd]])
            chain = self.prompt | llm_func_call | OpenAIFunctionsAgentOutputParser() | router
            result = chain.invoke({"input": question})
            
            # Ensure we always return something
            if not result:
                print("Empty result from chain, using fallback")
                result = retrieve_applicant_id(self.df["Name"].head(3).tolist())
                
            return result
            
        except Exception as e:
            print(f"Error in retrieve_docs: {str(e)}")
            return retrieve_applicant_id(self.df["Name"].head(3).tolist())
