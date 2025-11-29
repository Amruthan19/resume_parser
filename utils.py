# resume_ranker.py
import os
import logging
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import tempfile
import pandas as pd
import json
import re
from PyPDF2 import PdfReader
import io  # Added this import
from dotenv import load_dotenv
load_dotenv()

class SmoothException(Exception):
    """Custom exception class"""
    pass

class ResumeRanker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_vector_store(self, docs, collection, qdrant_host, qdrant_api_key, dimension, embeddings):
        """Create vector store for resume chunks"""
        try:
            client = QdrantClient(
                url=qdrant_host,
                api_key=qdrant_api_key,
                timeout=60.0
            )
            if not client.collection_exists(collection_name=collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                )
            
            # Use from_documents instead of add_documents to preserve metadata
            vector_store = Qdrant.from_documents(
                documents=docs,
                embedding=embeddings,
                url=qdrant_host,
                api_key=qdrant_api_key,
                collection_name=collection,
            )
            return vector_store
        except (ResponseHandlingException, UnexpectedResponse) as e:
            raise SmoothException(e.reason_phrase)

    def parse_pdf(self, file_content, filename):
        """Parse PDF file content"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
                
            # Ensure source is properly set in metadata
            for doc in docs:
                doc.metadata["source"] = filename
                # Also add the filename to page_content metadata for redundancy
                doc.metadata["filename"] = filename
                
            return docs
            
        except Exception as e:
            raise Exception(f"PDF parsing error: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def query_resumes(self, query, collection, qdrant_host, qdrant_api_key, embeddings, k):
        """Query resumes based on job description"""
        try:
            client = QdrantClient(
                url=qdrant_host,
                api_key=qdrant_api_key,
            )
            embeddings_data = embeddings.embed_query(query)
            
            result = client.query_points(
                collection_name=collection,
                query=embeddings_data,
                limit=k,
                with_payload=True  # Ensure payload is included
            )
            
            # Return both content and scores
            results = []
            for point in result.points:
                # Try multiple possible metadata fields for the filename
                source = point.payload.get("source") or point.payload.get("filename") or "Unknown"
                
                results.append({
                    'content': point.payload.get("page_content", ""),
                    'score': point.score,
                    'source': source
                })
            return results
            
        except Exception as e:
            self.logger.error(f"Qdrant query error: {str(e)}")
            raise SmoothException(f"Qdrant query failed: {str(e)}")

    def load_models(self, cohere_key, groq_key):
        """Load Cohere embeddings and Groq LLM"""
        embeddings = CohereEmbeddings(cohere_api_key=cohere_key, model="embed-english-v3.0")
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return embeddings, llm

    def validate_pdf(self, file_content, filename):
        """Validate PDF file before processing"""
        try:
            if len(file_content) == 0:
                return False, "File is empty"
            
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            if reader.is_encrypted:
                return False, "PDF is encrypted/password protected"
            
            if len(reader.pages) == 0:
                return False, "PDF has no pages"
            
            first_page = reader.pages[0]
            text = first_page.extract_text()
            if not text or len(text.strip()) < 10:
                return False, "PDF appears to be empty or contains no readable text"
            
            return True, "PDF is valid"
            
        except Exception as e:
            return False, f"PDF validation failed: {str(e)}"

    def parse_pdf(self, file_content, filename):
        """Parse PDF file content"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
                
            for doc in docs:
                doc.metadata["source"] = filename
                
            return docs
            
        except Exception as e:
            raise Exception(f"PDF parsing error: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def analyze_resume_match(self, llm, job_desc, resume_text, resume_name):
        """Analyze how well a resume matches the job description"""
        
        prompt = f"""
        Analyze this resume against the job description and provide a comprehensive assessment.
        
        JOB DESCRIPTION:
        {job_desc[:1500]}
        
        RESUME CONTENT:
        {resume_text[:2000]}
        
        Provide your analysis in this EXACT JSON format:
        {{
            "score": 85,
            "key_skills_match": "Python, Django, AWS",
            "experience_match": "5 years relevant experience",
            "strengths": "Strong technical skills, relevant project experience",
            "gaps": "Missing cloud certification, limited team leadership",
            "overall_assessment": "Excellent match with strong alignment to required skills"
        }}
        
        Scoring guidelines:
        - 90-100: Excellent match (most requirements met, strong alignment)
        - 80-89: Very good match (most requirements met, minor gaps)
        - 70-79: Good match (core requirements met, some gaps)
        - 60-69: Partial match (some requirements met, significant gaps)
        - Below 60: Poor match (few requirements met)
        
        Be objective and focus on:
        1. Skills and technologies match
        2. Experience level and relevance
        3. Education and qualifications
        4. Overall fit for the role
        """
        
        try:
            response = llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                # Fallback analysis
                return {
                    "score": 50,
                    "key_skills_match": "Analysis unavailable",
                    "experience_match": "Analysis unavailable",
                    "strengths": "Content extracted but analysis failed",
                    "gaps": "Unable to complete analysis",
                    "overall_assessment": "Analysis incomplete"
                }
        except Exception as e:
            print(f"Analysis error for {resume_name}: {str(e)}")
            return {
                "score": 50,
                "key_skills_match": "Analysis error",
                "experience_match": "Analysis error",
                "strengths": "Unable to analyze",
                "gaps": "Analysis failed",
                "overall_assessment": "Error in processing"
            }

    def rank_resumes(self, job_desc, pdf_files_data, cohere_key, groq_key, k=10):
        """Main function to rank resumes against job description"""
        
        if not cohere_key or not groq_key:
            raise Exception("Cohere & Groq API Keys are required!")
        
        if not pdf_files_data:
            raise Exception("No resume files provided!")
        
        # Qdrant configuration
        QDRANT_HOST = "https://7960e3d6-0728-42b6-8d09-aa6e3d9bd085.sa-east-1-0.aws.cloud.qdrant.io"
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        COLLECTION_NAME = "resume_collection"
        DIMENSION = 1024
        
        # Load models
        embeddings, llm = self.load_models(cohere_key, groq_key)
        
        # Process PDF files
        docs = []
        processed_files = 0
        
        for file_data in pdf_files_data:
            filename = file_data['name']
            file_content = file_data['content']
            print("--------file data----------------------",filename)
            try:
                is_valid, validation_msg = self.validate_pdf(file_content, filename)
                
                if not is_valid:
                    print(f"Skipping {filename}: {validation_msg}")
                    continue
                
                parsed_docs = self.parse_pdf(file_content, filename)
                
                if parsed_docs:
                    docs.extend(parsed_docs)
                    processed_files += 1
                    print(f"Processed: {filename}")
                else:
                    print(f"No content extracted from: {filename}")
                    
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
        
        if processed_files == 0:
            raise Exception("No valid PDF files were processed!")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create vector store
        vector_store = self.create_vector_store(
            docs=chunks,
            collection=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            dimension=DIMENSION,
            embeddings=embeddings
        )
        
        # Search for matching resume chunks
        matches = self.query_resumes(
            query=job_desc,
            collection=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            embeddings=embeddings,
            k=min(k * 3, len(chunks))  # Get more chunks for analysis
        )
        
        # Group matches by resume source
        resume_matches = {}
        for match in matches:
            source = match['source']
            if source not in resume_matches:
                resume_matches[source] = []
            resume_matches[source].append(match)
        
        # Analyze each resume
        results = []
        
        for resume_name, match_data in resume_matches.items():
            # Combine top chunks for this resume
            top_chunks = sorted(match_data, key=lambda x: x['score'], reverse=True)[:5]
            combined_text = " ".join([chunk['content'] for chunk in top_chunks])
            
            if len(combined_text) < 100:  # Skip if too little content
                continue
                
            print(f"Analyzing: {resume_name}")
            
            # Get AI analysis
            analysis = self.analyze_resume_match(llm, job_desc, combined_text, resume_name)
            
            results.append({
                "Resume": resume_name,
                "Score": max(0, min(100, analysis.get("score", 50))),
                "Key_Skills": analysis.get("key_skills_match", "Not specified"),
                "Experience_Match": analysis.get("experience_match", "Not specified"),
                "Strengths": analysis.get("strengths", "Not specified"),
                "Gaps": analysis.get("gaps", "Not specified"),
                "Overall_Assessment": analysis.get("overall_assessment", "Not specified")
            })
        
        if results:
            # Sort by score descending
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            return df.head(k)  # Return top k results
        else:
            raise Exception("No results generated from resume analysis!")