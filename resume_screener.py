# resume_screener.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
import shutil
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")

def clear_vectorstore():
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")

def process_resumes(resume_paths, job_desc):
    docs = []
    for path in resume_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    
    # Add job description as a document too
    from langchain.schema import Document
    docs.append(Document(page_content=job_desc, metadata={"type": "job_desc"}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    clear_vectorstore()
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="vectorstore")
    return vectorstore

def rank_resumes(vectorstore, job_desc):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    template = """You are an expert HR screener.
    Job Description: {job_desc}

    Relevant resume chunks:
    {context}

    Score this candidate from 0-100% match.
    Return ONLY JSON:
    {{
      "score": 95,
      "strengths": ["bullet 1", "bullet 2"],
      "gaps": ["gap 1", "gap 2"]
    }}
    """
    prompt = PromptTemplate.from_template(template)

    from langchain.chains import RetrievalQAWithSourcesChain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    
    result = chain.invoke({"question": "Rank all candidates", "job_desc": job_desc})
    return result
