# app.py
import streamlit as st
from resume_screener import process_resumes, rank_resumes, embeddings
from langchain_community.vectorstores import Chroma
import os, glob, json
import shutil
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Smart Resume Screener", layout="wide")
st.title("🤖 Smart Resume Screener")
st.caption("Built with LangChain + Ollama + Streamlit")

job_desc = st.text_area("Paste the Job Description here", height=200)

uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF)", 
                                 accept_multiple_files=True)

if st.button("🚀 Screen Resumes") and uploaded_files and job_desc:
    with st.spinner("Processing resumes..."):
        # Save uploads
        os.makedirs("uploads", exist_ok=True)
        paths = []
        for f in uploaded_files:
            path = f"uploads/{f.name}"
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        
        vectorstore = process_resumes(paths, job_desc)
        
        with st.spinner("Ranking candidates with LLM..."):
            # Simple ranking loop
            results = []
            for path in paths:
                # Filter chunks from this resume
                resume_name = os.path.basename(path)
                docs = vectorstore.similarity_search(resume_name, k=6)
                context = "\n".join([d.page_content for d in docs])
                
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model="llama3.2:3b", temperature=0)
                
                prompt = f"""Job: {job_desc}\n\nResume: {context}\n\n
                Return ONLY valid JSON:
                {{ "candidate": "{resume_name}", "score": 88, "strengths": [], "gaps": [] }}
                """
                response = llm.invoke(prompt)
                try:
                    data = json.loads(response.content)
                    results.append(data)
                except:
                    results.append({"candidate": resume_name, "score": 0, "error": "Parse failed"})
    
    # Display results
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    st.subheader("🎯 Ranking Results")
    for r in results:
        score = r.get("score", 0)
        col1, col2, col3 = st.columns([2,1,3])
        with col1:
            st.write(f"**{r['candidate']}**")
        with col2:
            st.metric("Match %", f"{score}%")
        with col3:
            if "strengths" in r:
                st.success(" | ".join(r["strengths"][:2]))
            if "gaps" in r:
                st.warning(" | ".join(r["gaps"][:2]))
        st.divider()

