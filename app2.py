import os
import shutil
import json
import logging
import requests
import fitz
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import config

# Configuration
TOGETHER_API_KEY = getattr(config, "TOGETHER_API_KEY", None)
if not TOGETHER_API_KEY:
    raise ValueError("❌ Together AI API Key is missing! Please check your config file.")

TEMP_DIR = "temp_pdfs"
VECTOR_DB_PATH = "faiss_index"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# System prompt for insurance document extraction
SYSTEM_PROMPT = """
You are an expert document parser. You will be given raw text extracted from a motor insurance policy PDF. Your task is to extract all important structured data fields from this text and return them in clean JSON format.

⸻

📝 Extract the following fields:
	•	Policy Number
	•	Issue Date
	•	Insured Name
	•	IDV (Insured Declared Value)
	•	OD Premium (Own Damage)
	•	TP Premium (Third Party)
	•	Net Premium
	•	Total Premium
	•	Manufacturing Year
	•	Fuel Type
	•	Vehicle Make
	•	Vehicle Model
	•	Cubic Capacity (CC)
	•	RTO Code
	•	Engine Number
	•	Chassis Number
	•	CPA Applicable (Yes/No)
	•	CPA Premium
	•	Broker Name
	•	Dealer Name
	•	Region
	•	Product Type (e.g., Two Wheeler - New)
	•	Policy Type (Comprehensive / Liability Only / Renewal)
	•	Sub Type (e.g., 1+5 without CPA)

⸻

Notes:
	•	If a field is not present, return "Not Available".
	•	Do not guess values. Be strict and extract only what’s clearly written.
	•	Return the result as valid JSON only, without extra explanation.

⸻
"""

def extract_text_from_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logging.error(f"⚠️ Error extracting text from {pdf_path}: {str(e)}")
        return []

def chat_with_qwen(prompt, system_prompt=None):
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful AI assistant."})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "messages": messages,
            "max_tokens": 2000,  # Increased for complex JSON responses
            "temperature": 0.2    # Lower temperature for more deterministic outputs
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        
        response = requests.post(TOGETHER_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.text}"
    except Exception as e:
        return f"❌ Exception: {str(e)}"

def process_pdfs(uploaded_files):
    os.makedirs(TEMP_DIR, exist_ok=True)
    documents = []
    
    for file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        extracted_docs = extract_text_from_pdf(temp_path)
        if not extracted_docs:
            logging.warning(f"⚠️ No extractable text found in {file.name}. Skipping.")
            continue
            
        for doc in extracted_docs:
            doc.metadata["source"] = file.name
            documents.append(doc)
            
        logging.info(f"✅ Successfully processed {file.name}!")
    
    if not documents:
        logging.error("❌ No valid PDFs processed!")
        return None
    
    try:
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
            logging.info("🗑️ Old FAISS index removed to prevent conflicts.")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"📄 Created {len(chunks)} text chunks from PDFs!")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(VECTOR_DB_PATH)
        logging.info("✅ FAISS index successfully created!")
        
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        def qa_chain(prompt):
            retrieved_docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # For document extraction, we use the context and the specialized system prompt
            extraction_prompt = f"Extract insurance document information from this text into JSON format:\n\n{context}"
            return chat_with_qwen(extraction_prompt, system_prompt=SYSTEM_PROMPT)
        
        return qa_chain
    except Exception as e:
        logging.error(f"❌ Error processing PDFs: {str(e)}")
        return None

def extract_json_from_response(response):
    """Try to extract valid JSON from the model's response"""
    try:
        # First attempt: try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            # Second attempt: look for JSON-like structure with curly braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # If both attempts fail, return None
            return None

def main():
    st.set_page_config(
        page_title="Insurance Document Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Insurance Document Information Extractor")
    st.markdown("""
    Upload your insurance PDFs to extract key information in JSON format.
    This tool works with motor insurance and personal accident insurance documents.
    """)
    
    # File uploader
    uploaded_files = st.file_uploader("Upload insurance document PDFs", 
                                     type=["pdf"], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs... This may take a minute."):
            qa_function = process_pdfs(uploaded_files)
            
        if qa_function:
            st.success("✅ PDFs processed successfully!")
            
            # Add extraction button
            if st.button("Extract Insurance Information"):
                with st.spinner("Extracting information..."):
                    # Extract information using the QA chain
                    response = qa_function("Extract all insurance document fields into JSON format")
                    
                    # Try to parse JSON from the response
                    json_data = extract_json_from_response(response)
                    
                    if json_data:
                        # Display the extracted JSON in a clean format
                        st.subheader("Extracted Information")
                        st.json(json_data)
                        
                        # Convert to JSON string for download
                        json_string = json.dumps(json_data, indent=2)
                        
                        # Create download button
                        st.download_button(
                            label="Download JSON",
                            data=json_string,
                            file_name="insurance_data.json",
                            mime="application/json"
                        )
                    else:
                        st.error("❌ Could not extract valid JSON from the response.")
                        st.text("Raw response:")
                        st.text(response)
        else:
            st.error("❌ Error processing the uploaded PDFs. Please check the logs.")

if __name__ == "__main__":
    main()