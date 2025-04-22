import os
import shutil
import json
import logging
import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
import base64
import pandas as pd
from datetime import datetime
import re

# Constants
TEMP_DIR = "temp_pdfs"
VECTOR_DB_PATH = "faiss_index"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check if the config file exists, if not, use environment variables
try:
    import config
    TOGETHER_API_KEY = getattr(config, "TOGETHER_API_KEY", os.environ.get("TOGETHER_API_KEY"))
except ImportError:
    TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    st.error("❌ Together AI API Key is missing! Please check your config file or set the environment variable.")

# Insurance companies list from template
INSURANCE_COMPANIES = [
    "Reliance", "Liberty", "Future General", "TATA AIG", "United India", "New India",
    "Magma", "Royal Sundaram", "Bajaj", "Universal", "Chola", "ICICI", "Digit",
    "Iffco", "National", "Oriental", "Raheja", "SBI", "Shriram"
]

# Functions for PDF processing
def extract_text_from_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logging.error(f"⚠️ Error extracting text from {pdf_path}: {str(e)}")
        return []

def chat_with_qwen(prompt):
    try:
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant specializing in extracting structured data from insurance documents. Extract ONLY the exact value without any explanations or additional text. If you cannot find the value, respond with 'NA'."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
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

def clean_extracted_value(value_text):
    """Clean up extracted values to just return the actual value without explanations"""
    if not value_text or value_text.lower() == "na":
        return "NA"
    
    # Try to extract numbers, dates, or short text values
    # Look for policy numbers (usually numeric with possible hyphens)
    policy_match = re.search(r'(\d+[-]?\d+)', value_text)
    
    # Look for currency values
    money_match = re.search(r'(\d+,?\d*\.?\d*)', value_text)
    
    # Look for dates in various formats
    date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', value_text)
    
    # Look for text within quotes or asterisks
    quoted_match = re.search(r'["\*]([^"\*]+)["\*]', value_text)
    
    # Try to extract just the first line which often contains the answer
    first_line = value_text.split('\n')[0].strip()
    
    # If the response is just a straightforward value, return it
    if len(value_text.strip()) < 30 and not value_text.startswith("Based on") and not "document" in value_text.lower():
        return value_text.strip()
    
    # Return the first match found
    if quoted_match:
        return quoted_match.group(1).strip()
    elif policy_match:
        return policy_match.group(1).strip()
    elif money_match:
        return money_match.group(1).strip()
    elif date_match:
        return date_match.group(1).strip()
    elif len(first_line) < 50:
        return first_line
        
    # Default case - just return "NA" if we couldn't extract a clean value
    return "NA"

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
            final_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nProvide ONLY the exact value as your answer without any explanations. If the value is not found, respond with 'NA'."
            return chat_with_qwen(final_prompt)
        
        return qa_chain, vector_db
    except Exception as e:
        logging.error(f"❌ Error processing PDFs: {str(e)}")
        return None

def generate_json_structure(qa_function, uploaded_files):
    st.info("🔍 Analyzing PDF content and generating JSON structure...")
    
    # Extract policy information from each PDF
    policies = []
    for index, file in enumerate(uploaded_files):
        try:
            # Determine if this is a two-wheeler or GCV policy
            policy_type_query = f"Based on the content of '{file.name}', is this a Two Wheeler policy or a GCV (Goods Carrier Vehicle) policy?"
            policy_type_response = qa_function(policy_type_query)
            
            if "two wheeler" in policy_type_response.lower() or "2w" in policy_type_response.lower():
                policy_category = "2W DATA"
                # Extract 2W specific fields
                fields_to_extract = [
                    "Policy Number", "INSURED NAME", "Issue Date", "IDV", "OD Premium", 
                    "TP Premium", "Nett Premium", "Total Premium", "Mfr Year", "Fuel", 
                    "MAKE", "Model", "GVW / CC", "RTOCode", "Engine No", "Chassis No",
                    "CPA Premium", "CPA Policy No"
                ]
            else:
                policy_category = "GCV DATA"
                # Extract GCV specific fields
                fields_to_extract = [
                    "Policy Number", "INSURED NAME", "Issue Date", "IDV", "OD Premium", 
                    "TP Premium", "Nett Premium", "Total Premium", "Mfr Year", "Fuel", 
                    "MAKE", "Model", "GVW / CC", "RTOCode", "Engine No", "Chassis No"
                ]
            
            policy_data = {
                "S No": index + 1,
                "Section": "Corporate",  # Default value, can be updated based on content
                "Month": datetime.now().strftime("%b"),
                "Category": policy_category,
                "Filename": file.name  # Add filename for reference
            }
            
            # Extract each field
            for field in fields_to_extract:
                query = f"What is the {field} in the policy document '{file.name}'? Extract only the specific value, nothing else."
                response = qa_function(query)
                clean_value = clean_extracted_value(response)
                policy_data[field] = clean_value
            
            # Add some default fields
            policy_data["Payment Mode"] = "Online"  # Default
            policy_data["Status"] = "Active"  # Default
            
            # Determine product type based on extracted data
            product_type_query = f"Based on the data extracted, what is the Product Type of this policy? Choose from: 'Two Wheeler - New', 'Two Wheeler - Renewal', etc. Just provide the exact product type without any explanation."
            product_type = qa_function(product_type_query)
            policy_data["PRODUCT - Type"] = clean_extracted_value(product_type)
            
            # Get insurance company
            insurance_company_query = f"What insurance company issued this policy? Provide just the company name."
            insurance_company = qa_function(insurance_company_query)
            policy_data["Policy Company"] = clean_extracted_value(insurance_company)
            
            # Calculate OD Discount (often needs manual input, but we can try)
            od_discount_query = f"What is the OD Discount percentage mentioned in the policy? Extract only the numeric value."
            od_discount = qa_function(od_discount_query)
            policy_data["OD Discount"] = clean_extracted_value(od_discount)
            
            policies.append(policy_data)
            
        except Exception as e:
            st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
    
    # Create the final JSON structure
    json_structure = {
        "policies": policies,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_policies": len(policies),
            "processed_files": [file.name for file in uploaded_files]
        }
    }
    
    return json_structure

def extract_company_details(policies):
    """Extract company-specific details from policies"""
    company_details = {}
    
    # Initialize with all insurance companies from our list
    for company in INSURANCE_COMPANIES:
        company_details[company] = []
    
    # Add "Other" category for companies not in our list
    company_details["Other"] = []
    
    # Group policies by company
    for policy in policies:
        company = policy.get("Policy Company", "").strip()
        if not company or company == "NA":
            company = "Unknown"
        
        # Check if this company is in our known list
        found = False
        for known_company in INSURANCE_COMPANIES:
            if known_company.lower() in company.lower():
                company_details[known_company].append(policy)
                found = True
                break
        
        # If not found in our list, add to "Other"
        if not found:
            company_details["Other"].append(policy)
    
    # Calculate statistics for each company
    company_stats = {}
    for company, policies in company_details.items():
        if not policies:
            continue
            
        total_net_premium = 0
        total_od_premium = 0
        total_tp_premium = 0
        policy_count = len(policies)
        
        for policy in policies:
            try:
                net_premium = float(policy.get("Nett Premium", "0").replace(",", ""))
                od_premium = float(policy.get("OD Premium", "0").replace(",", ""))
                tp_premium = float(policy.get("TP Premium", "0").replace(",", ""))
                
                total_net_premium += net_premium
                total_od_premium += od_premium
                total_tp_premium += tp_premium
            except:
                pass
        
        company_stats[company] = {
            "policy_count": policy_count,
            "total_net_premium": total_net_premium,
            "total_od_premium": total_od_premium,
            "total_tp_premium": total_tp_premium,
            "avg_net_premium": total_net_premium / policy_count if policy_count > 0 else 0
        }
    
    return {
        "company_policies": company_details,
        "company_stats": company_stats
    }

def get_download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    else:
        b64 = base64.b64encode(json.dumps(object_to_download, indent=4).encode()).decode()
        return f'<a href="data:file/json;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance PDF to JSON Converter", page_icon="📄", layout="wide")
    
    st.title("Insurance PDF to JSON Converter")
    st.markdown("""
    This application allows you to upload insurance policy PDFs and convert them to a structured JSON format.
    Upload your policy documents, and the system will extract relevant information based on the Greenwin Online MIS template.
    """)
    
    # API Key input (for security, use this only if not set in config or environment)
    if not TOGETHER_API_KEY:
        api_key = st.text_input("Enter Together AI API Key:", type="password")
        if api_key:
            os.environ["TOGETHER_API_KEY"] = api_key
    
    # File upload section
    st.header("1. Upload Policy PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} files")
        
        # Process button
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs and building search index..."):
                qa_function_result = process_pdfs(uploaded_files)
                
                if qa_function_result:
                    qa_function, vector_db = qa_function_result
                    st.session_state['qa_function'] = qa_function
                    st.session_state['vector_db'] = vector_db
                    st.session_state['uploaded_files'] = uploaded_files
                    st.success("✅ PDFs processed successfully! You can now generate the JSON.")
                else:
                    st.error("❌ Failed to process PDFs")
        
        # Generate JSON button (only show if PDFs are processed)
        if 'qa_function' in st.session_state and 'uploaded_files' in st.session_state:
            st.header("2. Generate JSON")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Full JSON"):
                    with st.spinner("Generating JSON structure..."):
                        json_data = generate_json_structure(st.session_state['qa_function'], st.session_state['uploaded_files'])
                        st.session_state['json_data'] = json_data
                        st.success("✅ JSON generated successfully!")
                        
                        # Display JSON preview
                        st.subheader("JSON Preview")
                        st.json(json_data)
                        
                        # Provide download link
                        download_link = get_download_link(json_data, "insurance_policies.json", "📥 Download JSON")
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Also provide CSV option for tabular view
                        if json_data and 'policies' in json_data:
                            df = pd.DataFrame(json_data['policies'])
                            st.subheader("Tabular View")
                            st.dataframe(df)
                            csv_download_link = get_download_link(df, "insurance_policies.csv", "📥 Download CSV")
                            st.markdown(csv_download_link, unsafe_allow_html=True)
            
            with col2:
                if st.button("Extract Company Details"):
                    # Check if we already have JSON data, if not generate it
                    if 'json_data' not in st.session_state:
                        with st.spinner("Generating policy data first..."):
                            json_data = generate_json_structure(st.session_state['qa_function'], st.session_state['uploaded_files'])
                            st.session_state['json_data'] = json_data
                    else:
                        json_data = st.session_state['json_data']
                    
                    # Extract company details from policies
                    with st.spinner("Extracting company-specific details..."):
                        company_data = extract_company_details(json_data['policies'])
                        st.success("✅ Company details extracted successfully!")
                        
                        # Display company statistics
                        st.subheader("Company Statistics")
                        stats_df = pd.DataFrame.from_dict(company_data['company_stats'], orient='index')
                        st.dataframe(stats_df)
                        
                        # Create download links for each company's data
                        st.subheader("Download Company-Specific Data")
                        for company, policies in company_data['company_policies'].items():
                            if policies:  # Only show companies with policies
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{company}**: {len(policies)} policies")
                                with col2:
                                    if len(policies) > 0:
                                        company_json = {
                                            "policies": policies,
                                            "company": company,
                                            "statistics": company_data['company_stats'].get(company, {})
                                        }
                                        download_link = get_download_link(
                                            company_json, 
                                            f"{company.lower().replace(' ', '_')}_policies.json", 
                                            "📥 Download"
                                        )
                                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Provide download link for all company details
                        st.subheader("Download All Company Details")
                        download_link = get_download_link(company_data, "company_details.json", "📥 Download All Company Details")
                        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()