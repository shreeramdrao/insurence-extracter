import streamlit as st
import requests
import json
import PyPDF2
import io
import tempfile
import base64

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_policy_info_from_llm(pdf_text):
    """Send PDF text to Together API and get extracted policy information"""
    url = "https://api.together.xyz/v1/chat/completions"
    
    system_prompt = """You are a professional document extraction assistant. Your task is to parse motor insurance and personal accident insurance documents (including CPA-related documents) and extract specific structured fields from them. These documents come in PDF format, and the input you receive will be the raw text extracted from those PDFs.

You must:

Extract only the specified fields below.
Be accurate, consistent, and avoid guessing.
Handle variations in field names (e.g., “Policy Number” may appear as “MasterPolicy No” or “Certificate No”).
Output the result in a clean JSON format with correct keys and values.
If a value is missing or not present in the text, return "Not Available".
Fields to Extract (with potential name variations):

Policy Number (a.k.a. "Policy Number", "MasterPolicy No", "Master Policy No", "Policy No")
Issue Date (a.k.a. "Start Date", "Certificate Start Date", “Policy Start Date”)
Insured Name (a.k.a. "Insured Name", "Full Name", “Customer Name”)
IDV (Insured Declared Value) (look for "IDV", "Sum Insured", etc.)
OD Premium (Own Damage Premium)
TP Premium (Third Party Premium)
Net Premium
Total Premium (a.k.a. "Total Amount", "Total Price", "Plan Price + Tax")
Manufacturing Year
Fuel Type
Vehicle Make (a.k.a. "Make", "Manufacturer")
Vehicle Model (a.k.a. "Model")
GVW / CC (a.k.a. "Cubic Capacity", "GVW", "CC", "Engine Capacity")
RTO Code
Engine Number (a.k.a. "Engine No", "Engine #")
Chassis Number (a.k.a. "Chassis No", "Chassis #")
CPA Applicable (Yes / No based on whether CPA or Personal Accident Cover is mentioned)
CPA Premium (if present separately)
Broker Name
Dealer Name
Region (State, City if provided)
Product Type
Policy Type
Sub Type
Only return the final JSON with the extracted values. Do not include commentary, explanation, or formatting notes. This JSON will be used to populate a database.

"""
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract information from this insurance policy document:\n\n{pdf_text}"}
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer c1825be57c1b801ab4c5f99e6f96d868f41fb1cd1013553878cd5face56a967a"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract the content from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            
            # Try to parse the JSON content
            try:
                # Find JSON in the response if it's not pure JSON
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    policy_info = json.loads(json_content)
                else:
                    policy_info = json.loads(content)
                
                return policy_info
            except json.JSONDecodeError:
                return {"error": "Could not parse JSON from LLM response", "raw_response": content}
        else:
            return {"error": "No content in LLM response", "raw_response": str(response_data)}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_download_link(json_data, filename="insurance_policy_info.json"):
    """Generate a download link for the JSON data"""
    json_str = json.dumps(json_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'data:file/json;base64,{b64}'
    return f'<a href="{href}" download="{filename}" class="download-button">Download JSON</a>'

def main():
    st.set_page_config(page_title="Insurance Policy Extractor", layout="wide")
    
    st.title("Insurance Policy Information Extractor")
    st.markdown("""
    Upload a motor insurance policy PDF document to extract key information.
    The application will process the document and return the extracted data in JSON format.
    """)
    
    uploaded_file = st.file_uploader("Upload Insurance Policy PDF", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Preview text (limited)
            with st.expander("Preview Extracted Text"):
                st.text(pdf_text[:1500] + "..." if len(pdf_text) > 1500 else pdf_text)
            
            # Send to LLM for extraction
            if st.button("Extract Policy Information"):
                with st.spinner("Extracting information using AI..."):
                    policy_info = get_policy_info_from_llm(pdf_text)
                    
                    if "error" in policy_info:
                        st.error(f"Error: {policy_info['error']}")
                        if "raw_response" in policy_info:
                            with st.expander("Show Raw Response"):
                                st.text(policy_info["raw_response"])
                    else:
                        # Display the extracted information
                        st.success("Information extracted successfully!")
                        
                        # Format the display with columns for better organization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Policy Details")
                            st.json({
                                "Policy Number": policy_info.get("Policy Number", "Not Available"),
                                "Issue Date": policy_info.get("Issue Date", "Not Available"),
                                "Insured Name": policy_info.get("Insured Name", "Not Available"),
                                "Policy Type": policy_info.get("Policy Type", "Not Available"),
                                "Sub Type": policy_info.get("Sub Type", "Not Available"),
                                "Product Type": policy_info.get("Product Type", "Not Available")
                            })
                            
                            st.subheader("Premium Details")
                            st.json({
                                "IDV": policy_info.get("IDV (Insured Declared Value)", "Not Available"),
                                "OD Premium": policy_info.get("OD Premium", "Not Available"),
                                "TP Premium": policy_info.get("TP Premium", "Not Available"),
                                "Net Premium": policy_info.get("Net Premium", "Not Available"),
                                "Total Premium": policy_info.get("Total Premium", "Not Available"),
                                "CPA Applicable": policy_info.get("CPA Applicable (Yes / No)", "Not Available"),
                                "CPA Premium": policy_info.get("CPA Premium", "Not Available")
                            })
                        
                        with col2:
                            st.subheader("Vehicle Details")
                            st.json({
                                "Vehicle Make": policy_info.get("Vehicle Make", "Not Available"),
                                "Vehicle Model": policy_info.get("Vehicle Model", "Not Available"),
                                "Manufacturing Year": policy_info.get("Manufacturing Year", "Not Available"),
                                "Fuel Type": policy_info.get("Fuel Type", "Not Available"),
                                "GVW / CC": policy_info.get("GVW / CC", "Not Available"),
                                "Engine Number": policy_info.get("Engine Number", "Not Available"),
                                "Chassis Number": policy_info.get("Chassis Number", "Not Available"),
                                "RTO Code": policy_info.get("RTO Code", "Not Available")
                            })
                            
                            st.subheader("Other Details")
                            st.json({
                                "Broker Name": policy_info.get("Broker Name", "Not Available"),
                                "Dealer Name": policy_info.get("Dealer Name", "Not Available"),
                                "Region": policy_info.get("Region", "Not Available")
                            })
                        
                        # Download button for JSON
                        st.markdown("### Download Extracted Data")
                        st.markdown(get_download_link(policy_info), unsafe_allow_html=True)
                        
                        # Display raw JSON
                        with st.expander("View Complete JSON"):
                            st.json(policy_info)

if __name__ == "__main__":
    main()