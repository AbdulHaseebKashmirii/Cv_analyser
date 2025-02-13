import streamlit as st
import PyPDF2
from openai import OpenAI

st.title("CV-Based Q&A with NVIDIA LLaMA 3.1")
st.write("Upload your one-page CV and ask questions about its content.")

# Upload CV PDF file
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

cv_text = ""
if uploaded_file is not None:
    with st.spinner("Extracting text from CV..."):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            cv_text += page.extract_text() + "\n"
    st.success("CV uploaded and processed successfully!")

# NVIDIA API Key (Set in Render Environment Variables)
nvidia_api_key = "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"

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# User question input
user_question = st.text_area("Ask a question about your CV:", placeholder="Type your question here...")

def generate_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI assistant. Use the provided CV text to answer user questions."},
                {"role": "system", "content": cv_text},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

if st.button("Analyze Resume"):
    with st.spinner("Generating response..."):
        response = generate_response("Analyze the mistakes in the CV and provide suggestions concisely.")
        st.subheader("Analysis:")
        st.write(response)

if st.button("Get Questions"):
    with st.spinner("Generating response..."):
        response = generate_response("Generate 8 questions that can be asked related to this Resume.")
        st.subheader("Questions:")
        st.write(response)

if st.button("Get Answer"):
    if cv_text.strip() == "":
        st.warning("Please upload a CV first.")
    elif user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            response = generate_response(f"Answer the question in 100 words max:\n{user_question}")
            st.subheader("Answer:")
            st.write(response)

# Deployment Instructions for Render
if __name__ == "__main__":
    st.write("Running on Render. Make sure NVIDIA_API_KEY is set in environment variables.")
