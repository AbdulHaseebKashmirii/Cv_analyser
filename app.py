import streamlit as st
import PyPDF2
from openai import OpenAI

nvidia_api_key = "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"
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

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# User question input
user_question = st.text_area("Ask a question about your CV:", placeholder="Type your question here...")
if st.button("Analyse Resume"):
    with st.spinner("Generating response..."):
        try:
            completion = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant. Use the provided CV text to answer user questions."},
                    {"role": "system", "content": cv_text},
                    {"role": "user",
                     "content": f"Analyse the mistakes in the CV but make sure to give suggestion but dont give long repsonses"}
                ],
                temperature=0.5,
                top_p=1,
                max_tokens=500
            )

            response = completion.choices[0].message.content
            st.subheader("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
if st.button("Get Question"):
    with st.spinner("Generating response..."):
        try:
            completion = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant. Use the provided CV text to answer user questions."},
                    {"role": "system", "content": cv_text},
                    {"role": "user",
                     "content": f"Generate 8 questions that can be asked related to this Resume"}
                ],
                temperature=0.5,
                top_p=1,
                max_tokens=500
            )

            response = completion.choices[0].message.content
            st.subheader("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
if st.button("Get Answer"):
    if cv_text.strip() == "":
        st.warning("Please upload a CV first.")
    elif user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            try:
                completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-70b-instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are an AI assistant. Use the provided CV text to answer user questions."},
                        {"role": "system", "content": cv_text},
                        {"role": "user", "content":f" Answer the QUestion but make sure to ansswer in 100 words max:\n  {user_question}"}
                    ],
                    temperature=0.5,
                    top_p=1,
                    max_tokens=500
                )

                response = completion.choices[0].message.content
                st.subheader("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
