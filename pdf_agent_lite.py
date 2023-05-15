import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

def run_pdf_agent(api_key: str, uploaded_file):
    st.title("PDF Agent Interaction")

    # Get the path to the uploaded PDF file
    pdf_file_path = "uploaded_file.pdf"
    with open(pdf_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Set up document loader
    loader = UnstructuredPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    # Set up embeddings
    embeddings = OpenAIEmbeddings()

    # Set up document search and retrieve relevant documents based on query
    docsearch = Chroma.from_document(pages, embeddings).as_retriever()
    docs = docsearch.get_relevant()

    # Set up the question-answering chain
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    # Execute the query on the document chain
    result = chain.run(input_document=docs, question=query)

    # Display the result
    st.write(result)

# Example usage
api_key = "your_openai_api_key"
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    run_pdf_agent(api_key, uploaded_file)
