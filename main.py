import streamlit as st
import pickle
import time
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key
openai_api_key="insert your key"

# Streamlit app
st.title("GENAI_Articles_Q&A_tool")

# Input fields for URLs and question
url1 = st.text_input("Enter the first URL:")
url2 = st.text_input("Enter the second URL:")
question = st.text_area("Enter your question:")

# Process the input when the button is clicked
if st.button("Get Answer"):
    if url1 and url2 and question:
        # Load documents from the provided URLs
        loaders = UnstructuredURLLoader(urls=[url1, url2])
        data = loaders.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        # Initialize OpenAI embeddings and vector search (FAISS)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        docsearch = FAISS.from_documents(docs, embeddings)

        # Initialize the QA system with the retriever
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

        # Run the query and get the answer
        with st.spinner("Processing..."):
            answer = qa.run(question)

        # Display the answer
        st.success("Answer:")
        st.write(answer)
    else:
        st.error("Please enter both URLs and a question.")
