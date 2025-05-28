import os
import re
import openai
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

# Load OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load documents from the data directory
def load_documents(directory):
    documents = []
    for file in os.listdir(directory):
        try:
            filepath = os.path.join(directory, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif file.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(filepath)
            elif file.endswith('.txt'):
                loader = TextLoader(filepath)
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return documents

# Set up the LangChain conversational chain
def setup_conversational_chain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview"),
        vectordb.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True,
        verbose=False
    )
    return pdf_qa

# Initialize the global QA chain once
pdf_qa = setup_conversational_chain(load_documents("data"))
chat_history = []

# Query function used by Streamlit frontend
def query(prompt):
    global chat_history
    response = pdf_qa.invoke({
        "question": prompt,
        "chat_history": chat_history
    })

    answer = response["answer"]

    # Convert [ ... ] blocks to $$ ... $$ for LaTeX rendering
    answer = re.sub(r"\[([^\[\]]+)\]", r"$$\1$$", answer)

    # Update chat history
    chat_history.append((prompt, answer))
    if len(chat_history) > 5:
        chat_history = chat_history[-5:]

    return {"answer": answer}
