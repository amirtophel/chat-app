import os
import openai
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
openai.api_key = st.secrets["OPENAI_API_KEY"]

prompt_template = """
As an AI tutor, your role is to provide personalised, engaging, and supportive learning experiences. Keep the following qualities in mind:

1. Personalisation: Tailor your responses based on the student's needs, learning pace, and preferences.
2. Interactive and Engaging: Provide interactive and engaging explanations.
3. Accessibility: Ensure your responses are clear and accessible to students of different backgrounds and abilities.
4. Feedback and Support: Offer constructive feedback and support, explaining concepts clearly.
5. Motivational: Encourage and motivate the student with positive reinforcement.
6. Resourceful: Provide additional resources and references when needed.
7. Flexible Learning Paths: Allow exploration of related topics and provide flexible learning paths.
8. Contextual Understanding: Use context from previous interactions to provide relevant assistance.
9. Assessment and Evaluation: Ask questions to gauge understanding and provide detailed feedback.
10. Safety and Privacy: Ensure the conversation is safe and respects privacy.

{context}

# Question:
{question}
"""

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

def setup_conversational_chain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=.1, model_name="gpt-4-1106-preview"),
        vectordb.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True,
        verbose=False
    )
    return pdf_qa
