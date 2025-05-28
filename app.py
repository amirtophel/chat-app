import os
import openai
import streamlit as st
from dotenv import load_dotenv
from query import load_documents, setup_conversational_chain


load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]
@st.cache_resource(show_spinner="Loading model and documents...")
def init_bot():
    docs = load_documents("data")
    chain = setup_conversational_chain(docs)
    return chain

# App config
st.set_page_config(page_title="Geotechnical Engineering Tutor", layout="wide")
st.title("📄 Welcome! How may I help you?")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = init_bot()

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Ask a question:", "")
    submit = st.form_submit_button("Send")

# Handle query
if submit and query:
    try:
        result = st.session_state.qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })
        answer = result["answer"]
        st.session_state.chat_history.append((query, answer))
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
        st.markdown(f"**🧑‍💬 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")
