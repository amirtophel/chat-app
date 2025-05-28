import os
import openai
import streamlit as st
from dotenv import load_dotenv
from query import load_documents, setup_conversational_chain

# Load OpenAI API key securely
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="Geotechnical Engineering Tutor", layout="wide")
st.title("📄 Welcome, I am your Geotechnical Engineering Tutor! How may I help you?")

# Load and cache the QA chain
@st.cache_resource(show_spinner="Loading model and documents...")
def init_bot():
    docs = load_documents("data")
    return setup_conversational_chain(docs)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = init_bot()

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask a question:", "")
    submit = st.form_submit_button("Send")

# Handle the user's query
if submit and user_query:
    try:
        result = st.session_state.qa_chain.invoke({
            "question": user_query,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        st.session_state.chat_history.append((user_query, answer))

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for q, a in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**🧑‍💬 You:** {q}")
        if "$" in a or "\\" in a:
            st.markdown(rf"""{a}""", unsafe_allow_html=False)
        else:
            st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")
