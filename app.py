import os
import streamlit as st
import openai
from query import load_documents, setup_conversational_chain

# Load OpenAI API key securely
#load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="Geotechnical Engineering Tutor", layout="wide")
st.title("📄 Welcome, I am your Geotechnical Engineering Tutor! How may I help you?")

# Load and cache the QA chain
@st.cache_resource(show_spinner="🔄 Loading model and documents...")
def init_bot():
    documents = load_documents("data")
    return setup_conversational_chain(documents)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = init_bot()

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Ask a question:", "")
    submit = st.form_submit_button("Send")

# Handle query submission
if submit and query:
    try:
        result = st.session_state.qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        # Handle result based on output type
        answer = result.content if hasattr(result, "content") else result.get("answer", "⚠️ No answer returned.")
        st.session_state.chat_history.append((query, answer))

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for user_input, bot_reply in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**🧑‍💬 You:** {user_input}")
        st.markdown(f"**🤖 Tutor:** {bot_reply}")
        st.markdown("---")
