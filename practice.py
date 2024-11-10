import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config("Python Assistant", page_icon="🐍")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Python Programming coding assistant 🐍")
st.write("Hi, I am here to help you with your Python programming questions and tasks")
st.sidebar.title("GROQ LLM")
st.sidebar.write("Enter Your Groq API Key")
key = st.sidebar.text_input("Enter Your API Key", type="password")

# Only initialize the LLM if the key is provided
if key:
    llm = ChatGroq(
        groq_api_key=key,
        model_name="llama-3.2-3b-preview"
    )
else:
    llm = None  # Set llm to None if key is not provided

def get_response(query, chat_history):
    template = """
    You are a helpful Python coding assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {Input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "Input": query
    })

user = st.chat_input("Enter Your Message")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

if user is not None and user != "":
    st.session_state.chat_history.append(HumanMessage(user))
    with st.chat_message("Human"):
        st.markdown(user)

    # Check if the API key is provided before calling the LLM
    if llm:
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user, st.session_state.chat_history))
            st.markdown(ai_response)
        st.session_state.chat_history.append(AIMessage(ai_response))
    else:
        st.warning("Please enter your API key to use the assistant.")
   