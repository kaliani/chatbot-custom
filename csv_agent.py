import os
import io
import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from streamlit_chat import message


def run_csv_agent(api_key: str, uploaded_file):
    os.environ["OPENAI_API_KEY"] = api_key
    st.title("CSV Agent Interaction")

    if uploaded_file is not None:
        st.write("CSV file uploaded successfully!")

        csv_file = io.BytesIO(uploaded_file.read())
        agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

        # Initialize the chat history in the session_state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Enter your question:", key="input_field")

        if user_input:
            answer = agent.run(user_input)
            # Add the question and answer to the chat_history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("agent", answer))

        # Display the chat_history in a chat-like format using streamlit-chat
        for i, (sender, message_text) in enumerate(st.session_state.chat_history):
            if sender == "user":
                message(message_text, is_user=True, key=f"{i}_user")
            else:
                message(message_text, key=f"{i}")

    else:
        st.write("Please upload a CSV file.")
