# Import necessary modules
import re
import time
from typing import Any, Dict, List
import os
import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from services.pdf_bot import parse_pdf, text_to_docs


@st.cache_data
def test_embed(api: str, _pages):
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(_pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index


def run_pdf_agent(api_key: str, uploaded_file: Any):
    os.environ["OPENAI_API_KEY"] = api_key
    st.title("Custom bot")
    if uploaded_file:
        name_of_file = uploaded_file.name
        doc = parse_pdf(uploaded_file)
        _pages = text_to_docs(doc)
        if _pages:
            # Allow the user to select a page and view its content
            with st.expander("Show Page Content", expanded=False):
                page_sel = st.number_input(
                    label="Select Page", min_value=1, max_value=len(_pages), step=1
                )
                _pages[page_sel - 1]
            # Allow the user to enter an OpenAI API key
            # api = "sk"

            if api_key:
                # Test the embeddings and save the index in a vector database
                index = test_embed(api_key, _pages)
                # Set up the question-answering system
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=api_key),
                    chain_type = "stuff",
                    retriever=index.as_retriever(),
                )
                # Set up the conversational agent
                tools = [
                    Tool(
                        name="QA System",
                        func=qa.run,
                        description="Input may be a partial or fully formed question.",
                    )
                ]
                prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                            Return any relevant text translated into Ukrainian.
                            You have access to a single tool:"""
                suffix = """Begin!"

                {chat_history}
                Question: {input}
                {agent_scratchpad}"""

                prompt = ZeroShotAgent.create_prompt(
                    tools,
                    prefix=prefix,
                    suffix=suffix,
                    input_variables=["input", "chat_history", "agent_scratchpad"],
                )

                if "memory" not in st.session_state:
                    st.session_state.memory = ConversationBufferMemory(
                        memory_key="chat_history"
                    )

                llm_chain = LLMChain(
                    llm=OpenAI(
                        temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo"
                    ),
                    prompt=prompt,
                )
                agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                agent_chain = AgentExecutor.from_agent_and_tools(
                    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                )

                # Allow the user to enter a query and generate a response
                query = st.text_input(
                    "**What's on your mind?**",
                    placeholder="Ask me anything from {}".format(name_of_file),
                )

                if query:
                    with st.spinner(
                        "Generating Answer to your Query : `{}` ".format(query)
                    ):
                        res = agent_chain.run(query)
                        st.info(res, icon="🤖")

                # Allow the user to view the conversation history and other information stored in the agent's memory
                with st.expander("History/Memory"):
                    st.session_state.memory
    else:
        st.write("Please upload a PDF file.")
