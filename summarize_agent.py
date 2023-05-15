from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

def run_summarize_agent(api_key: str, text: str):
    # Split the source text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)

    # Create Document objects for the texts
    docs = [Document(page_content=t) for t in texts[:3]]

    # Initialize the OpenAI module, load and run the summarize chain
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    prompt_template = """Write a concise summary of the following:
                        {text}
                    CONCISE SUMMARY IN UKRAINIAN:"""
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)

    return summary