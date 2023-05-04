# Додатковий імпорт для роботи з файлами
import tempfile
import streamlit as st

from csv_agent import run_csv_agent
from pdf_agent import run_pdf_agent
# from main import agent2

# Create a dropdown to select the agent
agent_selection = st.selectbox(
    "Select an agent", 
    ("Agent 1", "Agent 2")
)

if agent_selection == "Agent 1":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    run_csv_agent('', uploaded_file)
elif agent_selection == "Agent 2":
    uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])
    run_pdf_agent('', uploaded_file)