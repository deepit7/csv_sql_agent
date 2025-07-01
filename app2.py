import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

st.set_page_config(page_title="CSV or SQL Agent", layout="centered")
st.title("🔍 CSV & SQLite DB Agent")

# Initialize LLM once
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=st.secrets.get("openai_api_key")
)

# Select mode
mode = st.radio("Select data source:", ["CSV", "SQLite DB"])

# File uploader for selected mode
data_file = None
if mode == "CSV":
    data_file = st.file_uploader("Upload a CSV file", type=["csv"])
else:
    data_file = st.file_uploader("Upload a SQLite .db file", type=["db", "sqlite"])

# Query input
query = st.text_area("Enter your query:")
run = st.button("Run Agent")

if run:
    if not data_file:
        st.error(f"Please upload a {mode} file to proceed.")
    elif not query:
        st.error("Please enter a query.")
    else:
        try:
            if mode == "CSV":
                # Save and load CSV
                path = "uploaded.csv"
                with open(path, "wb") as f:
                    f.write(data_file.read())
                df = pd.read_csv(path)
                st.success(f"✅ CSV loaded: {df.shape}")

                # Create and run pandas agent
                pandas_agent = create_pandas_dataframe_agent(
                    chat,
                    df,
                    verbose=False,
                    allow_dangerous_code=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                )
                result = pandas_agent.run(query)

            else:
                # Save and load SQLite DB
                dbpath = "uploaded.db"
                with open(dbpath, "wb") as f:
                    f.write(data_file.read())
                engine = create_engine(f"sqlite:///{dbpath}")
                sql_db = SQLDatabase(engine)
                sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=chat)
                tools = sql_toolkit.get_tools()

                # Initialize and run SQL agent
                sql_agent = initialize_agent(
                    tools,
                    chat,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=True,
                    handle_parsing_errors=True
                )
                result = sql_agent.run(query)

            st.success("✅ Done")
            st.markdown(f"**Response:**\n\n{result}")
        except Exception as e:
            st.error(f"Error: {e}")
