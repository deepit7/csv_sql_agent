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
st.title("🔍DeepQuery Agent ")

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

def is_response_unhelpful(response: str) -> bool:
    vague_keywords = [
        "i don't know", "insufficient", "not enough", "no relevant",
        "can't answer", "unclear", "data missing", "unable to determine",
        "just a summary", "not specified", "no data", "no result"
    ]
    return (
        len(response.strip()) < 20
        or any(word in response.lower() for word in vague_keywords)
    )

if run:
    if not data_file:
        st.error(f"Please upload a {mode} file to proceed.")
    elif not query:
        st.error("Please enter a query.")
    else:
        try:
            if mode == "CSV":
                path = "uploaded.csv"
                with open(path, "wb") as f:
                    f.write(data_file.read())
                df = pd.read_csv(path)
                st.success(f"✅ CSV loaded: {df.shape}")

                pandas_agent = create_pandas_dataframe_agent(
                    chat,
                    df,
                    verbose=False,
                    allow_dangerous_code=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                )
                result = pandas_agent.run(query)

            else:
                dbpath = "uploaded.db"
                with open(dbpath, "wb") as f:
                    f.write(data_file.read())
                engine = create_engine(f"sqlite:///{dbpath}")
                sql_db = SQLDatabase(engine)
                sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=chat)
                tools = sql_toolkit.get_tools()

                sql_agent = initialize_agent(
                    tools,
                    chat,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                    handle_parsing_errors=True
                )
                result = sql_agent.run(query)

            st.success("✅ Done")
            st.markdown(f"**Response:**\n\n{result}")

            if is_response_unhelpful(result):
                explanation_prompt = f"""
                The user asked: {query}
                The final response was: {result}

                Explain in simple terms why the response might be incomplete or vague.
                Be concise and honest — mention if the query was vague, if tools failed, or if data is insufficient.
                """
                reason = chat.predict(explanation_prompt)
                st.warning("⚠️ The response may not fully answer the query.")
                st.markdown(f"**Why:** {reason}")

        except Exception as e:
            st.error(f"Error: {e}")
