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

st.set_page_config(page_title="CSV + SQL Agent", layout="centered")
st.title("CSV + SQL LangChain Agent")

csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
db_file = st.file_uploader("Upload a SQLite .db file", type=["db", "sqlite"])
query = st.text_area("Enter your query:")
submit = st.button("Run Agent")

if submit and not csv_file and not db_file:
    st.warning("Please upload at least one file (.csv or .db) to run a query.")
elif submit and query:
    try:
        tools = []

        if csv_file:
            csv_path = "uploaded_data.csv"
            with open(csv_path, "wb") as f:
                f.write(csv_file.read())
            df = pd.read_csv(csv_path)

            chat = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=st.secrets["OPENAI_API_KEY"])

            pandas_agent = create_pandas_dataframe_agent(
                chat,
                df,
                verbose=False,
                allow_dangerous_code=True,
                # agent_type=AgentType.OPENAI_FUNCTIONS
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,

            )

            pandas_tool = Tool(
                name="Pandas_DataFrame_Tool",
                func=pandas_agent.run,
                description="Returns data summaries, column info, and statistics from the uploaded CSV file."

            )
            tools.append(pandas_tool)

        if db_file:
            db_path = "uploaded_db.db"
            with open(db_path, "wb") as f:
                f.write(db_file.read())

            engine = create_engine(f"sqlite:///{db_path}")
            sql_db = SQLDatabase(engine)
            sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=chat)
            sql_tools = sql_toolkit.get_tools()

# Update description of each SQL tool
            for tool in sql_tools:
                if tool.name == "query_sql_db":
                    tool.description = "Executes SQL queries on the uploaded SQLite database."
                elif tool.name == "list_sql_db_tables":
                    tool.description = "Lists all tables available in the uploaded SQLite database."
                elif tool.name == "info_sql_db":
                    tool.description = "Provides schema and table-level summaries from the uploaded database."

            tools.extend(sql_tools)

        if not tools:
            st.error("No tools available. Please upload at least one valid file.")
        else:
            final_agent = initialize_agent(
                tools=tools,
                llm=chat,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )

            with st.spinner("Running..."):
                response = final_agent.run({
                    "input": query,
                    "chat_history": []
                })

            st.success("Done")
            st.markdown(f"**Response:**\n\n{response}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
