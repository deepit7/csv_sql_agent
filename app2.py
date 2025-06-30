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
db_file  = st.file_uploader("Upload a SQLite .db file", type=["db", "sqlite"])
query    = st.text_area("Enter your query:")
submit   = st.button("Run Agent")

# 1️⃣  Create the LLM once
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=st.secrets["openai_api_key"]
)

if submit and not csv_file and not db_file:
    st.warning("Please upload at least one file (.csv or .db) to run a query.")

elif submit and query:
    try:
        tools = []

        # 2️⃣  CSV  👉  Pandas tool
        if csv_file:
            csv_path = "uploaded_data.csv"
            with open(csv_path, "wb") as f:
                f.write(csv_file.read())
            df = pd.read_csv(csv_path)
            st.success(f"✅ CSV loaded: {df.shape}")

            pandas_agent = create_pandas_dataframe_agent(
                chat,
                df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )

            tools.append(
                Tool(
                    name="pandas_dataframe_tool",
                    func=pandas_agent.run,
                    description="Answer questions about the uploaded CSV."
                )
            )

        # 3️⃣  DB  👉  SQL tools
        if db_file:
            db_path = "uploaded_db.db"
            with open(db_path, "wb") as f:
                f.write(db_file.read())

            sql_db      = SQLDatabase(create_engine(f"sqlite:///{db_path}"))
            sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=chat)
            tools.extend(sql_toolkit.get_tools())

        if not tools:
            st.error("No tools available. Please upload at least one valid file.")
            st.stop()

        # 4️⃣  Initialise multi-tool agent
        final_agent = initialize_agent(
            tools=tools,
            llm=chat,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
        )

        # 5️⃣  RUN with **plain string**
        with st.spinner("Running…"):
            response = final_agent.run(query)

        st.success("Done")
        st.markdown(f"**Response:**\n\n{response}")

    except Exception as e:
        st.error(f"Error: {e}")
