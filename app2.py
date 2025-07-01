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
                    return_intermediate_steps=True  # enable steps
                )
                response = pandas_agent.invoke({"input": query})
                result = response["output"]
                steps = response.get("intermediate_steps", [])

            else:
                dbpath = "uploaded.db"
                with open(dbpath, "wb") as f:
                    f.write(data_file.read())
                engine = create_engine(f"sqlite:///{dbpath}")
                sql_db = SQLDatabase(engine)
                sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=chat)
                tools = sql_toolkit.get_tools()

                sql_agent_executor = initialize_agent(
                    tools,
                    chat,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                    handle_parsing_errors=True,
                    return_intermediate_steps=True
                )
                response = sql_agent_executor.invoke({"input": query})
                result = response["output"]
                steps = response.get("intermediate_steps", [])

            st.success("✅ Done")
            st.markdown(f"**Response:**\n\n{result}")

            # ✅ Show reasoning
            if steps:
                with st.expander("🧠 Show agent reasoning steps"):
                    for i, (action, observation) in enumerate(steps, 1):
                        st.markdown(f"### 🧩 Step {i}")

                        if hasattr(action, "tool") and hasattr(action, "tool_input"):
                            tool_name = action.tool
                            tool_input = str(action.tool_input)

                            readable_tool = {
                                "python_repl_ast": "🧠 Python Code",
                                "query_sql_db": "📊 SQL Query",
                                "pandas_dataframe_tool": "📈 Pandas Operations"
                            }.get(tool_name, tool_name)

                            st.markdown(f"**Tool used:** {readable_tool}")
                            # Use LLM to convert tool_input code into plain English explanation
                            plain_prompt = f"""Explain this code in simple words for a general user:\n\n{tool_input}"""
                            explanation = chat.predict(plain_prompt)

                            st.markdown(f"**What it tried to do (explained):** {explanation}")

                        else:
                            st.markdown("Tool information not available.")

                        #st.markdown(f"**Output/Result:** `{observation}`")
                        #st.markdown("---")


            # ✅ Explanation if vague
            if is_response_unhelpful(result):
                explanation_prompt = f"""
                The user asked: {query}
                The final response was: {result}

                Explain in simple terms why the response may be incomplete or vague.
                """
                reason = chat.predict(explanation_prompt)
                st.warning("⚠️ The response may not fully answer the query.")
                st.markdown(f"**Why:** {reason}")

        except Exception as e:
            st.error(f"Error: {e}")
