from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
                      "You are an AI that has access to a SQLite database.\n"
                      f"The database has tables of: {tables}\n"
                      "Do not make any assumptions about what tables exist "
                      "or what columns exist. Instead, use the 'describe_tables' function"
                      )),
        MessagesPlaceholder(variable_name="chat_history"), # Maintain the order - Add before human message              
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") #Expand into a new list of messages & Capture intermediate messages
    ]
)

# return_messages - True
# Return list of messages as message objects instead of string
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) 

tools = [
    run_query_tool,
    describe_tables_tool,
    write_report_tool
      
      ]

#
#Agent: A chain that knows how to use tools
# Will take that list of tools and convert them into JSON descriptions
# Still has input variables, memory, prompts, etc - all the normal things a chain has
#
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

#Agent Executor:
#Takes an agent and runs it until the response is not a function call
#Essentially a fancy while loop

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory
)

#agent_executor("How many users are in the database?")

#agent_executor("How many users have provided the shipping address?")

#agent_executor("Summarize the top 5 most popular products. Write the results to a report file.")

agent_executor("How many orders are there? Write the result to an html report file.")

agent_executor("Report the exact same process for users.")