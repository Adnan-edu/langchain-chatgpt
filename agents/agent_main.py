from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool

load_dotenv()

chat = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") #Expand into a new list of messages & Capture intermediate messages
    ]
)

tools = [run_query_tool]

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
    verbose=True,
    tools=tools
)

agent_executor("How many users are in the database?")