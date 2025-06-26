import os
from autogen import AssistantAgent , UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {'model' : model,
              'api_key' : os.getenv('OPENAI_API_KEY')}

assistant = AssistantAgent(
    name = 'assistant',
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name = 'user_proxy',
    code_execution_config = {
        'work_dir' : 'coding',
        'use_docker' : False,
    },
    human_input_mode = 'NEVER'
)


user_proxy.initiate_chat(
    assistant,
    message= "Plot a chart of META and TESLA stock price using yfinance and without using Alpha Vantage API. Download the chart in png format and save locally."
)