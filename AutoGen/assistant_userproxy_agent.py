import os
from autogen import AssistantAgent , UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {'model' : model,
              'api_key' : os.getenv('OPENAI_API_KEY')}

### Create assistant agent
assistant = AssistantAgent('assistant' , llm_config)

### Create user proxy agent agent
user_proxy = UserProxyAgent(
    name = 'user_proxy',
    llm_config=llm_config,
    code_execution_config = {
        "workd_dir" : "code_execution",
        'use_docker' : False,
    },
    human_input_mode = 'NEVER'
)


### Initiate the conversations between the agents
assistant.initiate_chat(
    user_proxy,
    message = "What is the capital of France?",
)