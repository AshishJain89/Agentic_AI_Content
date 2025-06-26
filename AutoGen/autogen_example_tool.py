import os
from autogen import ConversableAgent
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {'model' : model,
              'api_key' : os.getenv('OPENAI_API_KEY'),
              'temperature' : 0}

### Define simple calculation function
def add_numbers(a:Annotated[int, "First Number"] , b:Annotated[int, "Second Number"] ) -> str:
    return f"The sum of {a} and {b} is {a+b}"


def multiply_numbers(a:Annotated[int, "First Number"] , b:Annotated[int, "Second Number"] ) -> str:
    return f"The product of {a} and {b} is {a*b}"


### Create assistant agent that suggests tool calls
assistant = ConversableAgent(name = 'Calculator_Assistant' , 
                         system_message = "You are a helpful AI calculator. Return 'TERMINATE' when the task is completed",
                         llm_config = llm_config)


### Create a user proxy agent that is used for interacting with assistant agent and execute tool calls

user_proxy = ConversableAgent(name = 'User' , 
                         is_termination_msg = lambda msg: msg.get('content') is not None and 'TERMINATE' in msg['content'],
                         human_input_mode = 'NEVER',)


### Register the tool signatures with the assistant agent
assistant.register_for_llm(name = 'add_numbers' , 
                           description = 'adds two numbers')(add_numbers)

assistant.register_for_llm(name = 'multiply_numbers' , 
                           description = 'multiplies two numbers')(multiply_numbers)


### Register the tool functions with user proxy agent
user_proxy.register_for_execution(name = 'add_numbers')(add_numbers)
user_proxy.register_for_execution(name = 'multiply_numbers')(multiply_numbers)

### Invocation
user_proxy.initiate_chat(assistant, message = "What is the sum of 11 and 1?")
