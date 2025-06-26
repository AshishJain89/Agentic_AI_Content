import os
from autogen import ConversableAgent
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {'model' : model,
              'api_key' : os.getenv('OPENAI_API_KEY'),
              'temperature' : 0}

### Defining the tools

def get_flight_status(flight_number: Annotated[str, "Flight Number"]) -> str:
    dummy_data = {'AA123' : "On time" , 'DL456' : "Delayed" , 'UA789' : 'Cancelled'}
    return f'The current status of flight {flight_number}  is {dummy_data.get(flight_number)}'

def get_hotel_info(location: Annotated[str, 'location']) -> str:
    dummy_data = {
        'Delhi' : 'Top hotel in Delhi: The plaza - 5 star',
        'Mumbai' : 'Top hotel in Mumbai - 5 star',
        'Banglore' : 'Top hotel in Banglore - 5 star'
    }
    return dummy_data.get(location)

def get_travel_advice(location: Annotated[str, 'location']) -> str:
    dummy_data = {
        'Delhi' : 'Travel advice for Delhi',
        'Mumbai' : 'Travel advice for Mumbai',
        'Banglore' : 'Travel advice for Banglore'
    }
    return dummy_data.get(location)

### Create assistant agent that suggests tool calls
assistant = ConversableAgent(name = 'TravelAssistant' , 
                         system_message = "You are a helpful AI travel assistant. Return 'TERMINATE' when the task is completed",
                         llm_config = llm_config)


### Create a user proxy agent that is used for interacting with assistant agent and execute tool calls

user_proxy = ConversableAgent(name = 'User' , 
                         is_termination_msg = lambda msg: msg.get('content') is not None and 'TERMINATE' in msg['content'],
                         human_input_mode = 'NEVER',)



### Register the tool signatures with the assistant agent
assistant.register_for_llm(name = 'get_flight_status' , 
                           description = 'Get the current status of the flight based on the flight number')(get_flight_status)

assistant.register_for_llm(name = 'get_hotel_info' , 
                           description = 'Get information about the hotels based on specific location')(get_hotel_info)

assistant.register_for_llm(name = 'get_travel_advice' , 
                           description = 'Get travel advice for a specific location')(get_travel_advice)


### Register the tool functions with user proxy agent
user_proxy.register_for_execution(name = 'get_flight_status')(get_flight_status)
user_proxy.register_for_execution(name = 'get_hotel_info')(get_hotel_info)
user_proxy.register_for_execution(name = 'get_travel_advice')(get_travel_advice)


### Invocation
user_proxy.initiate_chat(assistant, message = "I need help with my travel plans. I am traveling to Mumbai. I need hotel information. Also give me status of flight AA123",)