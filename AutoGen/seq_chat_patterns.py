import os
from autogen import ConversableAgent
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {'model' : model,
              'api_key' : os.getenv('OPENAI_API_KEY'),
              'temperature' : 0}


### Initial agent always returns the given text
initial_agent = ConversableAgent(
    name = 'Initial_Agent',
    system_message = 'You return me the text I give you',
    llm_config = llm_config,
    human_input_mode = 'NEVER'
)

### The Uppercase agent that converts text to Uppercase
uppercase_agent = ConversableAgent(
    name = 'Uppercase_Agent',
    system_message = 'You convert the text I give you to uppercase',
    llm_config = llm_config,
    human_input_mode = 'NEVER'
)


### The Wordcount agent that counts the number of words in the text
word_count_agent = ConversableAgent(
    name = 'Wordcount_Agent',
    system_message = 'You count the number of words in the text I give you',
    llm_config = llm_config,
    human_input_mode = 'NEVER'
)


### The Reverse Text Agent
reverse_text_agent = ConversableAgent(
    name = 'Reversetext_Agent',
    system_message = 'You reverse the text I give you',
    llm_config = llm_config,
    human_input_mode = 'NEVER'
)

### The summarize agent
summarize_agent = ConversableAgent(
    name = 'Summarizet_Agent',
    system_message = 'You summarize the text I give you',
    llm_config = llm_config,
    human_input_mode = 'ALWAYS'
)

### Invocation of Sequential Chat


chat_results = initial_agent.initiate_chats([
    {'recipient' : uppercase_agent,
     'message' : 'This is a sample text document',
     'max_turns' : 1,
     'summary_method' : 'last_msg'},
    {'recipient' : word_count_agent,
     'message' : '',
     'max_turns' : 1,
     'summary_method' : 'last_msg'},
    {'recipient' : reverse_text_agent,
     'message' : '',
     'max_turns' : 1,
     'summary_method' : 'last_msg'},
    {'recipient' : summarize_agent,
     'message' : '',
     'max_turns' : 2,
     'summary_method' : 'last_msg'}

])

print('First Chat Summary', chat_results[0].summary)
print('First Chat Summary', chat_results[1].summary)
print('First Chat Summary', chat_results[2].summary)
print('First Chat Summary', chat_results[3].summary)




