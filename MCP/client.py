from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client = MultiServerMCPClient(
    {
    'math': {
        'command' : 'python',
        'args' : ['mathserver.py'],
        'transport' : 'stdio',
    },
    'weather': {
        'url': 'http://127.0.0.1:8000/mcp',
        'transport' : 'streamable_http',
    }})

    import os
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

    tools = await client.get_tools()
    model = ChatGroq(model='qwen-qwq-32b')
    agent = create_react_agent(model,tools)
    
    
    ###----------Math related question
    
    math_response = await agent.ainvoke(
        {'messages' : [{'role' : 'user' , 'content' : 'what is (4+7) x 13?'}]}
        )
    
    print('math response', math_response['messages'])

    ###----------Weather related question

    weather_response = await agent.ainvoke(
        {'messages' : [{'role' : 'user' , 'content' : 'what is weather in delhi'}]}
    )
    print('weather response', weather_response['messages'])


asyncio.run(main())