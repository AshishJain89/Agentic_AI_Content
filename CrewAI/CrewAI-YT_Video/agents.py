from crewai import Agent
from dotenv import load_dotenv
import os
from tools import yt_tool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


### Create the Blog Researcher Agent

blog_researcher = Agent(
    role = 'Blog Researcher for YouTube Videos',
    goal = 'get the relevant video transcript for the topic {topic} from the provided Youtube Channel',
    backstory = ('Expert in understanding videos in astronomy'),
    tools = [yt_tool],
    llm = 'gpt-4',
    allow_delegation = True,
    memory = True,
    verbose = True
)

### Create the Blog Writer Agent

blog_writer = Agent(
    role = 'Blog Writer',
    goal = 'Narrate compelling stories about the video {topic} from the provided Youtube Channel',
    backstory = ('You have the flair to simplify complex topics and provide engaging narratives'),
    tools = [yt_tool],
    llm = 'gpt-4',
    allow_delegation = False,
    memory = True,
    verbose = True
)