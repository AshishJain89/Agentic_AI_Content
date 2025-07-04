from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

### Researcher Task

research_task = Task(description=(
    'Identify the video {topic}'
    'Get detailed information about the video from the YouTube Channel'
),
expected_output = 'A comprehensive 3 paragraph long report based on the {topic} of video content',
tools = [yt_tool],
agent = blog_researcher
)

### Writing the blog from the information obtained

write_task = Task(description=(
    'get the information from the youtube video channel on the {topic}'
),
expected_output = 'Summarize the info from the youtube channel video content on the topic {topic} and create the content for the blog',
tools = [yt_tool],
agent = blog_writer,
async_execution = False,
output_file = 'ai.md')