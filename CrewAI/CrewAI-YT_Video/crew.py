from crewai import Crew , Process
from agents import blog_researcher,blog_writer
from tasks import research_task , write_task


### Create the Crew


crew = Crew(
    agents = [blog_researcher,blog_writer],
    tasks = [research_task,write_task],
    process = Process.sequential,        ### Sequential task execution
    memory = True,
    max_rpm = 100,
    cache = True,
    share_crew = True
)

### Crew Invocation

crew.kickoff(inputs={'topic' : 'But what is a neural network?'})