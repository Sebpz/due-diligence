import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI environment variables from .env
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
    raise ValueError("Azure OpenAI endpoint or API key not found in environment variables.")

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2025-01-01-preview",  # must match your deployment API version
    azure_endpoint=AZURE_OPENAI_ENDPOINT.split("/openai/deployments")[0]
)

# Example: Wrap client into a callable for CrewAI
def azure_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",  # must match your Azure deployment name
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# CrewAI Agent
researcher = Agent(
    role='Researcher',
    goal='Find the latest news about AI advancements',
    backstory='An AI agent that searches for the most recent news articles about artificial intelligence.',
    llm=azure_chat_completion
)

# CrewAI Task
task = Task(
    description='Find and summarize the top 3 latest AI news articles.',
    agent=researcher
)

# Crew
crew = Crew(
    agents=[researcher],
    tasks=[task]
)

# Run the crew
result = crew.kickoff()
print(result)

print("CrewAI implementation completed successfully.")