from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=model, 
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
    )

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[web_agent]
)

result = manager_agent.run("Who is the CEO of Hugging Face?")

print(result)

