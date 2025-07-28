from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
#model_id = "meta-llama/Llama-3.3-70B-Instruct"
model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(max_results=6, rate_limit=2.0)],
    model=model,    
)

# Now the agent can search the web!
result = agent.run("What is the current weather in Paris?")
print(result)