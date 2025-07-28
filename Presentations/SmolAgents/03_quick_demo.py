from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print(result)