from smolagents import CodeAgent, InferenceClientModel
import os
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"

model = InferenceClientModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

result = agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

print(result)