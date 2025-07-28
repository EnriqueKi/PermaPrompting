from smolagents import Tool
from smolagents import CodeAgent, InferenceClientModel

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[image_generation_tool], model=model)

result = agent.run(
    "Improve this prompt, then generate an image of it.", additional_args={'user_prompt': 'Brutalist building in the middle of Prague'}
)

print(result)
print('--end--')