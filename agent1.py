from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct" 

model = InferenceClientModel(model_id=model_id, token="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") # You can choose to not pass any model_id to InferenceClientModel to use a default model
# you can also specify a particular provider e.g. provider="together" or provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)