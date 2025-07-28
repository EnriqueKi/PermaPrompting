from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    " Transcribe the mp3 audio in the attached link and answer why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
)