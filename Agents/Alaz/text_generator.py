from transformers import GPT2LMHeadModel, GPT2Tokenizer

import re

def trim_to_last_sentence(text):
    # Match end of sentence punctuation: . ! ?
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        last_end = matches[-1].end()
        return text[:last_end]
    return text  # fallback if no sentence-ending punctuation

# Step 1: Load model/tokenizer
model_path = "./custom_gpt2_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Step 2: Define generation function
def generate_text(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 3: Generate and print output
prompt = "Objects: germany, spain, ideology\nDescription:"
result = generate_text(prompt, max_new_tokens=100)
cleaned = trim_to_last_sentence (result)
print(cleaned)
