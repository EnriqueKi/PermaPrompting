import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from datetime import datetime
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import re

# === GPT-2 setup ===
model_path = "./custom_gpt2_model"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def trim_to_last_sentence(text):
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        last_end = matches[-1].end()
        return text[:last_end]
    return text

def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, top_k=30):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.2,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# === BLIP setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def describe_image(img_array):
    image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# === YOLO Setup ===
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
classes_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_SIMPLEX

def detect_and_describe(image_input, manual_prompt="", creativity=0.5):
    image = np.array(image_input)
    # Fix for webcam images (convert float32 → uint8 if needed)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    all_descriptions = []

    object_captions = []

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        confidence = confidences[i]
    
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), font, 0.6, color, 2)
    
        # Map creativity slider to generation parameters
        temperature = 0.7 + (0.5 * creativity)
        top_p = 0.85 + (0.15 * creativity)
        top_k = int(20 + (80 * creativity))

        # Ensure crop stays within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        
        cropped = image[y1:y2, x1:x2]
        
        # Skip invalid crops
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            object_captions.append(f"{label}: [crop too small for BLIP]")
            continue

        try:
            caption = describe_image(cropped)
            object_captions.append(f"{label}: {caption}")
        except Exception as e:
            object_captions.append(f"{label}: description failed")
    
    # GPT2 Prompt: combined object descriptions
    prompt = "Scene contains the following objects:\n" + "\n".join(object_captions)
    if manual_prompt.strip():
        prompt += f"\n\nAdditional context: {manual_prompt.strip()}"
    prompt += "\nContext:"
    gpt2_out = trim_to_last_sentence(generate_text(prompt, max_new_tokens=150, temperature=temperature, top_p=top_p, top_k=top_k))
    
    # Combine and return
    all_descriptions = "\n".join(object_captions) + gpt2_out
    
    return image, all_descriptions


# === Gradio UI ===
interface = gr.Interface(
    fn=detect_and_describe,
    inputs=[
        gr.Image(type="numpy", label="Upload or capture an image"),
        gr.Textbox(label="Optional additional prompt", placeholder="anything can go here", lines=2),
        gr.Slider(label="Prompt Influence / Creativity", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
    ],
    outputs=[
        gr.Image(type="numpy", label="digested image"),
        gr.Textbox(label="agent excrement", lines=10)
    ],
    title="💩pseudo-intellectual poop machine: a not-so intelligent agent💩",
    description="Upload an image (or use webcam), optionally add context, send your input into the bowels of the agent and see shit come out the other end"
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7869, share=False, inbrowser=True)
