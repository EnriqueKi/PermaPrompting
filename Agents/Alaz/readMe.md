# 💩 Pseudo-Intellectual Poop Machine: A Not-So Intelligent Agent 💩

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/gradio-latest-orange.svg)](https://gradio.app)
[![Transformers](https://img.shields.io/badge/🤗%20transformers-4.0+-green.svg)](https://huggingface.co/transformers)
[![Status](https://img.shields.io/badge/status-gloriously%20dysfunctional-brown.svg)](#)

> *"Upload an image, optionally add context, send your input into the bowels of the agent and see shit come out the other end"*

## What is This Abomination?

The **Pseudo-Intellectual Poop Machine** is a highly specialized AI agent that takes your **toilet photos**, "digests" them through a combination of computer vision and language models, and produces wonderfully pretentious interpretations about lavatorial spaces. It's like having a philosophy professor who specialized in bathroom studies and can't stop analyzing the ontological implications of your toilet.

**⚠️ IMPORTANT: This agent is specifically designed for and only works with toilet photographs. Upload anything else and you'll get disappointing results.**

Think of it as:
- 🤖 **YOLO** for object detection (finding toilets and toilet-adjacent objects)
- 👁️ **BLIP** for image captioning (describing what it sees in your bathroom)  
- 🚽 **Custom GPT-2** fine-tuned on toilet theory texts for generating lavatorial commentary

## Features That Nobody Asked For

- **Toilet-Specific Object Detection**: Uses YOLOv3 to find toilets, bidets, toilet paper, and other lavatorial objects with suspicious accuracy
- **Bathroom Image Captioning**: BLIP describes each detected toilet element like it's writing alt-text for a philosophy journal
- **Lavatorial Literary Analysis**: Custom-trained GPT-2 model generates impressively meaningless toilet theory commentary
- **Scatological Creativity Control**: Slider to adjust how much the AI should theorize about your toilet
- **Real-time Toilet Processing**: Watch your bathroom photos get digested in real-time through the web interface
- **Webcam Toilet Support**: Point your camera at a toilet and get instant philosophical commentary

## Installation (The Boring Part)

### Prerequisites

- Python 3.7+ (because we're not savages)
- CUDA-capable GPU (optional, but recommended for faster pooping)
- A sense of humor (mandatory)

### Required Files

You'll need these files in your project directory:

| File | What It Does | Where to Get It |
|------|-------------|----------------|
| `yolov3.weights` | Pre-trained YOLO weights | [Download (248MB)](https://pjreddie.com/media/files/yolov3.weights) |
| `yolov3.cfg` | YOLO configuration | [Download](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) |
| `coco.names` | Object class names | [Download](https://github.com/pjreddie/darknet/blob/master/data/coco.names) |
### Custom GPT-2 Model Training Data

The GPT-2 model has been fine-tuned on a carefully curated selection of toilet-related theoretical texts, including:

- **Slavoj Žižek's remarks on the ideology of toilets** - Psychoanalytic interpretations of lavatorial spaces
- **Texts on "Perfect Days" (2023 film)** - Wim Wenders' meditation on Tokyo public toilets and human dignity
- **Academic articles on public toilets** - Scholarly analysis of restroom sociology and spatial politics
- **Toilet god mythology and folklore** - Cultural and spiritual dimensions of bathroom spaces
- **Additional toilet theory corpus** - Various philosophical, anthropological, and cultural texts about toilets

This specialized training ensures the agent can produce authentically pretentious commentary specifically about toilets, bidets, bathroom fixtures, and the existential implications of waste management.

### Installation Steps

1. **Clone or download the script**
   ```bash
   # Download the magnificent poop_machine.py file
   ```

2. **Install dependencies**
   ```bash
   pip install gradio opencv-python numpy pillow torch transformers
   ```

3. **Download YOLO files** (see table above)

4. **Prepare your toilet-theory GPT-2 model**
   - The model should be fine-tuned on toilet-related philosophical and cultural texts
   - Place the trained model in `./custom_gpt2_model/` directory
   - Alternatively, modify the code to use a standard GPT-2 model (results will be less toilet-specific)

5. **Run the machine**
   ```bash
   python "pseudo-intellectual poop machine.py"
   ```

6. **Open your browser** to `http://127.0.0.1:7869`

## Usage (The Fun Part)

### Basic Operation

1. **Upload a toilet photo** or use your webcam to capture bathroom fixtures
2. **Add optional context** in the text box (e.g., "this is a Japanese toilet in a hotel" or "public restroom in Berlin")
3. **Adjust creativity slider** (0 = dry academic analysis, 1 = completely unhinged toilet theory)
4. **Click Submit** and watch the lavatorial magic happen
5. **Enjoy the results**: 
   - Your toilet image with bounding boxes around detected bathroom objects
   - Gloriously pretentious commentary mixing toilet detection with philosophical bathroom discourse

**⚠️ Remember**: Only toilet photos will produce meaningful results. The agent is specifically trained for lavatorial analysis!

### Understanding the Output

**"Digested Image"**: Your original toilet photo with colorful boxes around detected bathroom objects (toilets, sinks, toilet paper, etc.) and confidence scores.

**"Agent Excrement"**: A combination of:
- Toilet object descriptions from BLIP ("toilet: a white ceramic toilet in a bathroom")  
- Toilet-theory commentary from GPT-2 ("The porcelain vessel stands as a liminal threshold between the private and the public, embodying Žižek's notion of ideological superstructure...")

### Pro Tips

- **Higher creativity** = more unhinged toilet philosophy
- **Lower creativity** = more grounded (but still pretentious) bathroom analysis
- **Add context** about the toilet's location, culture, or significance
- **Try different toilet types** - public restrooms, Japanese toilets, bidets, squat toilets!
- **Historic toilets work great** - the agent loves analyzing vintage bathroom fixtures

## How It Works (The Technical Bits)

```
Your Toilet Photo → YOLO Detection → BLIP Captioning → GPT-2 Toilet Theory → Pretentious Output
        ↓                ↓                ↓                    ↓                     ↓
    [Upload]      [Find Toilets]    [Describe Each]    [Generate Toilet BS]   [Present Results]
```

1. **YOLO** scans your toilet photo and draws boxes around detected bathroom objects
2. **BLIP** takes crops of each detected toilet element and generates captions
3. **Custom GPT-2** (trained on toilet theory) reads all the descriptions and generates lavatorial commentary
4. **Gradio** presents everything in a nice web interface optimized for toilet analysis

## Customization Options

### Model Parameters

The code includes several tweakable parameters:

```python
confidence_threshold = 0.5    # How confident YOLO should be
temperature = 0.7-1.2        # GPT-2 randomness (controlled by slider)
max_new_tokens = 150         # Length of generated text
```

### Custom GPT-2 Toilet Model

To train your own toilet-theory model:
1. Collect a dataset of toilet-related texts (philosophy, cultural studies, anthropology)
2. Include works like Žižek's toilet ideology, "Perfect Days" analysis, toilet folklore, etc.
3. Fine-tune GPT-2 on this lavatorial corpus
4. Save the model in `./custom_gpt2_model/`

**Recommended training sources:**
- Academic papers on bathroom sociology
- Cultural anthropology texts about toilet practices
- Film criticism of toilet-themed movies
- Philosophical texts on waste and privacy
- Toilet mythology and folklore

## Troubleshooting (When Things Go Wrong)

**Common Issues:**

- **"Custom GPT-2 model not found"**: Make sure your model is in the right directory
- **"YOLO files missing"**: Download the files from the links above  
- **"Out of memory"**: Try running on CPU or reduce batch sizes
- **"Results too coherent"**: Increase the creativity slider, or try a more unusual toilet
- **"Results completely insane"**: Decrease the creativity slider, or use a more conventional toilet photo
- **"No toilet detected"**: Make sure your image actually contains a toilet - this agent is toilet-specific!
- **"Boring toilet analysis"**: Try adding context about the cultural significance of your toilet

## Contributing to the Chaos

Want to make this even more ridiculous? Contributions welcome!

Ideas for improvements:
- Add more toilet-specific vocabulary and theory to the GPT-2 training
- Integrate with other vision models for better toilet detection
- Add audio output (text-to-speech with a posh accent reading toilet theory)
- Create specialized models (Japanese toilet expert, public restroom critic, bidet philosopher, etc.)
- Expand to other bathroom fixtures (showers, bidets, sinks)

## License & Disclaimer

This project is open-source and provided "as-is" for entertainment and educational purposes. 

**Warning**: May cause excessive toilet theorizing, spontaneous bathroom criticism, and uncontrollable urges to analyze the semiotics of your toilet. Use responsibly.

**Not recommended for**: Serious toilet research, professional bathroom analysis, or situations where accurate toilet detection is actually important.

## Acknowledgments

- **YOLO** by Joseph Redmon (for making object detection actually work)
- **BLIP** by Salesforce (for teaching machines to see toilets)
- **GPT-2** by OpenAI (for the gift of eloquent toilet nonsense)
- **Gradio** team (for making AI interfaces not suck)
- **Slavoj Žižek** for his profound insights into toilet ideology
- **Wim Wenders** for "Perfect Days" and its meditation on Tokyo toilets

---

*"In the grand theater of lavatorial space, we find ourselves not merely users, but active participants in the dialectical dance between human necessity and porcelain mediation."* - The Pseudo-Intellectual Poop Machine, analyzing your toilet

**Built with 💩, toilet theory, and a questionable sense of humor**
