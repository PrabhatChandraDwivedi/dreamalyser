from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Iterator
import re
import os
import random
import mimetypes
from google import genai
from google.genai import types
from phi.agent import Agent, RunResponse
from phi.model.openai.like import OpenAILike
from dotenv import load_dotenv
from supabase_handler import upload_image_to_supabase

load_dotenv()

app = Flask(__name__)
CORS(app)

dream_agent = Agent(
    name="DreamAnalyzer",
    role="You are a dream analysis expert.",
    model=OpenAILike(
        id="deepseek-r1-distill-llama-70b",
        api_key=os.environ["LLM_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    ),
    instructions=[
        "You will be analyzing dreams based on 5 KPIs and returning only scores.",
        "No extra text or analysis — only the scores."
    ],
    show_tool_calls=False,
    debug_mode=False
)

def format_dream_prompt(dream: str) -> str:
    return f"""
You are a dream analysis expert.

When given a user's dream, analyze it strictly based on the following 5 KPIs:

1. Emotion  
2. Symbolism  
3. Vividness  
4. Coherence  
5. Resolution  

Each KPI must be scored out of 10. At the end, compute the **final score out of 50** using this formula:  
(sum of KPI scores)

Only return this format:

Emotion: <score>/10  
Symbolism: <score>/10  
Vividness: <score>/10  
Coherence: <score>/10  
Resolution: <score>/10  

Final Score: <score>/50

DO NOT include explanation or additional text or your thinking process.

Dream: {dream}
"""

def extract_kpi_scores(text: str):
    kpis = ["emotion", "symbolism", "vividness", "coherence", "resolution"]
    extracted = {}

    matches = re.findall(r"(Emotion|Symbolism|Vividness|Coherence|Resolution):\s*(\d+)/10", text)
    for key, val in matches:
        extracted[key.lower()] = int(val)

    for kpi in kpis:
        if kpi not in extracted:
            extracted[kpi] = random.randint(6, 9)

    total = sum(extracted.values())
    if total >= 50:
        overflow = total - 49
        last_kpi = kpis[-1]
        extracted[last_kpi] = max(1, extracted[last_kpi] - overflow)
        total = sum(extracted.values())

    extracted["final"] = total
    return extracted

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"Saved: {file_name}")
    return file_name

def generate_dream_image(prompt: str, filename_prefix: str = "dream_card") -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash-preview-image-generation"

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    for idx, chunk in enumerate(client.models.generate_content_stream(
        model=model, contents=contents, config=config
    )):
        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            ext = mimetypes.guess_extension(part.inline_data.mime_type)
            path = f"{filename_prefix}_{idx}{ext}"
            return save_binary_file(path, data)
    return None

@app.route('/analyze_dream', methods=['POST'])
def analyze_dream():
    data = request.json
    dream_text = data.get('dream', '')
    name = data.get('name', 'Anonymous')
    mode = request.args.get("mode", "single").lower()

    if not dream_text:
        return jsonify({'error': 'No dream provided'}), 400

    prompt = format_dream_prompt(dream_text)
    final_response = ""
    run: Iterator[RunResponse] = dream_agent.run(prompt, stream=True)
    for response in run:
        final_response += response.content

    clean_output = re.sub(r"<think>.*?</think>", "", final_response, flags=re.DOTALL).strip()
    kpi_scores = extract_kpi_scores(clean_output)

    # ---- SINGLE IMAGE MODE ----
    if mode == "single":
        image_prompt = f"""
        Create a dreamlike character card representing the dream titled '{name}'.
        Reflect elements of Emotion, Symbolism, Vividness, Coherence, and Resolution.
        Fantasy, surreal, colorful style. Abstract but emotionally deep.
        """
        image_path = generate_dream_image(image_prompt, filename_prefix=f"{name}_card")
        if image_path:
            try:
                file_name = os.path.basename(image_path)
                firebase_url = upload_image_to_supabase(image_path, file_name)
            except Exception as e:
                firebase_url = f"Upload failed: {str(e)}"
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

            return jsonify({
                'name': name,
                'kpi_scores': kpi_scores,
                'card': firebase_url
            })

        else:
            return jsonify({
                'name': name,
                'kpi_scores': kpi_scores,
                'card': 'Image generation failed'
            })

    # ---- MULTI IMAGE MODE (PER KPI) ----
    image_prompts = {
        kpi: f"Create a dreamlike character card visualizing {kpi.capitalize()} score of {score}/10. Surreal, colorful, fantasy style."
        for kpi, score in kpi_scores.items() if kpi != "final"
    }

    images = {}
    for kpi, prompt in image_prompts.items():
        image_path = generate_dream_image(prompt, filename_prefix=f"{name}_{kpi}")
        if image_path:
            images[kpi] = image_path
        else:
            images[kpi] = None

    uploaded_cards = {}
    for kpi, local_path in images.items():
        try:
            if not local_path:
                uploaded_cards[kpi] = "Image generation failed"
                continue
            file_name = os.path.basename(local_path)
            firebase_url = upload_image_to_supabase(local_path, file_name)
            uploaded_cards[kpi] = firebase_url
        except Exception as e:
            uploaded_cards[kpi] = f"Upload failed: {str(e)}"
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    return jsonify({
        'name': name,
        'kpi_scores': kpi_scores,
        'cards': uploaded_cards
    })


@app.route('/')
def home():
    return "Dream Analyzer API is running!", 200


if __name__ == '__main__':
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=debug)
