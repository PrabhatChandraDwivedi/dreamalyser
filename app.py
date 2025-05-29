from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Iterator
import re
import random
from phi.agent import Agent, RunResponse
from phi.model.openai.like import OpenAILike

app = Flask(__name__)
CORS(app)

dream_agent = Agent(
    name="DreamAnalyzer",
    role="You are a dream analysis expert.",
    model=OpenAILike(
        id="deepseek-r1-distill-llama-70b",
        api_key="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",
        base_url="https://api.groq.com/openai/v1",
    ),
    instructions=[
        "You will be analyzing dreams based on 5 KPIs and returning only scores.",
        "No extra text or analysis — only the scores."
    ],
    show_tool_calls=False,
    debug_mode=False
)

# Format the LLM prompt to keep analysis structured
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

@app.route('/analyze_dream', methods=['POST'])
def analyze_dream():
    data = request.json
    dream_text = data.get('dream', '')
    name = data.get('name', 'Anonymous')

    if not dream_text:
        return jsonify({'error': 'No dream provided'}), 400

    prompt = format_dream_prompt(dream_text)

    final_response = ""
    run: Iterator[RunResponse] = dream_agent.run(prompt, stream=True)
    for response in run:
        final_response += response.content

    clean_output = re.sub(r"<think>.*?</think>", "", final_response, flags=re.DOTALL).strip()

    kpi_scores = extract_kpi_scores(clean_output)

    return jsonify({
        'name': name,
        'kpi_scores': kpi_scores
    })

@app.route('/')
def home():
    return "Dream Analyzer API is running!", 200

if __name__ == '__main__':
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=debug)

