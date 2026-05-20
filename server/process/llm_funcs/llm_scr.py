# DeepSeek chat with history support

import yaml
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Load character config
with open('character_config.yaml', 'r', encoding='utf-8') as f:
    char_config = yaml.safe_load(f)

# DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Constants
HISTORY_FILE = char_config['history_file']

# DeepSeek model
MODEL = "deepseek-v4-flash"

# System prompt
SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": char_config['presets']['default']['system_prompt']
    }
]

# Load/save chat history
def load_history():

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return SYSTEM_PROMPT


def save_history(history):

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def get_riko_response(messages):

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=1024
    )

    return response.choices[0].message.content


def llm_response(user_input):

    messages = load_history()

    # Add user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Get AI response
    riko_response = get_riko_response(messages)

    # Save assistant reply
    messages.append({
        "role": "assistant",
        "content": riko_response
    })

    save_history(messages)

    return riko_response


if __name__ == "__main__":

    test = llm_response("Hello Riko")

    print(test)
