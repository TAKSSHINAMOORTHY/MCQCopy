import pyautogui
from pynput import keyboard, mouse
import base64
import time
import requests
import os
import re
import platform
from PIL import Image
import pytesseract
from collections import Counter

# ===================== API KEYS =====================
OPENAI_API_KEY = [
    os.getenv("OPENAI_API_KEY")
]

GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY")
]

CLAUDE_API_KEY = None

DEEPSEEK_API_KEY = [
    "sk-xxxx"
]

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY")
]

PERPLEXITY_API_KEYS = [
    os.getenv("PERPLEXITY_API_KEY")
]

# ===================== SCREENSHOT CONFIG =====================
CAPTURE_MODE = "interactive"
CENTER_REGION_SIZE = 0.6
CUSTOM_REGION = None
VERTICAL_OFFSET = 0

# ===================================================
def get_key(keys):
    return keys[0] if keys else None


def extract_option(text):
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def majority_vote(votes):
    return Counter(votes).most_common(1)[0][0] if votes else None


def safe_call(fn, *args, debug_name=""):
    try:
        return fn(*args)
    except Exception as e:
        print(f"⚠️ {debug_name} failed: {str(e)[:80]}")
        return None


# ===================== OCR =====================
def ocr_text(path):
    return pytesseract.image_to_string(
        Image.open(path),
        config="--oem 3 --psm 6"
    )


# ===================== SCREENSHOT =====================
def select_region_interactive():
    clicks = []

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            clicks.append((x, y))
            if len(clicks) == 2:
                return False

    print("Click TOP-LEFT then BOTTOM-RIGHT")
    with mouse.Listener(on_click=on_click) as l:
        l.join()

    if len(clicks) != 2:
        return None

    x1, y1 = clicks[0]
    x2, y2 = clicks[1]
    return (int(min(x1, x2)), int(min(y1, y2)),
            int(abs(x2 - x1)), int(abs(y2 - y1)))


def take_screenshot_region(filename):
    w, h = pyautogui.size()

    if CAPTURE_MODE == "interactive":
        region = select_region_interactive()
        img = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
    elif CAPTURE_MODE == "center":
        cw, ch = int(w * CENTER_REGION_SIZE), int(h * CENTER_REGION_SIZE)
        img = pyautogui.screenshot(
            region=((w - cw) // 2, (h - ch) // 2 + VERTICAL_OFFSET, cw, ch)
        )
    else:
        img = pyautogui.screenshot()

    img.save(filename)
    return filename


def take_two_screenshots():
    return (
        take_screenshot_region("test_ques.png"),
        take_screenshot_region("test_option.png")
    )


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ===================== MODELS =====================
def chatgpt(image_b64):
    from openai import OpenAI
    client = OpenAI(api_key=get_key(OPENAI_API_KEY))  # ✅ FIX
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Solve this MCQ. Reply with only A, B, C, or D."},
                {"type": "image_url",
                 "image_url": f"data:image/png;base64,{image_b64}"}
            ]
        }]
    )
    return r.choices[0].message.content


def gemini(path):
    from google import genai
    client = genai.Client(api_key=get_key(GEMINI_API_KEYS))
    with open(path, "rb") as f:
        img = client.files.upload(f, config={"mime_type": "image/png"})
    r = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=["Solve this MCQ. Reply with only A, B, C, or D.", img]
    )
    return r.text


def deepseek(text):
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {get_key(DEEPSEEK_API_KEY)}",  # ✅ FIX
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": text}]
        },
        timeout=20
    )
    return r.json()["choices"][0]["message"]["content"]


def groq(text):
    from groq import Groq
    client = Groq(api_key=get_key(GROQ_API_KEYS))
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": text}]
    )
    return r.choices[0].message.content


def perplexity(text):
    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {get_key(PERPLEXITY_API_KEYS)}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": text}]
        },
        timeout=20
    )
    return r.json()["choices"][0]["message"]["content"]


# ===================== HOTKEY =====================
pressed_keys = set()

def handle_capture():
    pressed_keys.clear()
    ques_path, _ = take_two_screenshots()
    text = ocr_text(ques_path)
    img_b64 = image_to_base64(ques_path)

    votes = []

    def run(name, fn, arg):
        r = safe_call(fn, arg, debug_name=name)
        o = extract_option(r)
        if o:
            votes.append(o)
            print(f"{name}: {o}")

    run("ChatGPT", chatgpt, img_b64)
    run("Gemini", gemini, ques_path)
    run("DeepSeek", deepseek, text)
    run("Groq", groq, text)
    run("Perplexity", perplexity, text)

    print("\nFINAL ANSWER:", majority_vote(votes))


def on_press(key):
    try:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            pressed_keys.add("ctrl")
        elif key == keyboard.Key.shift:
            pressed_keys.add("shift")
        elif key == keyboard.Key.alt_l:
            pressed_keys.add("option")
        elif hasattr(key, "char"):
            pressed_keys.add(key.char)

        if {"ctrl", "shift", "y"} <= pressed_keys or {"ctrl", "option", "l"} <= pressed_keys:
            handle_capture()
    except:
        pass


def on_release(key):
    pressed_keys.clear()


# ===================== MAIN =====================
def run():
    os.system("cls" if os.name == "nt" else "clear")
    print("🟢 MCQ Multi-AI Bot")
    print("CTRL + SHIFT + Y (Windows)")
    print("CTRL + OPTION + L (macOS)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        l.join()


run()
