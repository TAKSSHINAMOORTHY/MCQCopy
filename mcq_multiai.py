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
import tkinter as tk
from threading import Thread
import subprocess
import sys
import json

# ===================== API KEYS =====================
# Disabled - Using local Ollama instead
USE_ONLINE_APIS = False  # Set to True to enable online API models



# ===================== OLLAMA CONFIG =====================
# Set to True to enable local Ollama model
USE_OLLAMA = True
OLLAMA_MODEL = "deepseek-r1:7b"  # Your local Ollama model (7B is faster than 32B)
OLLAMA_URL = "http://localhost:11434"  # Default Ollama API endpoint

# ===================== SCREENSHOT CONFIG =====================
# Capture mode options:
# "interactive" - Click and drag to select region (like CMD+SHIFT+4)
# "fullscreen" - Captures entire screen
# "center" - Captures centered region (good for most cases)
# "custom" - Captures specific region (set coordinates below)
CAPTURE_MODE = "interactive"

# For "center" mode: Percentage of screen to capture (0.0 to 1.0)
CENTER_REGION_SIZE = 0.6  # Captures center 60% (focused on content)

# For "custom" mode: Set specific coordinates (left, top, width, height)
# Example: CUSTOM_REGION = (100, 200, 1200, 800)
# Leave as None to auto-calculate on first run
CUSTOM_REGION = None

# Vertical offset for center mode (moves capture area up/down)
# Positive = down, Negative = up. 0 = perfectly centered
# Use -50 to move focus slightly above center (good for MCQs)
VERTICAL_OFFSET = 0

# ===================================================
# ---------- Helpers ----------
def get_key(keys):
    return keys[0] if keys else None


def extract_option(text):
    if not text:
        return None
    text = text.upper()
    match = re.search(r"(?:OPTION\s*)?\(?\b([ABCD])\b\)?", text)
    return match.group(1) if match else None


def extract_question_from_ocr(text):
    """
    Extract only the question text from OCR output.
    Removes browser UI, navigation elements, and other noise.
    """
    if not text or not text.strip():
        print("   🔍 DEBUG: No OCR text for question")
        return None

    print(f"   🔍 DEBUG: Question OCR text:\n{text[:200]}...")  # Show first 200 chars

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print(f"   🔍 DEBUG: Found {len(lines)} non-empty lines")

    cleaned_lines = []

    # Remove browser UI noise
    noise_patterns = [
        r'^(Back|Forward|Reload|Home|New Tab)$',
        r'^(https?://|www\.).*',
        r'^Question\s*\d+\s*$',
        r'^\d+\s*of\s*\d+$',
        r'^(Previous|Next|Submit|Continue)$',
    ]

    for line in lines:
        is_noise = False
        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_noise = True
                break

        # Remove lines with too many special characters
        if len(re.findall(r'[a-zA-Z0-9]', line)) < len(line) * 0.3:
            is_noise = True

        if not is_noise:
            cleaned_lines.append(line)

    print(f"   🔍 DEBUG: After cleaning: {len(cleaned_lines)} lines for question")

    if not cleaned_lines:
        return None

    # Join all lines as the question
    question = ' '.join(cleaned_lines)
    question = ' '.join(question.split())  # Normalize whitespace

    print(f"   ✅ Extracted question: {question[:100]}...")
    return question if question else None


def extract_options_from_ocr(text):
    """
    Extract the 4 options (A, B, C, D) from OCR output.
    Returns dict with keys 'A', 'B', 'C', 'D' or None if parsing failed.
    """
    if not text or not text.strip():
        print("   🔍 DEBUG: No OCR text for options")
        return None

    print(f"   🔍 DEBUG: Options OCR text:\n{text[:200]}...")  # Show first 200 chars

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print(f"   🔍 DEBUG: Found {len(lines)} non-empty lines")

    cleaned_lines = []

    # Remove browser UI noise (but be less aggressive)
    noise_patterns = [
        r'^(Back|Forward|Reload|Home|New Tab)$',
        r'^(https?://|www\.).*',
        r'^Question\s*\d+\s*$',
        r'^\d+\s*of\s*\d+$',
        r'^(Previous|Next|Submit|Continue)$',
    ]

    for line in lines:
        is_noise = False
        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_noise = True
                break

        # Don't skip short lines if they could be option markers
        # Only skip truly empty or single-char non-letter lines
        if len(line) < 1 or (len(line) == 1 and not line.isalpha()):
            is_noise = True

        if not is_noise:
            cleaned_lines.append(line)

    print(f"   🔍 DEBUG: After cleaning: {len(cleaned_lines)} lines")
    if cleaned_lines:
        print(f"   🔍 DEBUG: First few lines: {cleaned_lines[:6]}")

    if not cleaned_lines:
        return None

    # Try to find options with explicit markers (A, B, C, D with various formats)
    options = {'A': None, 'B': None, 'C': None, 'D': None}

    for line in cleaned_lines:
        # Check if line starts with an option marker
        # More flexible pattern - handles various formats including OCR errors
        option_match = re.match(r'^\(?([ABCD])\b[\)\.:,\s]*(.*)$', line, re.IGNORECASE)
        if option_match:
            option_letter = option_match.group(1).upper()
            option_text = option_match.group(2).strip()
            # Accept even if no text after marker (will use next line or just letter)
            if not option_text:
                option_text = option_letter  # Use just the letter as fallback
            options[option_letter] = option_text
            print(f"   🔍 DEBUG: Found option {option_letter}: {option_text}")

    # If no explicit markers found, try to detect lines with common OCR misreads
    # OCR often misreads A, B, C, D as: O, ©, @, etc.
    if not any(options.values()):
        print("   🔍 DEBUG: Looking for option-like lines (OCR may have misread A/B/C/D)")
        option_like_lines = []

        for line in cleaned_lines:
            # Check if line starts with common OCR mistakes or bullet points
            # Matches: O., ©, @, -, *, •, followed by text
            if re.match(r'^[O©@\-\*•●○]\s*\.?\s*(.+)$', line, re.IGNORECASE):
                text = re.sub(r'^[O©@\-\*•●○]\s*\.?\s*', '', line).strip()
                if text and len(text) > 2:  # Must have actual content
                    option_like_lines.append(text)
                    print(f"   🔍 DEBUG: Found option-like line: {text}")

        # If we found exactly 4 option-like lines, use them as A, B, C, D
        if len(option_like_lines) == 4:
            print("   ✅ Found 4 option-like lines, assigning as A/B/C/D")
            for idx, letter in enumerate(['A', 'B', 'C', 'D']):
                options[letter] = option_like_lines[idx]
        # If we found 4+ lines, take the first 4
        elif len(option_like_lines) >= 4:
            print(f"   ✅ Found {len(option_like_lines)} option-like lines, using first 4 as A/B/C/D")
            for idx, letter in enumerate(['A', 'B', 'C', 'D']):
                options[letter] = option_like_lines[idx]

    # If still no options found, try to detect 4 consecutive lines as options
    if not any(options.values()) and len(cleaned_lines) >= 4:
        print("   📝 No option markers found, using first 4 lines as A/B/C/D")
        for idx, letter in enumerate(['A', 'B', 'C', 'D']):
            if idx < len(cleaned_lines):
                options[letter] = cleaned_lines[idx]
                print(f"   🔍 DEBUG: Assigned {letter}: {cleaned_lines[idx]}")

    # Verify that we have all 4 options
    if all(options.values()):
        print(f"   ✅ Successfully extracted all 4 options")
        return options

    print(f"   ⚠️  Only found {sum(1 for v in options.values() if v)} options")
    return None


def parse_mcq_from_ocr(text):
    """
    Parse OCR text to extract question and options A, B, C, D
    Filters out browser UI noise and other irrelevant content
    """
    if not text:
        return None

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Noise patterns to skip
    noise_patterns = [
        r'^\s*@\s*',  # @ symbol at start
        r'https?://',  # URLs
        r'www\.',  # Web addresses
        r'\.com|\.org|\.net',  # Domain extensions
        r'^(File|Edit|View|History|Bookmarks|Profiles|Tab|Window|Help|Tools)',  # Browser menu
        r'^\d{1,2}:\d{2}\s*(AM|PM)',  # Timestamps
        r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\d+',  # Day+date patterns
        r'^\s*[<>|%\$#@&\*]+\s*$',  # Lines with only special chars
        r'geeksforgeeks|leetcode|hackerrank',  # Common sites (case insensitive)
        r'\.png|\.jpg|\.jpeg|\.pdf|\.ipynb',  # File extensions
        r'^Review and Finish',  # Quiz UI elements
        r'^Best QA|Hugging Face',  # Random UI text
        r'^\s*\d+\s*$',  # Lines with only numbers
        r'^[A-Z]{2,}\s*$',  # Lines with only caps (like "OCB RB")
    ]

    # Filter out noise lines
    cleaned_lines = []
    for line in lines:
        # Check if line matches any noise pattern
        is_noise = any(re.search(pattern, line, re.IGNORECASE) for pattern in noise_patterns)

        # Skip very short lines (likely UI elements)
        if len(line) < 3:
            is_noise = True

        # Skip lines that are mostly symbols
        if len(re.findall(r'[a-zA-Z0-9]', line)) < len(line) * 0.3:
            is_noise = True

        if not is_noise:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return None

    # Try to find options with explicit markers (A), B), C), D))
    options = {'A': None, 'B': None, 'C': None, 'D': None}
    question_lines = []
    option_indices = []

    option_found = False
    for i, line in enumerate(cleaned_lines):
        # Check if line contains an option marker
        option_match = re.search(r'^\(?([ABCD])[\).:]\s*(.+)$', line, re.IGNORECASE)
        if option_match:
            option_found = True
            option_letter = option_match.group(1).upper()
            option_text = option_match.group(2).strip()
            options[option_letter] = option_text
            option_indices.append(i)
        elif not option_found:
            # Before options are found, collect as question
            # Skip lines that look like headers (Question 4, Question 1, etc.)
            if not re.match(r'^Question\s*\d+$', line, re.IGNORECASE):
                question_lines.append(line)

    # If no explicit options found, check for OCR misreads (O, ©, @, etc.)
    if not option_found:
        option_like_lines = []
        question_end_idx = 0

        for i, line in enumerate(cleaned_lines):
            # Match common OCR mistakes for option markers
            if re.match(r'^[O©@\-\*•●○]\s*\.?\s*(.+)$', line, re.IGNORECASE):
                text = re.sub(r'^[O©@\-\*•●○]\s*\.?\s*', '', line).strip()
                if text and len(text) > 2:
                    option_like_lines.append((i, text))
                    if not option_found and len(option_like_lines) == 1:
                        question_end_idx = i

        # If we found 4 option-like lines, use them
        if len(option_like_lines) >= 4:
            option_found = True
            question_lines = cleaned_lines[:option_like_lines[0][0]]
            for idx, letter in enumerate(['A', 'B', 'C', 'D']):
                if idx < len(option_like_lines):
                    options[letter] = option_like_lines[idx][1]

    # If no explicit options found, try to detect 4 consecutive option-like lines
    if not option_found and len(cleaned_lines) >= 5:
        # Look for potential options: short lines (typically 1-5 words) near the end
        potential_options = []
        for i in range(len(cleaned_lines) - 1, -1, -1):
            line = cleaned_lines[i]
            word_count = len(line.split())
            # Options are typically 1-5 words
            if 1 <= word_count <= 5:
                potential_options.insert(0, (i, line))
                if len(potential_options) == 4:
                    break

        # If we found 4 potential options, use them
        if len(potential_options) == 4:
            option_start_idx = potential_options[0][0]
            # Everything before options is the question
            question_lines = cleaned_lines[:option_start_idx]
            # Assign options
            for idx, (letter, (_, opt_text)) in enumerate(zip(['A', 'B', 'C', 'D'], potential_options)):
                options[letter] = opt_text
            option_found = True

    # Build question text
    question = ' '.join(question_lines)

    # Clean up question: remove "Question N" prefixes and extra spaces
    question = re.sub(r'^Question\s*\d+\s*', '', question, flags=re.IGNORECASE)
    question = ' '.join(question.split())  # Normalize whitespace

    # Format: Question + Options
    formatted_parts = []
    if question:
        formatted_parts.append(f"Question: {question}")

    for letter in ['A', 'B', 'C', 'D']:
        if options[letter]:
            formatted_parts.append(f"{letter}. {options[letter]}")

    return '\n'.join(formatted_parts) if formatted_parts else None


def majority_vote(votes):
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def safe_call(fn, *args, debug_name=""):
    try:
        return fn(*args)
    except Exception as e:
        if debug_name:
            print(f"⚠️  {debug_name} failed: {str(e)[:80]}")
        return None


def create_overlay_window(answer_text, option_text=None):
    """
    Create a minimal overlay showing only the answer letter.
    This function is called either in a thread (Windows/Linux) or as a subprocess (macOS).
    """
    try:
        root = tk.Tk()
        root.attributes('-topmost', True)  # Always on top
        root.overrideredirect(True)  # No window decorations

        # Make the entire window transparent except the text
        root.attributes('-alpha', 1.0)
        # Use white background
        transparent_color = 'white'
        root.configure(bg=transparent_color)

        # Display only the answer letter - no background box, no padding
        if answer_text:
            answer_label = tk.Label(
                root,
                text=answer_text,  # Just the letter: A, B, C, or D
                font=('Helvetica', 48, 'bold'),  # Large 48px font
                fg='black',  # Black text
                bg=transparent_color,  # Same as window bg for transparency
                padx=0,  # No horizontal padding
                pady=0,  # No vertical padding
                borderwidth=0,  # No border
                highlightthickness=0  # No highlight border
            )
            answer_label.pack()
        else:
            answer_label = tk.Label(
                root,
                text="?",
                font=('Helvetica', 48, 'bold'),
                fg='red',
                bg=transparent_color,
                padx=0,
                pady=0,
                borderwidth=0,
                highlightthickness=0
            )
            answer_label.pack()

        # Update to get actual window size
        root.update_idletasks()

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Get window dimensions
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        # Position at bottom-left with 50px padding from edges
        x_pos = 50
        y_pos = screen_height - window_height - 50

        root.geometry(f"+{x_pos}+{y_pos}")

        # Click anywhere to close
        root.bind('<Button-1>', lambda e: root.destroy())
        answer_label.bind('<Button-1>', lambda e: root.destroy())

        # Auto-close after 10 seconds
        root.after(10000, root.destroy)

        root.mainloop()
    except Exception as e:
        # Silently handle errors to prevent crashes
        pass


def show_overlay_at_bottom_left(answer_text, options_dict=None):
    """
    Display a minimal overlay window showing only the answer letter.
    Cross-platform compatible:
    - macOS: Uses subprocess to avoid threading issues with tkinter
    - Windows/Linux: Uses threading for non-blocking overlay
    """
    if platform.system() == "Darwin":
        # macOS: Use subprocess to create overlay on its own main thread
        try:
            # Escape strings properly for Python code
            answer_escaped = json.dumps(answer_text)

            # Create a minimal overlay showing only the answer letter
            overlay_code = f'''
import tkinter as tk
try:
    answer = {answer_escaped}

    root = tk.Tk()
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.attributes('-alpha', 1.0)
    transparent_color = 'white'
    root.configure(bg=transparent_color)

    if answer:
        answer_label = tk.Label(root, text=answer, font=('Helvetica', 48, 'bold'),
                               fg='black', bg=transparent_color, padx=0, pady=0,
                               borderwidth=0, highlightthickness=0)
        answer_label.pack()
    else:
        answer_label = tk.Label(root, text="?", font=('Helvetica', 48, 'bold'),
                              fg='red', bg=transparent_color, padx=0, pady=0,
                              borderwidth=0, highlightthickness=0)
        answer_label.pack()

    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    x_pos = 50
    y_pos = screen_height - window_height - 50
'''
            # Launch as subprocess (detached, non-blocking)
            # Use start_new_session on Unix systems
            subprocess.Popen(
                [sys.executable, '-c', overlay_code],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"   ⚠️  Could not show overlay on macOS: {e}")
    else:
        # Windows/Linux: Use threading
        def create_overlay():
            try:
                create_overlay_window(answer_text, None)
            except Exception as e:
                print(f"   ⚠️  Overlay error: {e}")

        thread = Thread(target=create_overlay, daemon=True)
        thread.start()


# ---------- Screenshot ----------
def select_region_interactive():
    """Silent mouse click listener - waits for 2 clicks to define region"""
    clicks = []

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            clicks.append((x, y))
            if len(clicks) == 1:
                print(f"   ✓ Top-left: ({x}, {y})")
                print(f"   → Click BOTTOM-RIGHT corner...")
            elif len(clicks) == 2:
                print(f"   ✓ Bottom-right: ({x}, {y})")
                return False  # Stop listener

    print("\n📍 Click TOP-LEFT corner of the area...")

    # Start mouse listener
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    if len(clicks) == 2:
        # Calculate region from two clicks (convert floats to integers)
        x1, y1 = clicks[0]
        x2, y2 = clicks[1]

        left = int(min(x1, x2))
        top = int(min(y1, y2))
        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))

        if width > 10 and height > 10:  # Minimum size
            return (left, top, width, height)

    return None


def take_screenshot_region(filename):
    """
    Take a screenshot of a region defined by 2 mouse clicks.
    Returns the path to the saved image.
    """
    print(f"   👆 Click 2 corners to capture {filename}...")
    screen_width, screen_height = pyautogui.size()

    if CAPTURE_MODE == "interactive":
        # Wait for 2 mouse clicks to define region
        region = select_region_interactive()
        if region:
            left, top, width, height = region
            img = pyautogui.screenshot(region=(left, top, width, height))
        else:
            # Selection failed - use center mode as fallback
            print("   Using center 60% of screen instead...")
            capture_width = int(screen_width * 0.6)
            capture_height = int(screen_height * 0.6)
            left = (screen_width - capture_width) // 2
            top = (screen_height - capture_height) // 2
            img = pyautogui.screenshot(region=(left, top, capture_width, capture_height))

    elif CAPTURE_MODE == "custom" and CUSTOM_REGION:
        # Use custom defined region
        left, top, width, height = CUSTOM_REGION
        img = pyautogui.screenshot(region=(left, top, width, height))

    elif CAPTURE_MODE == "center":
        # Calculate centered region with optional vertical offset
        capture_width = int(screen_width * CENTER_REGION_SIZE)
        capture_height = int(screen_height * CENTER_REGION_SIZE)

        left = (screen_width - capture_width) // 2
        top = (screen_height - capture_height) // 2 + VERTICAL_OFFSET

        # Ensure we don't go off screen
        top = max(0, min(top, screen_height - capture_height))

        img = pyautogui.screenshot(region=(left, top, capture_width, capture_height))

    else:  # fullscreen
        img = pyautogui.screenshot()

    img.save(filename)
    return filename


def take_two_screenshots():
    """
    Capture two separate screenshots:
    1. Question area (test_ques.png)
    2. Options area (test_option.png)
    Returns tuple of (question_path, options_path)
    """
    print("\n📸 CAPTURE 1/2: Question Area")
    ques_path = take_screenshot_region("test_ques.png")

    print("\n📸 CAPTURE 2/2: Options Area")
    options_path = take_screenshot_region("test_option.png")

    return ques_path, options_path


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ocr_text(path):
    return pytesseract.image_to_string(
        Image.open(path),
        config="--oem 3 --psm 6"
    )


# ===================== MODELS =====================

def chatgpt(image_b64):
    from openai import OpenAI
    key = get_key(OPENAI_API_KEY)
    if not key:
        return None
    client = OpenAI(api_key=key)
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
    from google.genai import types
    key = get_key(GEMINI_API_KEYS)
    if not key:
        return None
    client = genai.Client(api_key=key)

    # Upload the image file with correct syntax
    uploaded_file = client.files.upload(path=path)

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            "Solve this MCQ. Reply with only A, B, C, or D.",
            uploaded_file
        ]
    )
    return response.text


def claude(image_b64):
    import anthropic
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    msg = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Solve this MCQ. Reply with only A, B, C, or D."},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }}
            ]
        }]
    )
    return msg.content[0].text


def deepseek(text):
    key = get_key(DEEPSEEK_API_KEY)
    if not key or key == "sk-xxxx":
        return None
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": "Solve this MCQ. Reply with only A, B, C, or D.\n\n" + text
            }]
        },
        timeout=20
    )
    response = r.json()
    if "choices" not in response:
        error_msg = response.get("error", {}).get("message", "API error")
        raise Exception(error_msg)
    return response["choices"][0]["message"]["content"]


def groq(text):
    from groq import Groq
    key = get_key(GROQ_API_KEYS)
    if not key:
        return None
    client = Groq(api_key=key)
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": "Solve this MCQ. Reply with only A, B, C, or D.\n\n" + text
        }]
    )
    return r.choices[0].message.content


def perplexity(text):
    key = get_key(PERPLEXITY_API_KEYS)
    if not key:
        return None
    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sonar-pro",
            "messages": [{
                "role": "user",
                "content": "Solve this MCQ. Reply with only A, B, C, or D.\n\n" + text
            }]
        },
        timeout=20
    )
    return r.json()["choices"][0]["message"]["content"]


def ollama(text):
    """
    Local Ollama model (no API key required)
    Uses deepseek-r1:32b or any other locally installed Ollama model
    """
    if not USE_OLLAMA:
        return None

    try:
        print(f"   → Sending request to Ollama ({OLLAMA_MODEL})...")
        print(f"   ⏳ Large model may take 1-2 minutes on first request...")
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{
                    "role": "user",
                    "content": "Solve this MCQ. Reply with only A, B, C, or D.\n\n" + text
                }],
                "stream": False
            },
            timeout=300  # 5 minutes for large models (32B takes time to load)
        )
        print(f"   ✓ Received response from Ollama")
        response = r.json()
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        return None
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"   💡 Make sure Ollama is running: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Ollama timeout - model took too long to respond")
        print(f"   💡 Try a smaller model like 'ollama run deepseek-r1:7b'")
        return None
    except Exception as e:
        print(f"   ⚠️  Ollama error: {str(e)[:80]}")
        return None


def local_bart(text):
    """
    Local BART zero-shot classification model (no API required)
    Uses facebook/bart-large-mnli for offline MCQ classification
    """
    from transformers import pipeline

    # Parse OCR text to extract only question and options
    parsed_text = parse_mcq_from_ocr(text)
    if not parsed_text:
        return None

    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Use the parsed question as the sequence to classify
    candidate_labels = ["Option A is correct", "Option B is correct", "Option C is correct", "Option D is correct"]

    result = classifier(parsed_text, candidate_labels)

    # Get the highest scoring option
    best_option = result['labels'][0]  # e.g., "Option A is correct"

    # Extract just the letter (A, B, C, or D)
    match = re.search(r'Option ([ABCD])', best_option)
    return match.group(1) if match else None


# ===================== HOTKEY HANDLER =====================

pressed_keys = set()

def handle_capture():
    # Clear pressed keys to prevent state pollution
    pressed_keys.clear()

    # Capture two separate screenshots
    ques_path, options_path = take_two_screenshots()

    # OCR both images
    question_text_raw = ocr_text(ques_path)
    options_text_raw = ocr_text(options_path)

    # Extract question and options
    question = extract_question_from_ocr(question_text_raw)
    options = extract_options_from_ocr(options_text_raw)

    votes = []

    print("\n" + "=" * 60)
    print("📸 MCQ CAPTURED\n")

    # Display extracted question
    print("❓ QUESTION:")
    print("-" * 60)
    print(question if question else "(No question detected)")
    print("-" * 60)

    # Display extracted options
    print("\n📋 OPTIONS:")
    print("-" * 60)
    if options:
        for letter in ['A', 'B', 'C', 'D']:
            print(f"{letter}. {options[letter]}")
    else:
        print("(No options detected)")
    print("-" * 60)

    # If options failed, try parsing both from the question screenshot
    if not options:
        print("\n🔄 Trying to extract both question and options from question image...")
        combined_text = ocr_text(ques_path)
        parsed_mcq = parse_mcq_from_ocr(combined_text)

        if parsed_mcq:
            print("✅ Successfully parsed combined text!")
            print(f"\n{parsed_mcq}\n")
            # Try to extract options from parsed text
            lines = parsed_mcq.split('\n')
            temp_options = {}
            for line in lines:
                match = re.match(r'^([ABCD])\.\s*(.+)$', line)
                if match:
                    temp_options[match.group(1)] = match.group(2)

            if len(temp_options) == 4:
                options = temp_options
                print("✅ Extracted options from combined parse!")

                # Update question if available
                for line in lines:
                    if line.startswith("Question:"):
                        question = line.replace("Question:", "").strip()
                        break

    # Build formatted MCQ for models
    if question and options:
        formatted_parts = [f"Question: {question}"]
        for letter in ['A', 'B', 'C', 'D']:
            formatted_parts.append(f"{letter}. {options[letter]}")
        text = '\n'.join(formatted_parts)
        parsed = text
    else:
        print("\n❌ Failed to extract question or options.")
        print("\n💡 SUGGESTIONS:")
        print("   1. Capture a LARGER area that includes the full question and all options")
        print("   2. Make sure the text is CLEAR and READABLE in the screenshot")
        print("   3. Try capturing both question AND options together in the FIRST screenshot")
        print("   4. Avoid capturing browser UI elements (address bar, tabs, etc.)")
        print("\n   Press the hotkey again to retry with a better selection.\n")
        return

    # Use the question screenshot for image-based models
    image_b64 = image_to_base64(ques_path)

    print()

    def handle(name, response):
        opt = extract_option(response)
        if opt:
            votes.append(opt)
            print(f"🤖 {name}: {opt}")
        else:
            print(f"🤖 {name}: {response if response else 'No response'}")

    # Primary: Local Ollama model (deepseek-r1:32b) - no API key needed
    print("\n🔄 Sending to Ollama model...")
    if USE_OLLAMA:
        ollama_response = safe_call(ollama, text, debug_name=f"Ollama-{OLLAMA_MODEL}")
        handle(f"Ollama-{OLLAMA_MODEL}", ollama_response)

    # Optional: Online API models (disabled by default)
    if USE_ONLINE_APIS:
        if OPENAI_API_KEY:
            handle("ChatGPT", safe_call(chatgpt, image_b64, debug_name="ChatGPT"))

        if GEMINI_API_KEYS:
            handle("Gemini", safe_call(gemini, ques_path, debug_name="Gemini"))

        if CLAUDE_API_KEY:
            handle("Claude", safe_call(claude, image_b64, debug_name="Claude"))

        if DEEPSEEK_API_KEY:
            handle("DeepSeek", safe_call(deepseek, text, debug_name="DeepSeek"))

        if GROQ_API_KEYS:
            handle("Groq", safe_call(groq, text, debug_name="Groq"))

        if PERPLEXITY_API_KEYS:
            handle("Perplexity", safe_call(perplexity, text, debug_name="Perplexity"))

    final_answer = majority_vote(votes)

    print("\n" + "=" * 60)
    if final_answer:
        if len(votes) == 1:
            print(f"✅ ANSWER: {final_answer}")
        else:
            print(f"✅ FINAL ANSWER (MAJORITY VOTE): {final_answer}")
            print(f"📊 Confidence: {votes.count(final_answer)}/{len(votes)} models agree")
    else:
        print("❌ ANSWER: Unable to determine")
        print("💡 Check if Ollama is running: ollama serve")
    print("=" * 60)

    # Show overlay at bottom-left corner automatically
    if final_answer:
        try:
            print(f"\n📍 Showing overlay at bottom-left corner with answer: {final_answer}...")
            show_overlay_at_bottom_left(final_answer, options)
            time.sleep(0.5)  # Give overlay time to appear
        except Exception as e:
            # Don't crash if overlay fails - answer is already shown in console
            print(f"   ⚠️  Overlay unavailable (answer shown above): {str(e)[:50]}")

    # Clear pressed keys after capture to ensure clean state for next run
    pressed_keys.clear()


def on_press(key):
    try:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            pressed_keys.add("ctrl")
        elif key == keyboard.Key.shift:
            pressed_keys.add("shift")
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            pressed_keys.add("option")
        elif hasattr(key, 'char') and key.char == 'l':
            pressed_keys.add("l")
        elif hasattr(key, 'char') and key.char == 'y':
            pressed_keys.add("y")

        # macOS: CONTROL + OPTION + L
        if {"ctrl", "option", "l"}.issubset(pressed_keys):
            handle_capture()
        # Windows: CTRL + SHIFT + Y
        elif {"ctrl", "shift", "y"}.issubset(pressed_keys):
            handle_capture()

    except AttributeError:
        pass


def on_release(key):
    try:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            pressed_keys.discard("ctrl")
        elif key == keyboard.Key.shift:
            pressed_keys.discard("shift")
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            pressed_keys.discard("option")
        elif hasattr(key, 'char') and key.char == 'l':
            pressed_keys.discard("l")
        elif hasattr(key, 'char') and key.char == 'y':
            pressed_keys.discard("y")
    except AttributeError:
        pass


# ===================== MAIN =====================

def run():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 60)
    print("🟢 MCQ Multi-AI Bot - Ollama Edition")
    print("=" * 60)

    # Verify tkinter is available for overlay (cross-platform check)
    try:
        test_root = tk.Tk()
        test_root.withdraw()  # Hide the window
        test_root.destroy()
        overlay_status = "✅ Overlay enabled"
    except Exception as e:
        overlay_status = "⚠️  Overlay disabled (answer will show in console only)"

    hotkey = "CONTROL + OPTION + L (macOS) / CTRL + SHIFT + Y (Windows)"
    print(f"\n➡️  Press {hotkey} to capture MCQ")
    print(f"🖥️  Platform: {platform.system()} | {overlay_status}")

    # Show capture mode
    if CAPTURE_MODE == "interactive":
        print(f"📸 Capture Mode: Interactive (2 silent clicks)")
    elif CAPTURE_MODE == "custom" and CUSTOM_REGION:
        print(f"📸 Capture Mode: Custom region {CUSTOM_REGION}")
    elif CAPTURE_MODE == "center":
        print(f"📸 Capture Mode: Center {int(CENTER_REGION_SIZE * 100)}% of screen")
    else:
        print(f"📸 Capture Mode: Fullscreen")

    # Show active model and check connection
    if USE_OLLAMA:
        print(f"\n🦙 Using Ollama Model: {OLLAMA_MODEL}")
        print(f"🌐 Ollama URL: {OLLAMA_URL}")

        # Test Ollama connection
        try:
            test_response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if test_response.status_code == 200:
                print(f"✅ Ollama is running and ready!")
            else:
                print(f"⚠️  Ollama responded but may have issues")
        except:
            print(f"❌ Cannot connect to Ollama!")
            print(f"   Run 'ollama serve' in another terminal first")
            print(f"   Or check if Ollama is installed: ollama list")

    if USE_ONLINE_APIS:
        print(f"\n⚠️  Online APIs enabled (may incur costs)")
    else:
        print(f"\n✅ Using local Ollama only (no API costs)")

    print("\n" + "=" * 60)
    print("Waiting for hotkey press...")
    print("=" * 60 + "\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


run()
