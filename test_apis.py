#!/usr/bin/env python3
"""
API Key Tester for MCQ Multi-AI Bot
Tests all configured API keys and shows which ones work
"""

import requests
from mcq_multiai import (
    OPENAI_API_KEY, GEMINI_API_KEYS, CLAUDE_API_KEY,
    DEEPSEEK_API_KEY, GROQ_API_KEYS, PERPLEXITY_API_KEYS,
    get_key
)

print("\n" + "=" * 60)
print("🔍 API KEY TESTER")
print("=" * 60 + "\n")

test_question = "What is 2+2? Reply with only the number."

# Test OpenAI (ChatGPT)
print("1️⃣  Testing ChatGPT (OpenAI)...")
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": test_question}],
            max_tokens=10
        )
        print(f"   ✅ ChatGPT: Working! Response: {r.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"   ❌ ChatGPT: Failed - {str(e)[:100]}")
else:
    print("   ⚠️  ChatGPT: No API key configured")

# Test Gemini
print("\n2️⃣  Testing Gemini (Google)...")
if GEMINI_API_KEYS:
    key = get_key(GEMINI_API_KEYS)
    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[test_question]
            )
            print(f"   ✅ Gemini: Working! Response: {response.text.strip()}")
        except Exception as e:
            print(f"   ❌ Gemini: Failed - {str(e)[:100]}")
    else:
        print("   ⚠️  Gemini: No valid API key in list")
else:
    print("   ⚠️  Gemini: No API keys configured")

# Test Claude
print("\n3️⃣  Testing Claude (Anthropic)...")
if CLAUDE_API_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": test_question}]
        )
        print(f"   ✅ Claude: Working! Response: {msg.content[0].text.strip()}")
    except Exception as e:
        print(f"   ❌ Claude: Failed - {str(e)[:100]}")
else:
    print("   ⚠️  Claude: No API key configured")

# Test DeepSeek
print("\n4️⃣  Testing DeepSeek...")
if DEEPSEEK_API_KEY:
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": test_question}],
                "max_tokens": 10
            },
            timeout=20
        )
        response = r.json()
        if "choices" in response:
            print(f"   ✅ DeepSeek: Working! Response: {response['choices'][0]['message']['content'].strip()}")
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error")
            print(f"   ❌ DeepSeek: Failed - {error_msg}")
    except Exception as e:
        print(f"   ❌ DeepSeek: Failed - {str(e)[:100]}")
else:
    print("   ⚠️  DeepSeek: No API key configured")

# Test Groq
print("\n5️⃣  Testing Groq...")
if GROQ_API_KEYS:
    key = get_key(GROQ_API_KEYS)
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": test_question}],
                max_tokens=10
            )
            print(f"   ✅ Groq: Working! Response: {r.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"   ❌ Groq: Failed - {str(e)[:100]}")
    else:
        print("   ⚠️  Groq: No valid API key in list")
else:
    print("   ⚠️  Groq: No API keys configured")

# Test Perplexity
print("\n6️⃣  Testing Perplexity...")
if PERPLEXITY_API_KEYS:
    key = get_key(PERPLEXITY_API_KEYS)
    if key:
        try:
            r = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar-pro",
                    "messages": [{"role": "user", "content": test_question}],
                    "max_tokens": 10
                },
                timeout=20
            )
            response = r.json()
            if "choices" in response:
                print(f"   ✅ Perplexity: Working! Response: {response['choices'][0]['message']['content'].strip()}")
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                print(f"   ❌ Perplexity: Failed - {error_msg}")
        except Exception as e:
            print(f"   ❌ Perplexity: Failed - {str(e)[:100]}")
    else:
        print("   ⚠️  Perplexity: No valid API key in list")
else:
    print("   ⚠️  Perplexity: No API keys configured")

print("\n" + "=" * 60)
print("✨ Test Complete!")
print("=" * 60 + "\n")
