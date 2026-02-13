#!/usr/bin/env python3
"""
Helper tool to find the best capture region for your MCQ screenshots.

This will help you determine the coordinates to use for CUSTOM_REGION
in mcq_multiai.py for optimal MCQ capture.
"""

import pyautogui
import time

print("\n" + "=" * 60)
print("📐 SCREEN REGION FINDER")
print("=" * 60)

# Get screen info
screen_width, screen_height = pyautogui.size()
print(f"\n🖥️  Your screen resolution: {screen_width} x {screen_height}")

print("\n" + "-" * 60)
print("INSTRUCTIONS:")
print("-" * 60)
print("1. Open your MCQ page in the browser")
print("2. Position the MCQ question in the center of your screen")
print("3. Come back to this terminal and press ENTER")
print("-" * 60)

input("\nPress ENTER when ready...")

print("\n⏳ Taking screenshot in 3 seconds...")
print("   (Switch to your MCQ page now!)")
time.sleep(3)

# Take test screenshot
screenshot_path = "test_region.png"
img = pyautogui.screenshot()
img.save(screenshot_path)

print(f"\n✅ Screenshot saved: {screenshot_path}")

# Show different region suggestions
print("\n" + "=" * 60)
print("📊 SUGGESTED CAPTURE REGIONS")
print("=" * 60)

configs = [
    ("Fullscreen", None),
    ("Center 60%", 0.60),
    ("Center 70%", 0.70),
    ("Center 80%", 0.80),
]

for name, size in configs:
    if size is None:
        print(f"\n📌 {name}:")
        print(f"   CAPTURE_MODE = \"fullscreen\"")
    else:
        capture_width = int(screen_width * size)
        capture_height = int(screen_height * size)
        left = (screen_width - capture_width) // 2
        top = (screen_height - capture_height) // 2

        print(f"\n📌 {name}:")
        print(f"   CAPTURE_MODE = \"center\"")
        print(f"   CENTER_REGION_SIZE = {size}")
        print(f"   Region: ({left}, {top}, {capture_width}, {capture_height})")

# Custom region suggestion
print("\n" + "-" * 60)
print("🎯 FOR CUSTOM REGION:")
print("-" * 60)
print("Move your mouse to the TOP-LEFT corner of where you want")
print("the capture to start, then press ENTER...")

input("Press ENTER when mouse is in position...")
pos1 = pyautogui.position()
print(f"   Top-left corner: ({pos1.x}, {pos1.y})")

print("\nNow move your mouse to the BOTTOM-RIGHT corner")
print("of the capture area, then press ENTER...")

input("Press ENTER when mouse is in position...")
pos2 = pyautogui.position()
print(f"   Bottom-right corner: ({pos2.x}, {pos2.y})")

# Calculate custom region
custom_left = min(pos1.x, pos2.x)
custom_top = min(pos1.y, pos2.y)
custom_width = abs(pos2.x - pos1.x)
custom_height = abs(pos2.y - pos1.y)

print("\n" + "=" * 60)
print("✨ YOUR CUSTOM REGION")
print("=" * 60)
print(f"\nAdd this to mcq_multiai.py:\n")
print(f"CAPTURE_MODE = \"custom\"")
print(f"CUSTOM_REGION = ({custom_left}, {custom_top}, {custom_width}, {custom_height})")

print("\n" + "=" * 60)
print("✅ Configuration complete!")
print("=" * 60)
print(f"\nTest screenshot saved as: {screenshot_path}")
print("Copy the configuration above to your mcq_multiai.py file.\n")
