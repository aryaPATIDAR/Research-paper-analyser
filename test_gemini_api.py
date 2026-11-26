#!/usr/bin/env python3
"""
Quick test script to verify Gemini API is working.
Run: python test_gemini_api.py
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv('backend/.env')

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ No API key found in backend/.env")
    exit(1)

print(f"✅ API key found ({len(api_key)} characters)")  # never print the key itself

try:
    # Initialize client
    client = genai.Client(api_key=api_key)
    print("✅ Client initialized successfully")
    
    # Test simple request
    print("\n🧪 Testing API with simple request...")
    response = client.models.generate_content(
        model=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
        contents='Say hello in one word',
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=10,
        )
    )
    
    if response and response.text:
        print(f"✅ API Response: {response.text}")
        print("\n✅ SUCCESS! Your Gemini API is working correctly.")
        print("\nIf you're still seeing errors in your app, check:")
        print("1. Render dashboard → Environment Variables")
        print("2. Make sure GEMINI_API_KEY is set there")
        print("3. Check Render logs for the actual error message")
    else:
        print("❌ Empty response from API")
        
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ ERROR: {error_msg}")
    
    if '429' in error_msg or 'quota' in error_msg.lower() or 'rate' in error_msg.lower():
        print("\n🚫 RATE LIMIT / QUOTA EXCEEDED")
        print("\nYour API key has hit its usage limit:")
        print("  • Free tier: 15 requests/min, 1,500 requests/day")
        print("  • Wait a few minutes and try again")
        print("  • Or upgrade at: https://ai.google.dev/pricing")
        
    elif 'api key' in error_msg.lower() or 'auth' in error_msg.lower():
        print("\n🔑 API KEY ISSUE")
        print("\nYour API key might be:")
        print("  • Invalid or expired")
        print("  • Not activated yet")
        print("  • Restricted to certain IPs")
        print("\nCheck: https://aistudio.google.com/apikey")
        
    else:
        print("\n🤔 UNKNOWN ERROR")
        print("Check the full error message above")
