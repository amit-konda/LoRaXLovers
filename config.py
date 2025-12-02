"""
Configuration file for API keys and settings.
You can set your API keys here or use environment variables.
"""

import os

# Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDubzBJ3Qwh3U68I9SV-SW3DlzjG1k6AV0")

# Gemini Model Configuration
# Available models: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

