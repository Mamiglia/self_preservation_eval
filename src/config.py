"""Configuration and environment setup."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys for various providers (will be validated when actually running eval)
# Inspect AI will automatically use the appropriate key based on model provider
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini
HF_TOKEN = os.getenv("HF_TOKEN")  # For HuggingFace
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For OpenRouter

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "all.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
