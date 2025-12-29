import os

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DIR = os.path.join(BASE_DIR, "users")

# Create directories if not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

# Model set up
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
EMBEDDING_MODEL = "text-embedding-3-large"

# Memory set up
MAX_VECTOR_RESULTS = 3
MEMORY_CATEGORIES = [
    "personal",
    "professional",
    "Preferences",
    "important_facts"
]

# UI set up
PAGE_TITLE = "Multi-user chat with advanced memory"
PAGE_ICON = "🤖"