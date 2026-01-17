"""
Auxiliary utilities for the application
"""
from datetime import datetime
import re

def format_timestamp(timestamp_str: str) -> str:
    """Format a timestamp to display in the interface"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return timestamp_str

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to display in the interface"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def validate_user_id(user_id: str) -> bool:
    """Verify that the user ID is valid"""
    # Letters, numbers, hyphens and underscores only, minimum 2 characters
    pattern = r'^[a-zA-Z0-9_-]{2,30}$'
    return bool(re.match(pattern, user_id))

def get_memory_category_icon(category: str) -> str:
    """Returns an icon for each memory category"""
    icons = {
        'personal': '👤',
        'professional': '💼',
        'preferences': '❤️',
        'important_facts': '⭐'
    }
    return icons.get(category, '📌')