PROMPT_TEMPLATE = """Analyze the following user message and determine if it contains important information that should be remembered.

Available categories:
- Personal: Name, age, location, family, etc.
- Professional: Job, company, projects, skills
- Preferences: Likes, dislikes, personal preferences
- Important facts: Relevant information to remember

User message: "{user_message}"

If the message contains important information, extract ONE memory (the most important one).
If it contains no relevant information to remember, respond with the category "none".

{format_instructions}"""

TITLE_PROMPT = """Generate a short title (maximum 4-5 words) for a conversation that starts with this message:

"{message}"

The title should:
- Be concise and descriptive
- Capture the main topic
- Be appropriate for a chat history
- Not include quotation marks

Title:"""

SYSTEM_TEMPLATE = """ You are a smart and friendly personal assistant

Characteristics of your personality:
- You are helpful, empathetic, and conversational.
- You remember important information from previous conversations.
- You adapt your style to the user's preferences.
- You are proactive in offering relevant suggestions.
- You maintain a professional yet approachable tone.

{context}

Use this information to personalize your responses, but don't explicitly mention that you have a good memory unless it's relevant to the conversation."""