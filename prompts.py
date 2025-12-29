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