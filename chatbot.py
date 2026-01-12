from typing import List, Dict, Any
from langgraph.graph import StateGraph, START
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from memory_manager import ModernMemoryManager, MemoryState
from prompts import SYSTEM_TEMPLATE
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
import os

from dotenv import load_dotenv
load_dotenv()

class ModernChatbot:
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_manager = ModernMemoryManager(user_id)
        
        #Setting up model LLM
        self.llm = ChatOpenAI(
            model = DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            api_key= os.getenv("OPENAI_API_KEY")
        )
        
        #System template with dinamic context
        self.system_template = SYSTEM_TEMPLATE
        
        # Setting message trimming to manage the context
        self.message_trimer = trim_messages(
            strategy="last",
            max_tokens=4000,
            token_counter=self.llm,
            start_on="human",
            include_system=True
        )
        
        # Create LangGraph application
        self.app = self._create_app()