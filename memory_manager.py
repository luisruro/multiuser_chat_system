import os
import uuid
import json
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import USERS_DIR, MAX_VECTOR_RESULTS, EMBEDDING_MODEL, DEFAULT_MODEL
from prompts import PROMPT_TEMPLATE, TITLE_PROMPT

# Extended state that combines messages with vector memory
class MemoryState(TypedDict):
    """State that combines LangGraph messages with vector memory."""
    messages: Annotated[List[BaseMessage], add_messages]
    vector_memories: List[str] # IDs of active vector memories
    user_profile: Dict[str, Any] # User profile
    last_memory_extraction: Optional[str] # Last processed message for memories

class ExtractedMemory(BaseModel):
    """Model for structured extracted memory."""
    category: str = Field(description="Category: personal, professional, preferences, important_facts")
    content: str = Field(description="Memory content")
    importance: int = Field(description="Importance from 1 to 5", ge=1, le=5)
    
class ModernMemoryManager:
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_dir = os.path.join(USERS_DIR, user_id)
        os.makedirs(self.user_dir, exist_ok=True)
        
        # Vector DB Chroma for transversal memory
        self.chromadb_path = os.path.join(self.user_dir, "chromadb")
        self._init_vector_db()
        
        # transversal memory smart extraction system
        self._init_extraction_system()
        
        # LangGraph DB route 
        self.langgraph_db_path = os.path.join(self.user_dir, "langgraph_memory.db")
        
        
    def _init_vector_db(self):
        """Initializes Chroma vector database"""
        try:
            self.vectorstore = Chroma(
                collection_name=f"memory_{self.user_id}",
                embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                persist_directory=self.chromadb_path
            )
            
            self.client = chromadb.PersistentClient(path=self.chromadb_path)
            try:
                self.collection = self.client.get_collection(f"memory_{self.user_id}")
            except:
                self.collection = self.client.create_collection(f"memory_{self.user_id}")
                
        except Exception as e:
          print(f"Error initializing Chromadb {e}")
          self.vectorstore = None
          self.collection = None
          
    def _init_extraction_system(self):
        """Initializes smart transversal memory extraction system"""
        try:
            self.extraction_llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
            self.memory_parser = PydanticOutputParser(pydantic_object=ExtractedMemory)
            
            self.extraction_template = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["user_message"],
                partial_variables={"format_instructions": self.memory_parser.get_format_instructions()}
            )
            
            self.extraction_chain = self.extraction_template | self.extraction_llm | self.memory_parser
        except Exception as e:
          print(f"Error initializing extraction system")
          self.extraction_chain = None
          
    # ===Chat Management (Hybrid: JSON lightweight + LangGraph persintence)===
    
    def get_user_chats(self):
        """Get all user chats"""
        try:
            # If not exist metadata file, return empty
            chats_meta_file = os.path.join(self.user_dir, "chats_meta.json")
            if not os.path.exists(chats_meta_file):
                return []
            
            # Load metadata
            with open(chats_meta_file, 'r', encoding='utf-8') as f:
                chats_data = json.load(f)
                
            # Order by last update
            chats_data.sort(key=lambda x: x.get('updated_at', ''), reversed=True)
            return chats_data
        except Exception as e:
          print(f'Error getting chats: {e}')
          return []
      
    def _save_chats_metadata(self, chats_data):
        """Save chat lightweight metadata"""
        try:
            chats_meta_file = os.path.join(self.user_dir, "chats_meta.json")
            with open(chats_meta_file, 'w', encoding='utf-8') as f:
                json.dump(chats_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
          print(f'Error saving chat metadata {e}')
      
    def create_new_chat(self, first_message: str = ""):
        """Create a new chat and update metadata"""
        chat_id = str(uuid.uuid4())
        
        # Generate a title based on the first message
        title = self._generate_chat_title(first_message) if first_message else "Nuevo Chat"
        
        # Create chat metadata
        new_chat = {
            "chat_id": chat_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        # Load exist chats and add the new one
        chats_data = self.get_user_chats()
        chats_data.append(new_chat)
        self._save_chats_metadata(chats_data)
        
        return chat_id
    
    def update_chat_metadata(self, chat_id, title: str = None, increment_messages: bool = False):
        """Update chat metadata"""
        chats_data = self.get_user_chats()
        
        for chat in chats_data:
            if chat['chat_id'] == chat_id:
                if title:
                    chat['title'] = title
                if increment_messages:
                    chat['message_count'] = chat.get('message_count', 0) + 1
                chat['updated_at'] = datetime.now().isoformat()
                break
        else:
            # if not ecist chat, creat input
            if chat_id:
                new_chat = {
                    "chat_id": chat_id,
                    "title": title or "Chat without title",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "message_count": 1 if increment_messages else 0
                }
                chats_data.append(new_chat)  
                
        self._save_chats_metadata(chats_data)
    
    def delete_chat(self, chat_id):
        """Delete a chat from metadata"""
        try:
            chats_data = self.get_user_chats()
            chats_data = [chat for chat in chats_data if chat["chat_id"] != chat_id]
            self._save_chats_metadata(chats_data)
            return True
        except Exception as e:
            print(f'Error deleting chat {e}')
            return False
        
    def get_chat_info(self, chat_id):
        """Get metadata for a specific chat """
        chats = self.get_user_chats()
        for chat in chats:
            if chat['chat_id'] == chat_id:
                return chat
        return None
    
    def _generate_chat_title(self, first_message):
        """Generate a title based on the first message"""
        try:
          if not self.extraction_llm:
              return first_message[:30] + "..." if len(first_message) > 30 else first_message
          
          title_prompt = PromptTemplate(
              template=TITLE_PROMPT,
              input_variables=["message"]
          )
          
          title_chain = title_prompt | self.extraction_llm
          
          response = title_chain.invoke({"message": first_message[:200]})
          
          title = response.content.strip().strip('"').strip("'")
          return title if len(title) <= 50 else title[:47] + "..."
      
        except Exception as e:
          print(f'Error generating title {e}')
          return first_message[:30] + "..." if len(first_message) > 30 else first_message