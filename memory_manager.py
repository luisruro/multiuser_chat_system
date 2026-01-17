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

from dotenv import load_dotenv
load_dotenv()

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
                embedding_function=OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
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
            self.extraction_llm = ChatOpenAI(
                model=DEFAULT_MODEL, 
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
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
            chats_data.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
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
      
    # ===Vector Memory===
    
    def save_vector_memory(self, text: str, metadata: Optional[Dict] = None):
        """Save information in vector memory"""
        
        if not self.collection:
            return ""
        
        try:
            memory_id = str(uuid.uuid4())
            doc_metadata = metadata or {}
            doc_metadata.update({
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "memory_id": memory_id
            })
            
            self.collection.add(
                documents=[text],
                ids=[memory_id],
                metadatas=[doc_metadata]
            )
            
            return memory_id
        except Exception as e:
          print(f'Error saving vector memory {e}')
          return ''
      
    def search_vector_memory(self, query:str, k: int = MAX_VECTOR_RESULTS):
        """Search relevant information in vector memory"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
          print(f'Error searching in the vector memory')
          return []
      
    def get_all_vector_memories(self):
        """Get all user vector memories"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get()
            memories = []
            
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    memory = {
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    memories.append(memory)
                    
            return memories
        
        except Exception as e:
          print(f'Error getting vector memories {e}')
          return []
    
    # ====Smart Extraction===
    
    def extract_and_store_memories(self, user_message: str):
        """Extract and store memories using LLM"""
        if not self.extraction_chain:
            return self._extract_memories_manual(user_message)
        
        try:
            extracted_memory = self.extraction_chain.invoke({
                "user_message": user_message
            })
            
            if extracted_memory.category != "none" and extracted_memory.importance >= 2:
                memory_id = self.save_vector_memory(
                    extracted_memory.content,
                    {
                        'category': extracted_memory.category,
                        'importance': extracted_memory.importance,
                        'original_message': user_message[:200]
                    }
                )
                return bool(memory_id)
            return False
        except Exception as e:
          print(f'Error in automatic extraction: {e}')
          return self._extract_memories_manual(user_message)
        
    def _extract_memories_manual(self, user_message: str) -> bool:
        """Método manual de extracción (fallback)"""
        message_lower = user_message.lower()
        
        memory_rules = [
            (["me llamo", "mi nombre es", "soy"], "personal", f"Personal info: {user_message}"),
            (["trabajo en", "trabajo como", "mi profesión"], "professional", f"professional info: {user_message}"),
            (["me gusta", "me encanta", "prefiero", "odio"], "preferences", f"Preference: {user_message}"),
            (["importante", "recuerda que", "no olvides"], "important_facts", f"Important fact: {user_message}")
        ]
        
        for phrases, category, memory_text in memory_rules:
            if any(phrase in message_lower for phrase in phrases):
                memory_id = self.save_vector_memory(memory_text, {'category': category})
                return bool(memory_id)
        
        return False
    
    
class UserManager:
    """Simplified user manager"""
    
    @staticmethod
    def get_users():
        """Get a list of existing users"""
        if not os.path.exists(USERS_DIR):
            return []
        
        users = []
        for item in os.listdir(USERS_DIR):
            user_path = os.path.join(USERS_DIR, item)
            if os.path.isdir(user_path):
                users.append(item)
                
        return sorted(users)
    
    @staticmethod
    def user_exists(user_id):
        """Verify if a user exists"""
        user_path = os.path.join(USERS_DIR, user_id)
        return os.path.exists(user_path)
    
    @staticmethod
    def create_user(user_id):
        """Create a new user"""
        try:
            user_path = os.path.join(USERS_DIR, user_id)
            os.makedirs(user_path, exist_ok=True)
            return True
        except Exception as e:
          print(f'Error creating user {e}')
          return False