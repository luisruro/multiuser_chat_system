from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
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
        
    def _create_app(self):
        """Create LangGraph with extended state."""
        workflow = StateGraph(state_schema=MemoryState)
        
        def memory_retrieval_node(state):
            """Node that retrieves relevant memories."""
            messages = state['messages']
            
            if not messages:
                return {"vector_memories": []}
            
            # Get the last user message
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg
                    break
                
            if not last_user_message:
                return {"vector_memories": []}
            
            # Search vector relevant memories
            relevant_memories = self.memory_manager.search_vector_memory(
                last_user_message.content
            )
            
            return {"vector_memories": relevant_memories}
        
        def context_optimization_node(state):
            """Node that optimizes the context using  trim_messages"""
            messages = state['messages']
            
            # Applying smart trimming
            trimmed_messages = self.message_trimer.invoke(messages)
            
            return {"messages": trimmed_messages}
        
        def response_generation_node(state):
            """Node that generates the answer using the optimized context"""
            messages = state['messages']
            vector_memories = state.get('vector_memories', [])
            
            if not messages:
                return {"messages": []}
            
            #Build context with vector memories
            if vector_memories:
                context_parts = ["Relevant user information:"]
                for memory in vector_memories:
                    context_parts.append(f"- {memory}")
                context = "\n".join(context_parts)
            else:
                context = "There is no relavant prior information"
            
            # Create prompt with the dinamyc context
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_template.format(context=context)),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            #Generate the response
            chain = prompt | self.llm
            response = chain.invoke({"messages": messages})
            
            return {"messages": response}
        
        def memory_extraction_node(state):
            "Extract and save new vector memories"
            messages = state['messages']
            last_extraction = state.get('last_memory_extraction')
            
            #Get the last user message
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg
                    break
            
            if not last_user_message:
                return {}
            
            # Only process if we have not extracted any memories from this message
            if last_extraction != last_user_message.content:
                self.memory_manager.extract_and_store_memories(last_user_message.content)
                return {"last_memory_extraction": last_user_message.content}
            
            return {}
        
        # Setting up the Graph
        workflow.add_node("memory_retrieval", memory_retrieval_node)
        workflow.add_node("context_optimization", context_optimization_node)
        workflow.add_node("response_generation", response_generation_node)
        workflow.add_node("memory_extraction", memory_extraction_node)
        
        # Define the graph flow
        workflow.add_edge(START, "memory_retrieval")
        workflow.add_edge("memory_retrieval", "context_optimization")
        workflow.add_edge("context_optimization", "response_generation")
        workflow.add_edge("response_generation", "memory_extraction")
        workflow.add_edge("memory_extraction", END)
        
        # Setting persistence with Sqliteserver
        db_path = os.path.join(
            self.memory_manager.user_dir,
            "langgraph_memory.db"
        )
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def chat(self, message: str, chat_id: str = "default"):
        """send a message and get chatbot response"""
        try:
            #Setting specific chat thread
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            
            # Update the title if it is necessary
            chat_info = self.memory_manager.get_chat_info(chat_id)
            if chat_info["title"] == "Nuevo chat":
                chat_title = self.memory_manager._generate_chat_title(message)
                self.memory_manager.update_chat_metadata(chat_id, chat_title)
                
                #Invoke chatbot with the new message
                result = self.app.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config
                )
                
                # Extract the answer
                assistant_response = result["messages"][-1].content
                
                return {
                    "success": True,
                    "response": assistant_response,
                    "error": None,
                    "memories_used" : len(result.get("vector_memories", [])),
                    "context_optimized": True
                }

        except Exception as e:
            return {
                    "success": False,
                    "response": None,
                    "error": str(e),
                    "memories_used" : 0,
                    "context_optimized": False
                }