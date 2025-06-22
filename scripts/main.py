#!/usr/bin/env python3
"""
Luna - Dog Expert RAG System
A comprehensive RAG-based chatbot specialized in dog-related queries
Uses ChromaDB, LangChain, and Chainlit with persistent chat history
"""

import os
import uuid
import traceback,random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Core dependencies
import chainlit as cl
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from chromadb.config import Settings
from utils.logger_setup import get_logger

### helper utils
from utils.helperUtils import GreetingHandler,LunaConfig


# Get the logger
logger = get_logger(__name__, "luna.log", console_output=False)




class PDFProcessor:
    """Handles PDF loading and text processing using DirectoryLoader"""
    
    def __init__(self, pdf_dir: Path):
        self.pdf_dir = pdf_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=LunaConfig.CHUNK_SIZE,
            chunk_overlap=LunaConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdfs(self) -> List[Document]:
        """Load all PDFs from directory using DirectoryLoader"""
        try:
            if not self.pdf_dir.exists():
                print(f"[{datetime.now()}] PDF directory {self.pdf_dir} does not exist")
                logger.warning(f"PDF directory {self.pdf_dir} does not exist")
                return []
            
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"[{datetime.now()}] No PDF files found in {self.pdf_dir}")
                logger.warning(f"No PDF files found in {self.pdf_dir}")
                return []
            
            print(f"[{datetime.now()}] Loading {len(pdf_files)} PDF files using DirectoryLoader...")
            logger.info(f"Loading {len(pdf_files)} PDF files using DirectoryLoader...")
            
            # Use DirectoryLoader to load all PDFs
            loader = DirectoryLoader(
                str(self.pdf_dir),
                glob="*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True
            )
            
            print(f"[{datetime.now()}] Processing PDF documents...")
            documents = loader.load()
            
            # Add enhanced metadata
            for i, doc in enumerate(documents):
                source_path = Path(doc.metadata.get('source', ''))
                doc.metadata.update({
                    'source_file': source_path.name if source_path else f'document_{i}',
                    'file_path': str(source_path),
                    'loaded_at': datetime.now().isoformat(),
                    'document_type': 'pdf',
                    'processor': 'DirectoryLoader'
                })
            
            print(f"[{datetime.now()}] Successfully loaded {len(documents)} documents from {len(pdf_files)} PDFs")
            logger.info(f"Successfully loaded {len(documents)} documents from {len(pdf_files)} PDFs")
            return documents
            
        except Exception as e:
            print(f"[{datetime.now()}] Error loading PDFs: {str(e)}")
            logger.error(f"Error loading PDFs: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            print(f"[{datetime.now()}] Splitting {len(documents)} documents into chunks...")
            logger.info(f"Splitting {len(documents)} documents into chunks...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            print(f"[{datetime.now()}] Split {len(documents)} documents into {len(chunks)} chunks")
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"[{datetime.now()}] Error splitting documents: {str(e)}")
            logger.error(f"Error splitting documents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

class ChromaManager:
    """Manages ChromaDB operations"""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model=LunaConfig.EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create and populate vector store"""
        try:
            if not documents:
                print(f"[{datetime.now()}] No documents provided for vector store creation")
                logger.error("No documents provided for vector store creation")
                return False
            
            print(f"[{datetime.now()}] Creating ChromaDB vector store...")
            logger.info("Creating ChromaDB vector store...")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name="luna_dogs"
            )
            
            print(f"[{datetime.now()}] Vector store created with {len(documents)} documents")
            logger.info(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] Error creating vector store: {str(e)}")
            logger.error(f"Error creating vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store"""
        try:
            if not self.persist_directory.exists():
                print(f"[{datetime.now()}] Vector store directory doesn't exist")
                logger.warning("Vector store directory doesn't exist")
                return False
            
            print(f"[{datetime.now()}] Loading existing vector store...")
            logger.info("Loading existing vector store...")
            
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name="luna_dogs"
            )
            
            # Test the connection
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count > 0:
                print(f"[{datetime.now()}] Loaded vector store with {count} documents")
                logger.info(f"Loaded vector store with {count} documents")
                return True
            else:
                print(f"[{datetime.now()}] Vector store is empty")
                logger.warning("Vector store is empty")
                return False
            
        except Exception as e:
            print(f"[{datetime.now()}] Error loading vector store: {str(e)}")
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_retriever(self):
        """Get retriever for the vector store"""
        if self.vectorstore is None:
            print(f"[{datetime.now()}] Vector store not initialized")
            logger.warning("Vector store not initialized")
            return None
        
        print(f"[{datetime.now()}] Creating retriever with top_k={LunaConfig.TOP_K}")
        logger.info(f"Creating retriever with top_k={LunaConfig.TOP_K}")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": LunaConfig.TOP_K}
        )

class ChatHistoryManager:
    """Manages chat history persistence"""
    
    def __init__(self, history_dir: Path):
        self.history_dir = history_dir
    
    def get_session_file(self, session_id: str) -> Path:
        """Get file path for session history"""
        return self.history_dir / f"session_{session_id}.txt"
    
    def save_message(self, session_id: str, role: str, content: str):
        """Save a message to history file"""
        try:
            file_path = self.get_session_file(session_id)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {role.upper()}: {content}\n")
                
            print(f"[{datetime.now()}] Saved {role} message to history")
            logger.info(f"Saved {role} message to history for session {session_id}")
                
        except Exception as e:
            print(f"[{datetime.now()}] Error saving message: {str(e)}")
            logger.error(f"Error saving message: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        """Load chat history for a session"""
        try:
            file_path = self.get_session_file(session_id)
            if not file_path.exists():
                print(f"[{datetime.now()}] No existing history for session {session_id}")
                logger.info(f"No existing history for session {session_id}")
                return []
            
            history = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "] " in line:
                        # Parse: [timestamp] ROLE: content
                        parts = line.split("] ", 1)
                        if len(parts) == 2:
                            timestamp = parts[0][1:]  # Remove opening bracket
                            role_content = parts[1]
                            if ": " in role_content:
                                role, content = role_content.split(": ", 1)
                                history.append({
                                    "timestamp": timestamp,
                                    "role": role.lower(),
                                    "content": content
                                })
            
            recent_history = history[-LunaConfig.MEMORY_WINDOW*2:]  # Keep recent messages
            print(f"[{datetime.now()}] Loaded {len(recent_history)} recent messages from history")
            logger.info(f"Loaded {len(recent_history)} recent messages from history for session {session_id}")
            return recent_history
            
        except Exception as e:
            print(f"[{datetime.now()}] Error loading history: {str(e)}")
            logger.error(f"Error loading history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

class LunaRAG:
    """Main RAG system for Luna with greeting support"""
    
    def __init__(self):
        self.chroma_manager = ChromaManager(LunaConfig.CHROMA_DIR)
        self.history_manager = ChatHistoryManager(LunaConfig.HISTORY_DIR)
        self.greeting_handler = GreetingHandler()  # Add greeting handler
        self.llm = ChatOpenAI(
            model=LunaConfig.OPENAI_MODEL,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.chain = None
        self.memory = None
        self.is_initialized = False
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system before starting chat"""
        try:
            print(f"[{datetime.now()}] Initializing Luna RAG System...")
            logger.info("Initializing Luna RAG System...")
            
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                print(f"[{datetime.now()}] OpenAI API key not found!")
                logger.error("OpenAI API key not found!")
                return False
            
            print(f"[{datetime.now()}] OpenAI API key found")
            logger.info("OpenAI API key found")
            
            # Try to load existing vector store first
            if self.chroma_manager.load_vectorstore():
                print(f"[{datetime.now()}] Using existing vector store")
                logger.info("Using existing vector store")
            else:
                print(f"[{datetime.now()}] Creating new vector store from PDFs...")
                logger.info("Creating new vector store from PDFs...")
                
                # Process PDFs using DirectoryLoader
                pdf_processor = PDFProcessor(LunaConfig.PDF_DIR)
                documents = pdf_processor.load_pdfs()
                
                if not documents:
                    print(f"[{datetime.now()}] No documents loaded. Please add PDF files to the data directory.")
                    logger.error("No documents loaded. Please add PDF files to the data directory.")
                    return False
                
                chunks = pdf_processor.split_documents(documents)
                if not chunks:
                    print(f"[{datetime.now()}] No document chunks created")
                    logger.error("No document chunks created")
                    return False
                    
                if not self.chroma_manager.create_vectorstore(chunks):
                    print(f"[{datetime.now()}] Failed to create vector store")
                    logger.error("Failed to create vector store")
                    return False
            
            # Setup retriever and chain
            print(f"[{datetime.now()}] Setting up retriever...")
            logger.info("Setting up retriever...")
            
            retriever = self.chroma_manager.get_retriever()
            if retriever is None:
                print(f"[{datetime.now()}] Failed to create retriever")
                logger.error("Failed to create retriever")
                return False
            
            print(f"[{datetime.now()}] Retriever created successfully")
            logger.info("Retriever created successfully")
            
            # Create custom prompt with word limit
            print(f"[{datetime.now()}] Setting up custom prompt template...")
            logger.info("Setting up custom prompt template...")
            
            prompt_template = f"""You are Luna, a friendly and knowledgeable dog expert assistant. Your specialty is providing helpful, accurate information about dogs, including breeds, training, health, behavior, and general care.

Use the following context from dog-related documents and Chat History to answer the user's question. If the context and Chat History don't contain relevant information, respond with "I don't have sufficient information about that topic in my knowledge base. Please ask me anything else about dogs!"

Context from documents:
{{context}}

Chat History:
{{chat_history}}

Human Question: {{question}}

IMPORTANT: Keep your response to exactly {LunaConfig.MAX_RESPONSE_WORDS} words or less. Be concise and direct.

Luna's Response:"""

            custom_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            print(f"[{datetime.now()}] Custom prompt template created")
            logger.info("Custom prompt template created")
            
            # Setup memory
            print(f"[{datetime.now()}] Setting up conversation memory...")
            logger.info("Setting up conversation memory...")
            
            self.memory = ConversationBufferWindowMemory(
                k=LunaConfig.MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            print(f"[{datetime.now()}] Memory configured with window size: {LunaConfig.MEMORY_WINDOW}")
            logger.info(f"Memory configured with window size: {LunaConfig.MEMORY_WINDOW}")
            
            # Create conversational retrieval chain
            print(f"[{datetime.now()}] Creating conversational retrieval chain...")
            logger.info("Creating conversational retrieval chain...")
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                verbose=True
            )
            
            print(f"[{datetime.now()}] Conversational chain created successfully")
            logger.info("Conversational chain created successfully")
            
            self.is_initialized = True
            print(f"[{datetime.now()}] Luna RAG system initialization completed successfully!")
            logger.info("Luna RAG system initialization completed successfully!")
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] Error initializing RAG system: {str(e)}")
            logger.error(f"Error initializing RAG system: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _get_formatted_history(self) -> str:
        """Get formatted chat history for logging"""
        if not self.memory or not self.memory.chat_memory:
            return "No chat history"
        
        history_messages = []
        for message in self.memory.chat_memory.messages:
            if hasattr(message, 'content'):
                role = "Human" if message.__class__.__name__ == "HumanMessage" else "AI"
                history_messages.append(f"{role}: {message.content}")
        
        return "\n".join(history_messages) if history_messages else "No chat history"
    
    def _construct_final_prompt(self, question: str, context: str) -> str:
        """Construct the final prompt that will be sent to the model"""
        chat_history = self._get_formatted_history()
        
        final_prompt = f"""You are Luna, a friendly and knowledgeable dog expert assistant. Your specialty is providing helpful, accurate information about dogs, including breeds, training, health, behavior, and general care.

Use the following context from dog-related documents and Chat History to answer the user's question. If the context and Chat History don't contain relevant information, respond with "I don't have sufficient information about that topic in my knowledge base. Please ask me anything else about dogs!"

Context from documents:
{context}

Chat History:
{chat_history}

Human Question: {question}

IMPORTANT: Keep your response to exactly {LunaConfig.MAX_RESPONSE_WORDS} words or less. Be concise and direct.

Luna's Response:"""
        
        return final_prompt
    
    def _determine_response_source(self, sources: List[Document], answer: str) -> str:
        """Determine if response is from PDF or general AI knowledge"""
        # Check if answer contains the "insufficient information" message
        if "don't have sufficient information" in answer.lower():
            return "GENERAL_AI"
        
        # Check if we have relevant sources
        if not sources:
            return "GENERAL_AI"
        
        # If we have sources and the answer doesn't say insufficient info, it's likely from PDF
        return "PDF_SEARCH"
    
    def get_response(self, question: str, session_id: str) -> Tuple[str, List[Document], str]:
        """Get response from RAG system with greeting handling"""
        try:
            if not self.is_initialized or self.chain is None:
                error_msg = "Sorry, the system is not properly initialized."
                print(f"[{datetime.now()}] {error_msg}")
                logger.error("System not initialized when trying to get response")
                return error_msg, [], "ERROR"
            
            print(f"[{datetime.now()}] Processing question: {question[:50]}...")
            logger.info(f"Processing question from session {session_id}")
            
            # Check if it's a greeting first
            if self.greeting_handler.is_greeting(question):
                print(f"[{datetime.now()}] Detected greeting, responding without vector search")
                logger.info("Detected greeting, responding without vector search")
                
                greeting_response = self.greeting_handler.get_greeting_response(question)
                
                # Still save to history
                self.history_manager.save_message(session_id, "user", question)
                self.history_manager.save_message(session_id, "ai", greeting_response)
                
                # Add to memory for conversation context
                if self.memory:
                    self.memory.chat_memory.add_user_message(question)
                    self.memory.chat_memory.add_ai_message(greeting_response)
                
                print(f"[{datetime.now()}] Greeting response: {greeting_response}")
                logger.info(f"Greeting response provided")
                
                return greeting_response, [], "GREETING"
            
            # For non-greetings, proceed with normal RAG processing
            print(f"[{datetime.now()}] Non-greeting detected, proceeding with RAG search")
            logger.info("Non-greeting detected, proceeding with RAG search")
            
            # Get response from RAG
            result = self.chain({"question": question})
            answer = result.get("answer", "I couldn't generate a response.")
            sources = result.get("source_documents", [])
            
            # Determine response source
            response_source = self._determine_response_source(sources, answer)
            
            # Log the final prompt and response details
            context = "\n".join([doc.page_content[:200] + "..." for doc in sources[:3]]) if sources else "No relevant context found"
            final_prompt = self._construct_final_prompt(question, context)
            
            print("\n" + "="*80)
            print(f"[{datetime.now()}] FINAL INPUT PROMPT TO MODEL:")
            print("="*80)
            print(final_prompt)
            print("="*80)
            print(f"[{datetime.now()}] MODEL RESPONSE: {answer}")
            print(f"[{datetime.now()}] RESPONSE SOURCE: {response_source}")
            print(f"[{datetime.now()}] NUMBER OF SOURCES: {len(sources)}")
            print("="*80 + "\n")
            
            # Log to file
            logger.info("="*80)
            logger.info("FINAL INPUT PROMPT TO MODEL:")
            logger.info(final_prompt)
            logger.info("="*80)
            logger.info(f"MODEL RESPONSE: {answer}")
            logger.info(f"RESPONSE SOURCE: {response_source}")
            logger.info(f"NUMBER OF SOURCES: {len(sources)}")
            logger.info("="*80)
            
            # Count words in response
            word_count = len(answer.split())
            print(f"[{datetime.now()}] Response word count: {word_count}/{LunaConfig.MAX_RESPONSE_WORDS}")
            logger.info(f"Response word count: {word_count}/{LunaConfig.MAX_RESPONSE_WORDS}")
            
            print(f"[{datetime.now()}] Generated response with {len(sources)} source documents")
            logger.info(f"Generated response with {len(sources)} source documents")
            
            # Save to history
            self.history_manager.save_message(session_id, "user", question)
            self.history_manager.save_message(session_id, "ai", answer)
            
            return answer, sources, response_source
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"[{datetime.now()}] Error getting response: {str(e)}")
            logger.error(f"Error getting response: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return error_msg, [], "ERROR"
    
    def load_session_history(self, session_id: str):
        """Load chat history for a session"""
        try:
            print(f"[{datetime.now()}] Loading session history for: {session_id}")
            logger.info(f"Loading session history for: {session_id}")
            
            history = self.history_manager.load_history(session_id)
            
            # Clear current memory
            if self.memory:
                self.memory.clear()
                print(f"[{datetime.now()}] Cleared existing memory")
                logger.info("Cleared existing memory")
            
            # Add history to memory
            for entry in history:
                if entry["role"] == "user":
                    self.memory.chat_memory.add_user_message(entry["content"])
                elif entry["role"] == "ai":  # Changed from "assistant" to "ai"
                    self.memory.chat_memory.add_ai_message(entry["content"])
            
            print(f"[{datetime.now()}] Loaded {len(history)} messages into memory")
            logger.info(f"Loaded {len(history)} messages into memory")
                    
        except Exception as e:
            print(f"[{datetime.now()}] Error loading session history: {str(e)}")
            logger.error(f"Error loading session history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

# Global RAG system - will be initialized automatically
luna_rag = None

def initialize_luna_before_chat():
    """Initialize Luna system before starting the chat interface"""
    global luna_rag
    
    print(f"[{datetime.now()}] Pre-initializing Luna system...")
    logger.info("Pre-initializing Luna system...")
    
    try:
        # Setup directories
        LunaConfig.setup_directories()
        print(f"[{datetime.now()}] Directories set up")
        logger.info("Directories set up")
        
        # Initialize RAG system
        luna_rag = LunaRAG()
        if luna_rag.initialize_system():
            print(f"[{datetime.now()}] Luna system ready for chat!")
            logger.info("Luna system ready for chat!")
            return True
        else:
            print(f"[{datetime.now()}] Failed to initialize Luna system")
            logger.error("Failed to initialize Luna system")
            return False
    except Exception as e:
        print(f"[{datetime.now()}] Error during initialization: {str(e)}")
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

### ----------------------------------  main entry point when we run chainlit run main.py  ----------------------------------------

# Auto-initialize when module is imported (for Chainlit)
print(f"\n\n[{datetime.now()}] Welcome! Luna 🐶 is waking up :) Initializing...\n\n")
try:
    if initialize_luna_before_chat():
        print(f"[{datetime.now()}] Luna system auto-initialized successfully!")
    else:
        print(f"[{datetime.now()}] Luna system auto-initialization failed!")
except Exception as e:
    print(f"[{datetime.now()}] Error during auto-initialization: {str(e)}")
    logger.error(f"Error during auto-initialization: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")



# Chainlit Application
@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        global luna_rag
        
        # Check if system is pre-initialized, if not try to initialize
        if luna_rag is None or not luna_rag.is_initialized:
            print(f"[{datetime.now()}] System not initialized, attempting initialization...")
            logger.info("System not initialized, attempting initialization...")
            
            # Show loading message
            loading_msg = cl.Message(
                content="🐕 Initializing Luna... Please wait while I set up the dog knowledge base.",
                author="Luna"
            )
            await loading_msg.send()
            
            # Try to initialize
            if not initialize_luna_before_chat():
                await loading_msg.update(
                    content="Setup Failed\n\nCouldn't initialize the knowledge base. Please check:\n- PDF files are in the '../data/pdfs' directory\n- OpenAI API key is valid\n- All dependencies are installed"
                )
                return
            
            await loading_msg.update(content="Luna system initialized successfully!")
        
        # Use user session for consistent session management
        session_id = cl.user_session.get("id")
        if not session_id:
            # Fall back to generating if not available
            session_id = str(uuid.uuid4())
        
        # Store session ID consistently
        cl.user_session.set("session_id", session_id)
        
        print(f"[{datetime.now()}] Chat session ID: {session_id}")
        logger.info(f"Chat session ID: {session_id}")
        
        # Load existing history if any
        luna_rag.load_session_history(session_id)
        
        # Welcome message
        welcome_msg = f"""Hi, I am Luna 🐶

I can help you with:
• **Dog breeds** and their characteristics
• **Training tips** and behavioral guidance  
• **Health and nutrition** advice
• **General dog care** information

What would you like to know about dogs today? 🐾"""
        
        await cl.Message(content=welcome_msg, author="Luna").send()
        
    except Exception as e:
        print(f"[{datetime.now()}] Error in chat start: {str(e)}")
        logger.error(f"Error in chat start: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.Message(
            content=f"Initialization Error: {str(e)}",
            author="System"
        ).send()

conv = 0
@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with greeting detection"""
    thinking_msg = None
    try:
        global luna_rag 
        global conv
        conv +=1
        print(f"\n\n ==============x============= conversation :{conv} ==============x============= \n\n")
        logger.info(f"\n\n ==============x============= conversation :{conv} ==============x============= \n\n")
        
        session_id = cl.user_session.get("session_id")
        if not session_id:
            await cl.Message(
                content="Session not initialized. Please refresh the page.",
                author="System"
            ).send()
            return
        
        if luna_rag is None or not luna_rag.is_initialized:
            await cl.Message(
                content="Luna system is not initialized. Please restart the application.",
                author="System"
            ).send()
            return
        
        # Check if it's a greeting - if so, don't show thinking indicator
        if luna_rag.greeting_handler.is_greeting(message.content):
            print(f"[{datetime.now()}] Greeting detected, responding immediately")
            logger.info("Greeting detected, responding immediately")
        else:
            # Show thinking indicator only for non-greetings
            thinking_msg = cl.Message(content="🤔 Thinking...", author="Luna")
            await thinking_msg.send()
        
        print(f"[{datetime.now()}] Processing message from session: {session_id}")
        logger.info(f"Processing message from session: {session_id}")
        
        # Get response from RAG system (now with greeting handling)
        answer, sources, response_source = luna_rag.get_response(message.content, session_id)
        
        # Send response appropriately
        if thinking_msg:
            # Update the thinking message with the actual response
            thinking_msg.content = answer
            await thinking_msg.update()
        else:
            # Send greeting response directly
            await cl.Message(content=answer, author="Luna").send()
        
        print(f"[{datetime.now()}] Response sent - Source: {response_source}")
        logger.info(f"Response sent - Source: {response_source}")

    except Exception as e:
        print(f"[{datetime.now()}]❌ Error handling message: {str(e)}")
        logger.error(f"Error handling message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.Message(
            content=f"🐕 Woof! I encountered an error: {str(e)}\n\nPlease try asking your question again!",
            author="Luna"
        ).send()

@cl.on_chat_end
async def end():
    """Handle chat end"""
    try:
        session_id = cl.user_session.get("session_id")
        if session_id:
            print(f"👋 Chat session ended: {session_id}")
            logger.info(f"Chat session ended: {session_id}")
    except Exception as e:
        print(f"❌ Error in chat end: {str(e)}")
        logger.error(f"Error in chat end: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")