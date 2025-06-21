#!/usr/bin/env python3
"""
Luna - Dog Expert RAG System
A comprehensive RAG-based chatbot specialized in dog-related queries
Uses ChromaDB, LangChain, and Chainlit with persistent chat history
"""

import os
import sys
import json
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio

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
import chromadb
from chromadb.config import Settings
from utils.logger_setup import get_logger
# Get the logger
logger = get_logger(__name__, "luna.log", console_output=False)

# Configuration
class LunaConfig:
    """Configuration class for Luna system"""
    
    # Paths
    DATA_DIR = Path("../data")
    PDF_DIR = DATA_DIR / "pdfs"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    HISTORY_DIR = DATA_DIR / "chat_history"
    
    # Model settings
    OPENAI_MODEL =  "gpt-4o-mini" #"gpt-3.5-turbo" # "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Chunk settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K = 5
    MEMORY_WINDOW = 10
    
    # Response settings
    MAX_RESPONSE_WORDS = 50
    SIMILARITY_THRESHOLD = 0.7  # Threshold for determining if response is from PDF
    
    # Create directories
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.PDF_DIR, cls.CHROMA_DIR, cls.HISTORY_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)



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
                print(f"⚠️  PDF directory {self.pdf_dir} does not exist")
                logger.warning(f"PDF directory {self.pdf_dir} does not exist")
                return []
            
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"⚠️  No PDF files found in {self.pdf_dir}")
                logger.warning(f"No PDF files found in {self.pdf_dir}")
                return []
            
            print(f"📚 Loading {len(pdf_files)} PDF files using DirectoryLoader...")
            logger.info(f"Loading {len(pdf_files)} PDF files using DirectoryLoader...")
            
            # Use DirectoryLoader to load all PDFs
            loader = DirectoryLoader(
                str(self.pdf_dir),
                glob="*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True
            )
            
            print("🔄 Processing PDF documents...")
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
            
            print(f"✅ Successfully loaded {len(documents)} documents from {len(pdf_files)} PDFs")
            logger.info(f"Successfully loaded {len(documents)} documents from {len(pdf_files)} PDFs")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading PDFs: {str(e)}")
            logger.error(f"Error loading PDFs: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            print(f"✂️  Splitting {len(documents)} documents into chunks...")
            logger.info(f"Splitting {len(documents)} documents into chunks...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error splitting documents: {str(e)}")
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
                print("❌ No documents provided for vector store creation")
                logger.error("No documents provided for vector store creation")
                return False
            
            print("🔄 Creating ChromaDB vector store...")
            logger.info("Creating ChromaDB vector store...")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name="luna_dogs"
            )
            
            print(f"✅ Vector store created with {len(documents)} documents")
            logger.info(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            logger.error(f"Error creating vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store"""
        try:
            if not self.persist_directory.exists():
                print("⚠️  Vector store directory doesn't exist")
                logger.warning("Vector store directory doesn't exist")
                return False
            
            print("🔄 Loading existing vector store...")
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
                print(f"✅ Loaded vector store with {count} documents")
                logger.info(f"Loaded vector store with {count} documents")
                return True
            else:
                print("⚠️  Vector store is empty")
                logger.warning("Vector store is empty")
                return False
            
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_retriever(self):
        """Get retriever for the vector store"""
        if self.vectorstore is None:
            print("⚠️  Vector store not initialized")
            logger.warning("Vector store not initialized")
            return None
        
        print(f"🔍 Creating retriever with top_k={LunaConfig.TOP_K}")
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
                
            print(f"💾 Saved {role} message to history")
            logger.info(f"Saved {role} message to history for session {session_id}")
                
        except Exception as e:
            print(f"❌ Error saving message: {str(e)}")
            logger.error(f"Error saving message: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        """Load chat history for a session"""
        try:
            file_path = self.get_session_file(session_id)
            if not file_path.exists():
                print(f"📝 No existing history for session {session_id}")
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
            print(f"📚 Loaded {len(recent_history)} recent messages from history")
            logger.info(f"Loaded {len(recent_history)} recent messages from history for session {session_id}")
            return recent_history
            
        except Exception as e:
            print(f"❌ Error loading history: {str(e)}")
            logger.error(f"Error loading history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

class LunaRAG:
    """Main RAG system for Luna"""
    
    def __init__(self):
        self.chroma_manager = ChromaManager(LunaConfig.CHROMA_DIR)
        self.history_manager = ChatHistoryManager(LunaConfig.HISTORY_DIR)
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
            print("🐕 Initializing Luna RAG System...")
            logger.info("Initializing Luna RAG System...")
            
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OpenAI API key not found!")
                logger.error("OpenAI API key not found!")
                return False
            
            print("✅ OpenAI API key found")
            logger.info("OpenAI API key found")
            
            # Try to load existing vector store first
            if self.chroma_manager.load_vectorstore():
                print("✅ Using existing vector store")
                logger.info("Using existing vector store")
            else:
                print("🔄 Creating new vector store from PDFs...")
                logger.info("Creating new vector store from PDFs...")
                
                # Process PDFs using DirectoryLoader
                pdf_processor = PDFProcessor(LunaConfig.PDF_DIR)
                documents = pdf_processor.load_pdfs()
                
                if not documents:
                    print("❌ No documents loaded. Please add PDF files to the data directory.")
                    logger.error("No documents loaded. Please add PDF files to the data directory.")
                    return False
                
                chunks = pdf_processor.split_documents(documents)
                if not chunks:
                    print("❌ No document chunks created")
                    logger.error("No document chunks created")
                    return False
                    
                if not self.chroma_manager.create_vectorstore(chunks):
                    print("❌ Failed to create vector store")
                    logger.error("Failed to create vector store")
                    return False
            
            # Setup retriever and chain
            print("🔍 Setting up retriever...")
            logger.info("Setting up retriever...")
            
            retriever = self.chroma_manager.get_retriever()
            if retriever is None:
                print("❌ Failed to create retriever")
                logger.error("Failed to create retriever")
                return False
            
            print("✅ Retriever created successfully")
            logger.info("Retriever created successfully")
            
            # Create custom prompt with word limit
            print("📝 Setting up custom prompt template...")
            logger.info("Setting up custom prompt template...")
            
            prompt_template = f"""You are Luna, a friendly and knowledgeable dog expert assistant. Your specialty is providing helpful, accurate information about dogs, including breeds, training, health, behavior, and general care.

Use the following context from dog-related documents to answer the user's question. If the context doesn't contain relevant information, respond with "I don't have sufficient information about that topic in my knowledge base. Please ask me anything else about dogs!"

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
            
            print("✅ Custom prompt template created")
            logger.info("Custom prompt template created")
            
            # Setup memory
            print("🧠 Setting up conversation memory...")
            logger.info("Setting up conversation memory...")
            
            self.memory = ConversationBufferWindowMemory(
                k=LunaConfig.MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            print(f"✅ Memory configured with window size: {LunaConfig.MEMORY_WINDOW}")
            logger.info(f"Memory configured with window size: {LunaConfig.MEMORY_WINDOW}")
            
            # Create conversational retrieval chain
            print("⛓️  Creating conversational retrieval chain...")
            logger.info("Creating conversational retrieval chain...")
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                verbose=True
            )
            
            print("✅ Conversational chain created successfully")
            logger.info("Conversational chain created successfully")
            
            self.is_initialized = True
            print("🎉 Luna RAG system initialization completed successfully!")
            logger.info("Luna RAG system initialization completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {str(e)}")
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

Use the following context from dog-related documents to answer the user's question. If the context doesn't contain relevant information, respond with "I don't have sufficient information about that topic in my knowledge base. Please ask me anything else about dogs!"

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
        """Get response from RAG system"""
        try:
            if not self.is_initialized or self.chain is None:
                error_msg = "Sorry, the system is not properly initialized."
                print(f"⚠️  {error_msg}")
                logger.error("System not initialized when trying to get response")
                return error_msg, [], "ERROR"
            
            print(f"🤔 Processing question: {question[:50]}...")
            logger.info(f"Processing question from session {session_id}")
            
            # Get response
            result = self.chain({"question": question})
            answer = result.get("answer", "I couldn't generate a response.")
            sources = result.get("source_documents", [])
            
            # Determine response source
            response_source = self._determine_response_source(sources, answer)
            
            # Log the final prompt and response details
            context = "\n".join([doc.page_content[:200] + "..." for doc in sources[:3]]) if sources else "No relevant context found"
            final_prompt = self._construct_final_prompt(question, context)
            
            print("\n" + "="*80)
            print("📝 FINAL INPUT PROMPT TO MODEL:")
            print("="*80)
            print(final_prompt)
            print("="*80)
            print(f"🤖 MODEL RESPONSE: {answer}")
            print(f"📊 RESPONSE SOURCE: {response_source}")
            print(f"📚 NUMBER OF SOURCES: {len(sources)}")
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
            print(f"📝 Response word count: {word_count}/{LunaConfig.MAX_RESPONSE_WORDS}")
            logger.info(f"Response word count: {word_count}/{LunaConfig.MAX_RESPONSE_WORDS}")
            
            print(f"✅ Generated response with {len(sources)} source documents")
            logger.info(f"Generated response with {len(sources)} source documents")
            
            # Save to history
            self.history_manager.save_message(session_id, "user", question)
            self.history_manager.save_message(session_id, "ai", answer)
            
            return answer, sources, response_source
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"❌ Error getting response: {str(e)}")
            logger.error(f"Error getting response: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return error_msg, [], "ERROR"
    
    def load_session_history(self, session_id: str):
        """Load chat history for a session"""
        try:
            print(f"📚 Loading session history for: {session_id}")
            logger.info(f"Loading session history for: {session_id}")
            
            history = self.history_manager.load_history(session_id)
            
            # Clear current memory
            if self.memory:
                self.memory.clear()
                print("🧹 Cleared existing memory")
                logger.info("Cleared existing memory")
            
            # Add history to memory
            for entry in history:
                if entry["role"] == "user":
                    self.memory.chat_memory.add_user_message(entry["content"])
                elif entry["role"] == "ai":  # Changed from "assistant" to "ai"
                    self.memory.chat_memory.add_ai_message(entry["content"])
            
            print(f"✅ Loaded {len(history)} messages into memory")
            logger.info(f"Loaded {len(history)} messages into memory")
                    
        except Exception as e:
            print(f"❌ Error loading session history: {str(e)}")
            logger.error(f"Error loading session history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

# Global RAG system - will be initialized automatically
luna_rag = None

def initialize_luna_before_chat():
    """Initialize Luna system before starting the chat interface"""
    global luna_rag
    
    print("🚀 Pre-initializing Luna system...")
    logger.info("Pre-initializing Luna system...")
    
    try:
        # Setup directories
        LunaConfig.setup_directories()
        print("✅ Directories set up")
        logger.info("Directories set up")
        
        # Initialize RAG system
        luna_rag = LunaRAG()
        if luna_rag.initialize_system():
            print("🎉 Luna system ready for chat!")
            logger.info("Luna system ready for chat!")
            return True
        else:
            print("❌ Failed to initialize Luna system")
            logger.error("Failed to initialize Luna system")
            return False
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
### ----------------------------------  main entry point when we run chainlit run main.py  ----------------------------------------

# Auto-initialize when module is imported (for Chainlit)
print("\n\n ------------------- Welcome! Luna is waking up :) Initializing... ---------------------\n\n")
try:
    if initialize_luna_before_chat():
        print("✅ Luna system auto-initialized successfully!")
    else:
        print("❌ Luna system auto-initialization failed!")
except Exception as e:
    print(f"❌ Error during auto-initialization: {str(e)}")
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
            print("🔄 System not initialized, attempting initialization...")
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
                    content="❌ **Setup Failed**\n\nCouldn't initialize the knowledge base. Please check:\n- PDF files are in the '../data/pdfs' directory\n- OpenAI API key is valid\n- All dependencies are installed"
                )
                return
            
            await loading_msg.update(content="✅ Luna system initialized successfully!")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        
        print(f"👋 New chat session started: {session_id}")
        logger.info(f"New chat session started: {session_id}")
        
        # Load existing history if any
        luna_rag.load_session_history(session_id)
        
        # Welcome message
        welcome_msg = f"""Hi, I am Luna 🐶

I can help you with:
• **Dog breeds** and their characteristics
• **Training tips** and behavioral guidance  
• **Health and nutrition** advice
• **General dog care** information

* What would you like to know about dogs today?* 🐾"""
        
        await cl.Message(content=welcome_msg, author="Luna").send()
        
    except Exception as e:
        print(f"❌ Error in chat start: {str(e)}")
        logger.error(f"Error in chat start: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.Message(
            content=f"❌ **Initialization Error**: {str(e)}",
            author="System"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        global luna_rag
        
        session_id = cl.user_session.get("session_id")
        if not session_id:
            await cl.Message(
                content="Session not initialized. Please refresh the page.",
                author="System"
            ).send()
            return
        
        if luna_rag is None or not luna_rag.is_initialized:
            await cl.Message(
                content="❌ Luna system is not initialized. Please restart the application.",
                author="System"
            ).send()
            return
        
        # Show typing indicator
        async with cl.Step(name="Luna is thinking...") as step:
            step.output = "Searching knowledge base and generating response..."
            
            print(f"💭 Processing message from session: {session_id}")
            logger.info(f"Processing message from session: {session_id}")
            
            # Get response from RAG system
            answer, sources, response_source = luna_rag.get_response(message.content, session_id)
        
        # Prepare response with source indicator
        source_emoji = "📚" if response_source == "PDF_SEARCH" else "🤖"
        source_text = "PDF knowledge base" if response_source == "PDF_SEARCH" else "general AI knowledge"
        
        # Prepare source information
        source_info = ""
        if sources and response_source == "PDF_SEARCH":
            unique_sources = set()
            for doc in sources[:3]:  # Show top 3 sources
                source_file = doc.metadata.get('source_file', 'Unknown')
                unique_sources.add(source_file)
            
            if unique_sources:
                source_info = f"\n\n📚 *Sources: {', '.join(unique_sources)}*"
                print(f"📚 Response includes sources: {', '.join(unique_sources)}")
                logger.info(f"Response includes sources: {', '.join(unique_sources)}")
        
        # Add response source indicator
        response_indicator = f"\n\n{source_emoji} *Response from: {source_text}*"
        
        # Send response
        await cl.Message(
            content=answer + source_info + response_indicator,
            author="Luna"
        ).send()
        
        print(f"✅ Response sent - Source: {response_source}")
        logger.info(f"Response sent - Source: {response_source}")
        
    except Exception as e:
        print(f"❌ Error handling message: {str(e)}")
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

# Setup and initialization functions
def setup_luna():
    """Setup Luna system - run this first
       When you want to verify your PDF files and API key setup before running Chainlit.
       As a diagnostic or bootstrap step in a script, CI/CD, or testing environment."""
    try:
        print("🐕 Setting up Luna Dog Expert System...")
        
        # Setup directories
        LunaConfig.setup_directories()
        print(f"✅ Created directories:")
        print(f"   - PDF directory: {LunaConfig.PDF_DIR}")
        print(f"   - ChromaDB directory: {LunaConfig.CHROMA_DIR}")
        print(f"   - Chat history directory: {LunaConfig.HISTORY_DIR}")
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Please set your OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            return False
        
        # Check for PDFs
        pdf_files = list(LunaConfig.PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {LunaConfig.PDF_DIR}")
            print("   Please add your dog-related PDF files to this directory")
            return False
        
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        # Initialize system completely
        if initialize_luna_before_chat():
            print("✅ Luna system setup completed successfully!")
            print(f"\nLuna is now ready to start chatting with {LunaConfig.MAX_RESPONSE_WORDS}-word responses!")
            print("To start Luna, run:")
            print("   chainlit run app.py")
            return True
        else:
            print("❌ System setup failed. Check the logs for details.")
            return False
            
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        logger.error(f"Error during setup: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Check if running setup
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_luna()
    else:
        print("🐕 Luna Dog Expert RAG System")
        print(f"✅ Configured for {LunaConfig.MAX_RESPONSE_WORDS}-word responses")
        print("✅ Enhanced logging with detailed error tracebacks")
        print("✅ Response source tracking (PDF vs AI)")
        print("✅ Final prompt logging for debugging")
        print("Note: When running with 'chainlit run app.py', the system will auto-initialize.")
        print("For manual setup, run: python app.py setup")