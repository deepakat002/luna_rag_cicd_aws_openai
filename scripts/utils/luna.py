### importing required libraries
from datetime import datetime
from pathlib import Path
from typing import List 
import traceback, os
from utils.helperUtils import GreetingHandler
from utils.lunaConfig import LunaConfig
from utils.chromaManager import ChromaManager
from utils.historyManager import ChatHistoryManager
from utils.pdfmanager import PDFProcessor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Tuple


from dotenv import load_dotenv
load_dotenv()


from utils.loggerSetup import get_logger

# Get the logger
logger = get_logger("luna", "luna.log",  console_output=os.getenv('CMD_OUTPUT') == 't')


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
            logger.info("Initializing Luna RAG System...")
            
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OpenAI API key not found!")
                return False
            
            logger.info("OpenAI API key found")
            
            # Try to load existing vector store first
            if self.chroma_manager.load_vectorstore():
                logger.info("Using existing vector store")
            else:
                logger.info("Creating new vector store from PDFs...")
                
                # Process PDFs using DirectoryLoader
                pdf_processor = PDFProcessor(LunaConfig.PDF_DIR)
                documents = pdf_processor.load_pdfs()
                
                if not documents:
                    logger.error("No documents loaded. Please add PDF files to the data directory.")
                    return False
                
                chunks = pdf_processor.split_documents(documents)
                if not chunks:
                    logger.error("No document chunks created")
                    return False
                    
                if not self.chroma_manager.create_vectorstore(chunks):
                    logger.error("Failed to create vector store")
                    return False
            
            # Setup retriever and chain
            logger.info("Setting up retriever...")
            
            retriever = self.chroma_manager.get_retriever()
            if retriever is None:
                logger.error("Failed to create retriever")
                return False
            
            logger.info("Retriever created successfully")
            
            # Create custom prompt with word limit
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
            
            logger.info("Custom prompt template created")
            
            # Setup memory
            logger.info("Setting up conversation memory...")
            
            self.memory = ConversationBufferWindowMemory(
                k=LunaConfig.MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            logger.info(f"Memory configured with window size: {LunaConfig.MEMORY_WINDOW}")
            
            # Create conversational retrieval chain
            logger.info("Creating conversational retrieval chain...")
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                verbose=True,
                rephrase_question=False
            )
            
            logger.info("Conversational chain created successfully")
            
            self.is_initialized = True
            logger.info("Luna RAG system initialization completed successfully!")
            return True
            
        except Exception as e:
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
                logger.error("System not initialized when trying to get response")
                return error_msg, [], "ERROR"
            
            logger.info(f"Processing question  {question[:50]} from session {session_id}")
            
            # Check if it's a greeting first
            if self.greeting_handler.is_greeting(question):
                logger.info("Detected greeting, responding without vector search")
                
                greeting_response = self.greeting_handler.get_greeting_response(question)
                
                # Still save to history
                self.history_manager.save_message(session_id, "user", question)
                self.history_manager.save_message(session_id, "ai", greeting_response)
                
                # Add to memory for conversation context
                if self.memory:
                    self.memory.chat_memory.add_user_message(question)
                    self.memory.chat_memory.add_ai_message(greeting_response)
                
                logger.info(f"Greeting response {greeting_response} provided")
                
                return greeting_response, [], "GREETING"
            
            # For non-greetings, proceed with normal RAG processing
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
                        
            # Log to file
            logger.info("\n" + "="*80)
            logger.info("FINAL INPUT PROMPT TO MODEL:")
            logger.info("="*80)
            logger.info(final_prompt)
            logger.info("="*80)
            logger.info(f"MODEL RESPONSE: {answer}")
            logger.info(f"RESPONSE SOURCE: {response_source}")
            logger.info(f"NUMBER OF SOURCES: {len(sources)}")
            logger.info("="*80 + "\n")
            
            # Count words in response
            word_count = len(answer.split())
            logger.info(f"Response word count: {word_count}/{LunaConfig.MAX_RESPONSE_WORDS}")
            
            logger.info(f"Generated response with {len(sources)} source documents")
            
            # Save to history
            self.history_manager.save_message(session_id, "user", question)
            self.history_manager.save_message(session_id, "ai", answer)
            
            return answer, sources, response_source
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            logger.error(f"Error getting response: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return error_msg, [], "ERROR"
    
    def load_session_history(self, session_id: str):
        """Load chat history for a session"""
        try:
            logger.info(f"Loading session history for: {session_id}")
            
            history = self.history_manager.load_history(session_id)
            
            # Clear current memory
            if self.memory:
                self.memory.clear()
                logger.info("Cleared existing memory")
            
            # Add history to memory
            for entry in history:
                if entry["role"] == "user":
                    self.memory.chat_memory.add_user_message(entry["content"])
                elif entry["role"] == "ai":  # Changed from "assistant" to "ai"
                    self.memory.chat_memory.add_ai_message(entry["content"])
            
            logger.info(f"Loaded {len(history)} messages into memory")
                    
        except Exception as e:
            logger.error(f"Error loading session history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
