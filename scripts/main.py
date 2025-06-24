#!/usr/bin/env python3
"""
Luna - Dog Expert RAG System
A comprehensive RAG-based chatbot specialized in dog-related queries
Uses ChromaDB, LangChain, and Chainlit with persistent chat history
"""

import uuid
import traceback,random
from datetime import datetime

# Core dependencies
import chainlit as cl
from utils.lunaConfig import LunaConfig
from utils.luna import LunaRAG
from utils.helperUtils import log_with_boxed_format
import os
from dotenv import load_dotenv
load_dotenv()

from utils.loggerSetup import get_logger


# Get the logger
logger = get_logger(__name__, "luna.log", console_output=os.getenv('CMD_OUTPUT','t') == 't')



### ----------------------------------  main entry point when we run chainlit run main.py  ----------------------------------------
def initialize_luna_before_chat(luna_rag):
    """Initialize Luna system before starting the chat interface"""
    
    logger.info("Pre-initializing Luna system...")
    
    try:
        # Setup directories
        LunaConfig.setup_directories()
        logger.info("Directories set up")
        
        # Initialize RAG system
        luna_rag = LunaRAG()
        if luna_rag.initialize_system():
            logger.info("Luna system ready for chat!")
            return luna_rag, True
        else:
            logger.error("Failed to initialize Luna system")
            return luna_rag, False
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return luna_rag, False
    
# Global RAG system - will be initialized automatically
luna_rag = None

# Auto-initialize when module is imported (for Chainlit)
# logger.info(f"\n\n[{datetime.now()}] Welcome! Luna 🐶 is waking up :) Initializing...\n\n")

log_with_boxed_format(logger)

try:
    luna_rag, active = initialize_luna_before_chat(luna_rag)
    if active:
        logger.info(f"[{datetime.now()}] Luna system auto-initialized successfully!")
    else:
        logger.info(f"[{datetime.now()}] Luna system auto-initialization failed!")
except Exception as e:
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
            logger.info("System not initialized, attempting initialization...")
            
            # Show loading message
            loading_msg = cl.Message(
                content="🐕 Initializing Luna... Please wait while I set up the dog knowledge base.",
                author="Luna"
            )
            await loading_msg.send()
            
            # Try to initialize
            luna_rag, active = initialize_luna_before_chat(luna_rag)
            if not active:
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
            logger.info("Greeting detected, responding immediately")
        else:
            # Show thinking indicator only for non-greetings
            thinking_msg = cl.Message(content="🤔 Thinking...", author="Luna")
            await thinking_msg.send()
        
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
        
        logger.info(f"Response sent - Source: {response_source}")

    except Exception as e:
        logger.error(f"❌ Error handling message: {str(e)}")
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
            logger.info(f"Chat session ended: {session_id}")
    except Exception as e:
        logger.error(f"❌ Error in chat end: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")