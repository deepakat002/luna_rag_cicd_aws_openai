import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
import asyncio
import warnings
import re

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

load_dotenv()

# -------------------- Globals --------------------
vectordb = None
fallback_chain = None
initialization_complete = False

# -------------------- Helper Functions --------------------
def is_greeting_or_simple(message_content):
    """
    Check if the message is a simple greeting or basic question that doesn't need document search.
    More restrictive to avoid misclassifying single words.
    """
    message_lower = message_content.lower().strip()

    # Define explicit greeting phrases and simple questions
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'whats up', 'sup', 'yo', 'greetings',
        'hi there', 'hello there', 'hey there'
    ]

    simple_questions = [
        'who are you', 'what are you', 'what can you do', 'help me',
        'what is your name', 'tell me about yourself', 'introduce yourself',
        'how can you help', 'can you help'
    ]

    # Combine for checking
    all_simple_phrases = greetings + simple_questions

    # Check for exact matches or if message starts with these patterns
    for phrase in all_simple_phrases:
        if message_lower == phrase or message_lower.startswith(phrase + " "):
            return True

    # Rule out single words unless they are explicit greetings
    # This prevents single, non-greeting words from being caught
    if len(message_lower.split()) == 1 and message_lower not in greetings:
        return False
        
    # Check if it's just a short casual message without question marks, and not too complex
    # This part should be less aggressive.
    # It now only applies if it's short AND clearly not a question (lacks common question words/punctuation)
    if len(message_content) < 25 and not any(char in message_content for char in ['?', 'what', 'how', 'when', 'where', 'why', 'who', 'tell me']):
        # Further refine: check if it contains any common verbs that might imply a query
        # This is a bit of a heuristic, but helps prevent short commands from being ignored
        common_query_verbs = ['show', 'list', 'find', 'get', 'explain', 'describe']
        if not any(verb in message_lower for verb in common_query_verbs):
            return True

    return False


def extract_name_from_message(message_content):
    """Extract name from messages like 'my name is John' or 'I am John'"""
    message_lower = message_content.lower()

    # Patterns to match name introductions
    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"call me (\w+)",
        r"name's (\w+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            return match.group(1).capitalize()

    return None

# -------------------- Vector Store Loader --------------------
def load_vectorstore():
    try:
        print("🔄 Loading PDFs from directory...")
        loader = DirectoryLoader(
            path="../data",
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()

        if not docs:
            raise Exception("No documents loaded from ../data directory.")

        print(f"📚 Total Documents Loaded: {len(docs)}")

        # Improved text splitter to avoid fragmented content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=100,  # Increased overlap
            separators=["\n\n", "\n", ". ", " ", ""],  # Better separators
            keep_separator=True
        )
        chunks = splitter.split_documents(docs)

        # Filter out chunks that are mostly just names/editors (common in Wikipedia dumps)
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            # Skip chunks that are mostly comma-separated names or very short
            if len(content) > 100 and content.count(',') < len(content.split()) * 0.5:
                filtered_chunks.append(chunk)
        
        chunks = filtered_chunks
        print(f"🔢 Total Chunks (after filtering): {len(chunks)}")

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

        print("✅ Vectorstore ready.")
        return vectordb

    except Exception as e:
        print("❌ Error loading vectorstore:", e)
        return None

# -------------------- Chain Initializer --------------------
def initialize_chains():
    global vectordb, fallback_chain, initialization_complete

    print("🚀 Initializing chains...")
    
    # Initialize fallback chain
    fallback_chain = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Try to initialize vectorstore (not the chain yet - we'll create chains per session)
    vectordb = load_vectorstore()

    if vectordb is not None:
        print("✅ Vectorstore initialized.")
    else:
        print("❌ Vectorstore setup failed. Will use fallback only.")

    initialization_complete = True
    print("✅ Initialization complete.")

def create_retrieval_chain():
    """Create a new retrieval chain instance for each session"""
    if vectordb is None:
        return None
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Use modern approach without deprecated memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        verbose=False,  # Set to False to reduce console output
        return_source_documents=False,
        # Custom prompt for better context handling
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are Luna, a helpful AI assistant. Use the following pieces of context to answer questions when relevant.

If the context doesn't contain information relevant to the user's question, use your general knowledge or politely say you don't have that specific information.

Only use context that is actually relevant to the question. Ignore any irrelevant text like lists of names, editors, or unrelated content.

Context:
{context}

Question: {question}

Remember any previous conversation context when answering. If you know the user's name, use it naturally in conversation.

Answer:""",
                input_variables=["context", "question"]
            )
        }
    )
    
    return chain

# Initialize everything when the module is imported
print("🔥 Starting Luna initialization...")
initialize_chains()

# -------------------- Chainlit Configuration --------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="📋 Ask about documents",
            message="What information is available in the documents?",
            icon="/public/document.svg",
        ),
        cl.Starter(
            label="💬 General chat",
            message="Hello! How can you help me today?",
            icon="/public/chat.svg",
        )
    ]

@cl.on_chat_start
async def on_chat_start():
    print("🟢 Chat started.")
    
    # Display LUNA title as a welcome message
    await cl.Message(
        content="# 🌟 **LUNA** 🌟\n### *Your Intelligent Document Assistant*\n---"
    ).send()
    
    # Wait for initialization to complete if it hasn't already
    while not initialization_complete:
        await asyncio.sleep(0.1)
    
    # Initialize chat history, user name, and simple chat LLM for this session
    cl.user_session.set("chat_history", [])
    cl.user_session.set("user_name", None)
    cl.user_session.set("simple_chat_llm", ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7))
    
    # Create a new retrieval chain for this session
    if vectordb is not None:
        retrieval_chain = create_retrieval_chain()
        cl.user_session.set("chat_chain", retrieval_chain)
        cl.user_session.set("use_retrieval", True)
        await cl.Message("👋 Hi! I'm Luna. I can answer questions from your documents or have a general conversation. What would you like to know?").send()
    else:
        cl.user_session.set("chat_chain", fallback_chain)
        cl.user_session.set("use_retrieval", False)
        await cl.Message("👋 Hi! I'm Luna. The document knowledge base couldn't be loaded, but I'm ready for general conversation. How can I help?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Show thinking message
    thinking_msg = await cl.Message(content="🤔 Thinking...").send()
    
    chain = cl.user_session.get("chat_chain")
    use_retrieval = cl.user_session.get("use_retrieval", False)
    user_name = cl.user_session.get("user_name")
    simple_chat_llm = cl.user_session.get("simple_chat_llm")
    
    # Get conversation history from session
    chat_history = cl.user_session.get("chat_history", [])
    
    # Check if user is introducing their name
    extracted_name = extract_name_from_message(message.content)
    if extracted_name:
        cl.user_session.set("user_name", extracted_name)
        user_name = extracted_name
    
    # Check if this is a greeting or simple question that doesn't need document search
    should_skip_retrieval = is_greeting_or_simple(message.content)
    
    try:
        if use_retrieval and isinstance(chain, ConversationalRetrievalChain) and not should_skip_retrieval:
            print("🔍 Responding with vectorstore retrieval")
            
            # Use invoke with proper parameters (modern approach)
            response = await cl.make_async(chain.invoke)({
                "question": message.content,
                "chat_history": chat_history
            })
            
            # Extract answer from response
            if isinstance(response, dict):
                content = response.get("answer", str(response))
            else:
                content = str(response)
            
            # Update chat history
            chat_history.extend([
                ("human", message.content),
                ("ai", content)
            ])
            cl.user_session.set("chat_history", chat_history)
            
        else:
            print("🟡 Responding with general AI (no retrieval - greeting/simple question)")
            
            # Create a personalized prompt for greetings and simple questions
            if extracted_name:
                personalized_message = f"The user just told me their name is {extracted_name}. Here's their message: {message.content}"
            elif user_name:
                personalized_message = f"The user's name is {user_name}. Here's their message: {message.content}"
            else:
                personalized_message = message.content
            
            # Add context about being Luna
            system_context = "You are Luna, a helpful AI assistant. Be friendly and conversational. "
            if user_name:
                system_context += f"The user's name is {user_name}, so use their name naturally in conversation when appropriate. "
            if extracted_name:
                system_context += f"The user just introduced themselves as {extracted_name}, so acknowledge this warmly. "
            
            full_message = system_context + personalized_message
            
            # Use the dedicated simple chat LLM for greetings and simple conversations
            response = await cl.make_async(simple_chat_llm.invoke)(full_message)
            
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        content = "I apologize, but I encountered an error processing your request. Please try again."
    
    # Remove the thinking message and send the actual response
    await thinking_msg.remove()
    await cl.Message(content=content).send()

# -------------------- Error Handler --------------------
@cl.on_stop
async def on_stop():
    print("🔴 Chat session ended.")

if __name__ == "__main__":
    print("🚀 Starting Luna Chainlit app...")
    cl.run()