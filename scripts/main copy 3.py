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

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

load_dotenv()

# -------------------- Globals --------------------
vectordb = None
fallback_chain = None
initialization_complete = False

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

Remember any previous conversation context when answering.

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

# -------------------- Chainlit Hooks --------------------
@cl.on_chat_start
async def on_chat_start():
    print("🟢 Chat started.")
    
    # Wait for initialization to complete if it hasn't already
    while not initialization_complete:
        await asyncio.sleep(0.1)
    
    # Initialize chat history for this session
    cl.user_session.set("chat_history", [])
    
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
    chain = cl.user_session.get("chat_chain")
    use_retrieval = cl.user_session.get("use_retrieval", False)
    
    # Get conversation history from session
    chat_history = cl.user_session.get("chat_history", [])

    try:
        if use_retrieval and isinstance(chain, ConversationalRetrievalChain):
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
            print("🟡 Responding with general AI (no retrieval)")
            response = await cl.make_async(chain.invoke)(message.content)
            
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

    await cl.Message(content=content).send()

# -------------------- Error Handler --------------------
@cl.on_stop
async def on_stop():
    print("🔴 Chat session ended.")

if __name__ == "__main__":
    print("🚀 Starting Luna Chainlit app...")
    cl.run()