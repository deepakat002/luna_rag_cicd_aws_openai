import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()  # Load your OpenAI key etc.

# -------------------- Load Vector Store --------------------
def load_vectorstore():
    try:
        print("🔄 Loading PDF...")
        loader = PyPDFLoader("../data/beagle.pdf")
        docs = loader.load()

        print(f"📄 Loaded {len(docs)} pages")

        print("✂️ Splitting chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        print(f"🔢 Total Chunks: {len(chunks)}")

        print("🧠 Loading Embeddings...")
        embeddings = OpenAIEmbeddings()

        print("💾 Building Chroma Vector DB...")
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        vectordb.persist()

        print("✅ Vectorstore ready.")
        return vectordb

    except Exception as e:
        print("❌ Error loading vectorstore:", e)
        return None

# -------------------- Start Chat --------------------
@cl.on_chat_start
async def on_chat_start():
    print("🟢 Chat started.")
    vectorstore = load_vectorstore()

    if vectorstore is None:
        await cl.Message("⚠️ Could not load knowledge base. Only general chat is available.").send()
        chain = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        cl.user_session.set("chat_chain", chain)
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    cl.user_session.set("chat_chain", chain)
    await cl.Message("👋 Hi! Ask me anything about Beagles 🐶").send()

# -------------------- Handle Messages --------------------
@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("chat_chain")

    if isinstance(chain, ChatOpenAI):
        # fallback mode (no vector)
        print("🟡 Responding without retrieval")
        response = chain.invoke(message.content)
    else:
        # retrieval mode
        print("🔍 Responding with vectorstore")
        response = chain.run(message.content)

    await cl.Message(content=response).send()
