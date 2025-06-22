
### importing required libraries
from datetime import datetime
from pathlib import Path
import os, traceback
from langchain.schema import Document
from typing import List 
from langchain_openai import OpenAIEmbeddings
from utils.loggerSetup import get_logger
from langchain_community.vectorstores import Chroma
from utils.lunaConfig import LunaConfig



# Get the logger
logger = get_logger("chromamanager", "luna.log", console_output=False)

class ChromaManager:
    """Manages ChromaDB operations"""
    
    def __init__(self,persist_directory: Path):
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