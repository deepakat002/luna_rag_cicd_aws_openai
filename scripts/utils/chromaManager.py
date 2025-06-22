
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
from dotenv import load_dotenv
load_dotenv()


# Get the logger
logger = get_logger("chromamanager", "luna.log",console_output=os.getenv('CMD_OUTPUT') == 't')

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
                logger.error("No documents provided for vector store creation")
                return False
            
            logger.info("Creating ChromaDB vector store...")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name="luna_dogs"
            )
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store"""
        try:
            if not self.persist_directory.exists():
                logger.warning("Vector store directory doesn't exist")
                return False
            
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
                logger.info(f"Loaded vector store with {count} documents")
                return True
            else:
                logger.warning("Vector store is empty")
                return False
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_retriever(self):
        """Get retriever for the vector store"""
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return None
        
        logger.info(f"Creating retriever with top_k={LunaConfig.TOP_K}")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": LunaConfig.TOP_K}
        )