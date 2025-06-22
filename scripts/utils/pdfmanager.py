### importing required libraries
from datetime import datetime
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.schema import Document
from typing import List 
import traceback
from utils.lunaConfig import LunaConfig

from utils.loggerSetup import get_logger

# Get the logger
logger = get_logger("pdfmanager", "luna.log", console_output=False)


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