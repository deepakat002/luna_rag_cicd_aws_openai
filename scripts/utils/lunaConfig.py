from pathlib import Path


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
