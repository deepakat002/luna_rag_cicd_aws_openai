
### importing required libraries
from datetime import datetime
from pathlib import Path
from typing import List 
import traceback, os
from utils.lunaConfig import LunaConfig
from typing import List, Dict

from utils.loggerSetup import get_logger
from dotenv import load_dotenv
load_dotenv()

# Get the logger
logger = get_logger("historymanager", "luna.log", console_output=os.getenv('CMD_OUTPUT') == 't')



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
                
            logger.info(f"Saved {role} message to history for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        """Load chat history for a session"""
        try:
            file_path = self.get_session_file(session_id)
            if not file_path.exists():
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
            logger.info(f"Loaded {len(recent_history)} recent messages from history for session {session_id}")
            return recent_history
            
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []