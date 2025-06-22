import re
import random
from utils.loggerSetup import get_logger

# Get the logger
logger = get_logger("pdfmanager", "luna.log", console_output=False)


class GreetingHandler:
    """Handles greeting detection and responses"""
    
    def __init__(self):
        # Define greeting patterns
        self.greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening|greetings?)\.?$',
            r'^(hi|hello|hey)\s+(there|luna|dog|bot)\.?$',
            r'^(what\'s up|whats up|sup|yo)\.?$',
            r'^(good day|howdy)\.?$'
        ]
        
        # Define how are you patterns
        self.how_are_you_patterns = [
            r'^(how are you|how\'re you|how r u|how do you do)\??\.?$',
            r'^(how are you doing|how\'s it going|how is it going)\??\.?$',
            r'^(how are things|how\'s everything|how is everything)\??\.?$',
            r'^(are you (okay|ok|good|well))\??\.?$'
        ]
        
        # Define combined greeting + how are you patterns
        self.greeting_with_inquiry_patterns = [
            r'^(hi|hello|hey),?\s+(how are you|how\'re you|how r u)\??\.?$',
            r'^(hi|hello|hey)\s+(there|luna),?\s+(how are you|how\'s it going)\??\.?$'
        ]
        
        # Greeting responses
        self.greeting_responses = [
            "Hello! 🐕 I'm Luna, your friendly dog expert! How can I help you with your dog questions today?",
            "Hi there! 🐾 I'm Luna, and I love talking about dogs! What would you like to know?",
            "Hey! 🐶 Luna here, ready to help with all your dog-related questions!",
            "Hello! 🦴 I'm Luna, your dog expert companion. What can I help you learn about dogs today?"
        ]
        
        # How are you responses
        self.how_are_you_responses = [
            "I'm doing great, thank you for asking! 🐕 I'm always excited to help with dog questions. How can I assist you today?",
            "I'm wonderful! 🐾 I love helping people learn about dogs. What would you like to know about our furry friends?",
            "I'm fantastic! 🐶 Always ready to share dog knowledge. What dog topic interests you most?",
            "I'm doing well, thanks! 🦴 I'm here and ready to help with any dog questions you have!"
        ]
        
        # Combined greeting responses
        self.combined_responses = [
            "Hello! 🐕 I'm doing great, thank you! I'm Luna, your dog expert assistant. What would you like to know about dogs today?",
            "Hi there! 🐾 I'm wonderful, thanks for asking! I'm here to help with all your dog questions. How can I assist you?",
            "Hey! 🐶 I'm fantastic! I'm Luna, and I love helping people learn about dogs. What can I help you with today?"
        ]
    
    def is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        message_clean = message.lower().strip()
        
        # Check all greeting patterns
        for pattern in (self.greeting_patterns + self.how_are_you_patterns + 
                       self.greeting_with_inquiry_patterns):
            if re.match(pattern, message_clean):
                return True
        
        return False
    
    def get_greeting_response(self, message: str) -> str:
        """Get appropriate greeting response"""
        message_clean = message.lower().strip()
        
        # Check for combined greeting + inquiry
        for pattern in self.greeting_with_inquiry_patterns:
            if re.match(pattern, message_clean):
                return random.choice(self.combined_responses)
        
        # Check for how are you
        for pattern in self.how_are_you_patterns:
            if re.match(pattern, message_clean):
                return random.choice(self.how_are_you_responses)
        
        # Check for simple greeting
        for pattern in self.greeting_patterns:
            if re.match(pattern, message_clean):
                return random.choice(self.greeting_responses)
        
        # Fallback (shouldn't reach here if is_greeting returned True)
        return random.choice(self.greeting_responses)






