# chatbot.py
"""
Flight Booking Chatbot Logic
Handles all bot operations, data processing, and conversation flow
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class FlightBot:
    """Main chatbot class that handles all flight booking logic"""
    
    def __init__(self, data_file: str = 'flight_bot_data.json'):
        """Initialize the bot with data from JSON file"""
        self.data = self._load_data(data_file)
        self.categories = self.data['flight_bot']['categories']
        
    def _load_data(self, file_path: str) -> Dict:
        """Load flight bot data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Data file '{file_path}' not found!")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in '{file_path}'")
    
    def get_booking_types(self) -> List[str]:
        """Get available booking types (Visa/Travel)"""
        return [category['section'] for category in self.categories]
    
    def get_topics(self, booking_type: str) -> List[Dict]:
        """Get all topics for a specific booking type"""
        for category in self.categories:
            if category['section'] == booking_type:
                for option in category['options']:
                    if option['type'] == 'Enquiry':
                        return option['topics']
        return []
    
    def get_questions_for_topic(self, booking_type: str, topic_name: str) -> List[Dict]:
        """Get all questions for a specific topic"""
        topics = self.get_topics(booking_type)
        for topic in topics:
            if topic['topic'] == topic_name:
                return topic['questions']
        return []
    
    def find_answer(self, booking_type: str, question_text: str) -> Optional[str]:
        """Search for an answer based on question text"""
        topics = self.get_topics(booking_type)
        question_lower = question_text.lower()
        
        for topic in topics:
            for q in topic['questions']:
                if question_lower in q['question'].lower() or q['question'].lower() in question_lower:
                    return q['answer']
        return None
    
    def get_required_booking_details(self, booking_type: str) -> List[str]:
        """Get list of required fields for booking"""
        for category in self.categories:
            if category['section'] == booking_type:
                for option in category['options']:
                    if option['type'] == 'Booking':
                        return option['required_details']
        return []
    
    def get_next_steps(self, booking_type: str) -> List[str]:
        """Get next steps after booking"""
        for category in self.categories:
            if category['section'] == booking_type:
                for option in category['options']:
                    if option['type'] == 'Booking':
                        return option.get('next_steps', [])
        return []
    
    def validate_booking_details(self, details: Dict) -> Tuple[bool, List[str]]:
        """Validate booking details and return missing fields"""
        required = ['from', 'to', 'passengers', 'date']
        missing = [field for field in required if not details.get(field)]
        return len(missing) == 0, missing
    
    def format_booking_summary(self, details: Dict) -> str:
        """Format booking details into a readable summary"""
        summary = "**✅ Booking Details Received:**\n\n"
        summary += f"📍 **From:** {details.get('from', 'N/A')}\n"
        summary += f"📍 **To:** {details.get('to', 'N/A')}\n"
        summary += f"👥 **Passengers:** {details.get('passengers', 'N/A')}\n"
        summary += f"📅 **Date:** {details.get('date', 'N/A')}\n"
        summary += f"🔄 **Trip Type:** {details.get('trip_type', 'N/A')}\n"
        summary += f"⏰ **Preferred Time:** {details.get('time_preference', 'N/A')}\n"
        summary += f"📆 **Flexible Dates:** {details.get('flexible_dates', 'N/A')}\n"
        
        if details.get('preferred_airline'):
            summary += f"✈️ **Preferred Airline:** {details['preferred_airline']}\n"
        
        summary += f"🛫 **Flight Type:** {details.get('flight_type', 'N/A')}\n"
        return summary
    
    def get_welcome_message(self) -> str:
        """Get the initial welcome message"""
        return "👋 **Welcome to Flight Booking Assistant!**\n\nI can help you with:\n- ✈️ Flight reservations for visa applications\n- 🌍 Actual flight bookings for travel\n\nHow can I assist you today?"
    
    def get_booking_type_prompt(self, booking_type: str) -> str:
        """Get prompt after selecting booking type"""
        if "Visa" in booking_type:
            return "Great! I can help with **flight reservations for visa applications**. Would you like to:\n\n1️⃣ Ask questions about our visa flight booking service\n2️⃣ Proceed with booking"
        else:
            return "Perfect! I can help you **book your flight tickets**. Would you like to:\n\n1️⃣ Ask questions about flight bookings and travel\n2️⃣ Proceed with booking"
    
    def get_completion_message(self, booking_type: str) -> str:
        """Get message after booking completion"""
        next_steps = self.get_next_steps(booking_type)
        message = "🎉 **Thank you for providing your details!**\n\n"
        
        if next_steps:
            message += "**Next Steps:**\n"
            for i, step in enumerate(next_steps, 1):
                message += f"{i}. {step}\n"
        else:
            message += "Our agent will contact you shortly to confirm your booking and send the payment link."
        
        return message


class ConversationManager:
    """Manages conversation state and message history"""
    
    def __init__(self):
        self.messages = []
        self.stage = 'welcome'
        self.booking_type = None
        self.current_topic = None
        self.booking_details = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def get_messages(self) -> List[Dict]:
        """Get all messages in the conversation"""
        return self.messages
    
    def reset(self):
        """Reset conversation to initial state"""
        self.messages = []
        self.stage = 'welcome'
        self.booking_type = None
        self.current_topic = None
        self.booking_details = {}
    
    def set_stage(self, stage: str):
        """Update conversation stage"""
        self.stage = stage
    
    def get_stage(self) -> str:
        """Get current conversation stage"""
        return self.stage
    
    def set_booking_type(self, booking_type: str):
        """Set the selected booking type"""
        self.booking_type = booking_type
    
    def get_booking_type(self) -> Optional[str]:
        """Get the current booking type"""
        return self.booking_type
    
    def set_current_topic(self, topic: Dict):
        """Set the currently selected topic"""
        self.current_topic = topic
    
    def get_current_topic(self) -> Optional[Dict]:
        """Get the current topic"""
        return self.current_topic
    
    def update_booking_details(self, details: Dict):
        """Update booking details"""
        self.booking_details.update(details)
    
    def get_booking_details(self) -> Dict:
        """Get current booking details"""
        return self.booking_details