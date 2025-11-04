# app.py
"""
Flight Booking Assistant UI
Streamlit interface for the flight booking chatbot
"""

import streamlit as st
from chatbot import FlightBot, ConversationManager

# Page configuration
st.set_page_config(
    page_title="Flight Booking Assistant",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        padding: 12px;
        margin: 5px 0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        line-height: 1.6;
    }
    
    /* Form styling */
    .stTextInput input, .stSelectbox select, .stRadio label {
        border-radius: 8px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Free text input section - sticky at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 2px solid #e0e0e0;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    /* Add padding to main content to account for fixed input */
    .main .block-container {
        padding-bottom: 120px;
    }
    
    /* Suggested buttons */
    .suggested-btn {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
    }
    
    /* Send button styling */
    [data-testid="stButton"][aria-label="send_button"] button {
        background-color: #667eea;
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    [data-testid="stButton"][aria-label="send_button"] button:hover {
        background-color: #764ba2;
        transform: scale(1.05);
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 25px;
        padding: 12px 20px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'bot' not in st.session_state:
        st.session_state.bot = FlightBot()
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if 'initialized' not in st.session_state:
        st.session_state.conversation.add_message("bot", st.session_state.bot.get_welcome_message())
        st.session_state.initialized = True


def display_chat_history():
    """Display all messages in the conversation"""
    for msg in st.session_state.conversation.get_messages():
        if msg['role'] == 'bot':
            with st.chat_message("assistant", avatar="✈️"):
                st.markdown(msg['content'])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg['content'])


def handle_booking_type_selection(booking_type: str):
    """Handle when user selects a booking type"""
    conv = st.session_state.conversation
    bot = st.session_state.bot
    
    user_msg = f"I need {booking_type.lower()}"
    conv.add_message("user", user_msg)
    conv.add_message("bot", bot.get_booking_type_prompt(booking_type))
    conv.set_booking_type(booking_type)
    conv.set_stage('choose_action')


def handle_action_selection(action: str):
    """Handle when user chooses enquiry or booking"""
    conv = st.session_state.conversation
    
    if action == 'enquiry':
        conv.add_message("user", "I have some questions")
        conv.add_message("bot", "📚 **Please select a topic you'd like to know more about:**")
        conv.set_stage('show_topics')
    else:
        conv.add_message("user", "I want to proceed with booking")
        conv.add_message("bot", "📝 **Excellent! Please provide your booking details below:**")
        conv.set_stage('collect_booking')


def handle_topic_selection(topic: dict):
    """Handle when user selects a topic"""
    conv = st.session_state.conversation
    conv.add_message("user", f"Tell me about {topic['topic']}")
    conv.add_message("bot", f"**{topic['topic']}** - Select a question:")
    conv.set_current_topic(topic)
    conv.set_stage('show_questions')


def handle_question_selection(question: dict):
    """Handle when user selects a question"""
    conv = st.session_state.conversation
    conv.add_message("user", question['question'])
    conv.add_message("bot", f"**Answer:** {question['answer']}")


def send_free_text_query(user_text: str):
    """
    Send free-text query to backend RAG system and update conversation
    """
    conv = st.session_state.conversation
    bot = st.session_state.bot
    
    # Get current conversation ID (if exists)
    conversation_id = conv.get_conversation_id()
    
    # Query backend
    response_data = bot.query_backend(user_text, conversation_id)
    
    # Update conversation ID
    conv.set_conversation_id(response_data.get("conversation_id"))
    
    # Add messages to conversation
    conv.add_message("user", user_text)
    conv.add_message("bot", response_data.get("answer", "No response received."))
    
    # Store suggested buttons if any
    suggested_buttons = response_data.get("suggested_buttons", [])
    conv.set_suggested_buttons(suggested_buttons)


def display_booking_form():
    """Display the booking details form"""
    bot = st.session_state.bot
    conv = st.session_state.conversation
    
    with st.form("booking_form", clear_on_submit=False):
        st.markdown("### ✈️ Flight Booking Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            from_city = st.text_input(
                "📍 From (City & Airport)",
                placeholder="e.g., New York JFK",
                help="Enter departure city and airport code"
            )
            passengers = st.text_input(
                "👥 Passengers",
                placeholder="e.g., 2 Adults, 1 Kid (5 years)",
                help="Specify number and age of passengers"
            )
            travel_date = st.date_input("📅 Travel Date")
            time_pref = st.selectbox(
                "⏰ Preferred Time",
                ["Morning", "Afternoon", "Evening"]
            )
        
        with col2:
            to_city = st.text_input(
                "📍 To (City & Airport)",
                placeholder="e.g., London Heathrow",
                help="Enter destination city and airport code"
            )
            trip_type = st.radio(
                "🔄 Trip Type",
                ["One-way", "Round-trip"],
                horizontal=True
            )
            flexible_dates = st.radio(
                "📆 Flexible Dates?",
                ["Yes", "No"],
                horizontal=True
            )
            flight_type = st.radio(
                "✈️ Flight Preference",
                ["Direct flight only", "Stopovers acceptable"]
            )
        
        airline = st.text_input(
            "🛫 Preferred Airlines (Optional)",
            placeholder="e.g., Emirates, Qatar Airways"
        )
        
        submitted = st.form_submit_button("✅ Submit Booking Details", use_container_width=True)
        
        if submitted:
            details = {
                'from': from_city,
                'to': to_city,
                'passengers': passengers,
                'date': str(travel_date),
                'trip_type': trip_type,
                'time_preference': time_pref,
                'flexible_dates': flexible_dates,
                'preferred_airline': airline,
                'flight_type': flight_type
            }
            
            is_valid, missing = bot.validate_booking_details(details)
            
            if is_valid:
                conv.update_booking_details(details)
                conv.add_message("user", "Here are my booking details")
                conv.add_message("bot", bot.format_booking_summary(details))
                conv.add_message("bot", bot.get_completion_message(conv.get_booking_type()))
                conv.set_stage('booking_complete')
                st.rerun()
            else:
                st.error(f"⚠️ Please fill in: {', '.join(missing)}")


def main():
    """Main application function"""
    init_session_state()
    
    bot = st.session_state.bot
    conv = st.session_state.conversation
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Booking Assistant</h1>
        <p>Your personal travel booking companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history()
    
    st.markdown("---")
    
    # Current stage handling - MIDDLE SECTION
    stage = conv.get_stage()
    
    # Render suggested buttons from backend (if any) - TOP OF MIDDLE SECTION
    suggested_buttons = conv.get_suggested_buttons()
    if suggested_buttons:
        st.markdown("#### 💡 Suggested next steps")
        cols = st.columns(min(len(suggested_buttons), 3))  # Max 3 columns
        for i, btn in enumerate(suggested_buttons):
            col_idx = i % len(cols)
            with cols[col_idx]:
                if st.button(btn.get("label", ""), key=f"suggest_{i}_{btn.get('value', '')}"):
                    send_free_text_query(btn.get("label", ""))
                    conv.clear_suggested_buttons()
                    st.rerun()
        st.markdown("---")
    
    st.markdown("### 🎯 Guided Booking Options")
    
    # Stage: Choose booking type (Visa/Travel)
    if stage == 'welcome':
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Visa Process Booking", key="visa_btn"):
                handle_booking_type_selection("Flight Booking for Visa Process")
                st.rerun()
        with col2:
            if st.button("🌍 Actual Travel Booking", key="travel_btn"):
                handle_booking_type_selection("Flight Booking for Travelling Purpose")
                st.rerun()
    
    # Stage: Choose action (Enquiry/Booking)
    elif stage == 'choose_action':
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❓ I have questions", key="enquiry_btn"):
                handle_action_selection('enquiry')
                st.rerun()
        with col2:
            if st.button("✅ Proceed with Booking", key="booking_btn"):
                handle_action_selection('booking')
                st.rerun()
    
    # Stage: Show topics
    elif stage == 'show_topics':
        topics = bot.get_topics(conv.get_booking_type())
        
        for topic in topics:
            if st.button(f"📌 {topic['topic']}", key=f"topic_{topic['topic']}"):
                handle_topic_selection(topic)
                st.rerun()
        
        st.markdown("---")
        if st.button("🔙 Start Over", key="restart_from_topics"):
            conv.reset()
            conv.add_message("bot", bot.get_welcome_message())
            st.rerun()
    
    # Stage: Show questions for selected topic
    elif stage == 'show_questions':
        current_topic = conv.get_current_topic()
        
        if current_topic:
            for q in current_topic['questions']:
                if st.button(q['question'], key=f"q_{q['question'][:30]}"):
                    handle_question_selection(q)
                    st.rerun()
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 Back to Topics", key="back_to_topics"):
                conv.set_stage('show_topics')
                st.rerun()
        with col2:
            if st.button("🏠 Start Over", key="restart_from_questions"):
                conv.reset()
                conv.add_message("bot", bot.get_welcome_message())
                st.rerun()
    
    # Stage: Collect booking details
    elif stage == 'collect_booking':
        display_booking_form()
        
        st.markdown("---")
        if st.button("🔙 Go Back", key="back_from_booking"):
            conv.set_stage('choose_action')
            st.rerun()
    
    # Stage: Booking complete
    elif stage == 'booking_complete':
        st.balloons()
        
        if st.button("🏠 Start New Booking", key="new_booking"):
            conv.reset()
            conv.add_message("bot", bot.get_welcome_message())
            st.rerun()
    
    # Add spacing before input
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    
    # BOTTOM INPUT SECTION - Fixed at bottom
    st.markdown("---")
    
    # Create input area with custom styling
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 15px 20px;
        border-top: 1px solid #e0e0e0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        z-index: 1000;
    ">
    </div>
    """, unsafe_allow_html=True)
    
    # Input container
    input_container = st.container()
    with input_container:
        input_col1, input_col2 = st.columns([9, 1])
        
        with input_col1:
            user_input = st.text_input(
                "Message",
                key="free_text_input",
                placeholder="💬 Ask me anything about flights, bookings, or travel...",
                label_visibility="collapsed"
            )
        
        with input_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button
            send_button = st.button("➤", key="send_text_button", help="Send message", use_container_width=True)
    
    # Handle send action
    if send_button and user_input:
        send_free_text_query(user_input)
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Backend Status")
        # Check backend connection
        try:
            import requests
            response = requests.get(f"{bot.backend_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected to RAG backend")
            else:
                st.warning("⚠️ Backend not responding")
        except:
            st.error("❌ Backend offline")
        
        st.markdown("---")
        st.markdown("### 📞 Contact & Help")
        st.info("🕐 **Available 24/7**\n\nOur agents are ready to assist you with any questions about your flight booking!")
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Have your passport details ready
        - Check visa requirements
        - Book in advance for better prices
        - Compare airlines for best deals
        """)
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", key="sidebar_reset"):
            conv.reset()
            conv.add_message("bot", bot.get_welcome_message())
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Current Stage:** `{}`".format(stage))
        if conv.get_booking_type():
            st.markdown("**Booking Type:** `{}`".format(conv.get_booking_type().split(' for ')[-1]))
        if conv.get_conversation_id():
            st.markdown("**Session ID:** `{}`".format(conv.get_conversation_id()[:8]))


if __name__ == "__main__":
    main()