import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from database import UserDatabase
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.getenv("API_KEY")
def exercise1():
    # Initialize the user database
    user_db = UserDatabase()
    
    # Set the page configuration to use the full width of the page
    #st.set_page_config(layout="wide")

    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    
    # HEADER
    st.markdown("<h1 style='text-align: center; font-size: 20px;'>AI BOOTCAMP Lab</h1>", unsafe_allow_html=True)
    
    # Handle authentication if not already authenticated
    if not st.session_state.authenticated:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            login_button = st.button("Login")
            
            if login_button:
                if user_db.verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}! Your responses will be saved.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        
        with col2:
            st.subheader("Continue as Guest")
            guest_button = st.button("Continue without logging in")
            
            if guest_button:
                st.session_state.authenticated = True
                st.session_state.username = "guest"
                st.info("You're continuing as a guest. Your responses won't be saved.")
                st.rerun()
    
    # Only show the main application if authenticated
    if st.session_state.authenticated:
        # Display login status
        if st.session_state.username != "guest":
            st.sidebar.success(f"Logged in as: {st.session_state.username}")
        else:
            st.sidebar.info("Using as guest (responses won't be saved)")
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        # CREATE LLM model  
        llm_models = [
            "llama3-70b-8192", "gemma2-9b-it", "mixtral-8x7b-32768", "qwen-2.5-32b","deepseek-r1-distill-qwen-32b","allam-2-7b"
        ]

        # Initialize session state for chat history and selected models
        if "chatHist" not in st.session_state:
            st.session_state.chatHist = {model: [{"role": "system", "content": "You are a helpful assistant."}] for model in llm_models}
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = llm_models[:1]  # Default to the first model
        if "temperature" not in st.session_state:
            st.session_state.temperature = 1.0
        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = 300  
        if "selected_scenario" not in st.session_state:
            st.session_state.selected_scenario = None
        if "uploaded_file_content" not in st.session_state:
            st.session_state.uploaded_file_content = None
        if "model_responses" not in st.session_state:
            st.session_state.model_responses = {}

        # Create two tabs
        tab1, tab2 = st.tabs(["Model Selection", "Compare Models"])

        with tab1:
            # Header with logo and title
            colHeader = st.columns([2, 2])  # Add a third column with width 3
            with colHeader[0]:
                st.header("LLM Performance Comparison")
            with colHeader[1]:
                st.markdown(" Upload Document")
                uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "docx"])
                if uploaded_file is not None:
                    st.write("Filename:", uploaded_file.name)
                    # Read the uploaded file content
                    st.session_state.uploaded_file_content = uploaded_file.read().decode("utf-8", errors="ignore")

                    # Save the uploaded file to a folder
                    if not os.path.exists("Exercise/uploads"):
                        os.makedirs("Exercise/uploads")
                    with open(os.path.join("Exercise/uploads", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                else:
                    st.write("No document uploaded. You can ask questions without a document.")

            # Model Settings in a popover
            with st.popover("⚙️ Model Settings"):
                # Temperature Slider
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls the randomness of the model's responses. Lower values make the model more deterministic."
                )

                # Max Tokens Slider
                st.session_state.max_tokens = st.slider(
                    "Max Tokens",
                    min_value=200,
                    max_value=8192,  # Set max to 8192
                    value=st.session_state.max_tokens,
                    step=100,
                    help="Controls the maximum number of tokens the model will generate."
                )

                # Maximum Models to Compare
                max_models_to_compare = st.slider(
                    "Maximum Models to Compare",
                    min_value=1,
                    max_value=4,
                    value=4,
                    step=1,
                    help="Limit the number of models that can be compared for better viewing experience."
                )

                # Maximum Scenarios Allowed (New Setting)
                max_scenarios_to_select = st.slider(
                    "Maximum Scenarios to Select",
                    min_value=1,
                    max_value=3,
                    value=1,
                    step=1,
                    help="Limit the number of scenarios that can be selected."
                )

            # MULTI-SELECT DROPDOWN FOR MODEL SELECTION (SMALLER SIZE)
            st.session_state.selected_models = st.multiselect(
                "Choose models to compare (minimum 1):",
                llm_models,
                default=st.session_state.selected_models,  # Default to the first model
                key="model_selector"
            )

            # Limit the number of selected models based on the slider value
            if len(st.session_state.selected_models) > max_models_to_compare:
                st.warning(f"You can compare a maximum of {max_models_to_compare} models. Please deselect some models.")
                st.session_state.selected_models = st.session_state.selected_models[:max_models_to_compare]

            # Scenario Selection Header
            st.write("📌 Select the Scenario (You can tick scenarios to pass to models):")

            # Define scenarios
            scenarios = {
                "📱 Mobile Security Policy": {
                    "question": '''A university IT administrator discovers that a faculty member has installed a custom operating system on their university-provided smartphone. According to best practices aligned with WPI's Mobile Device Management Policy, what should happen next?''',
                    "options": [
                        " The device should be remotely wiped immediately",
                        " The device should be confiscated and recycled",
                        " The OS must be restored to manufacturer specifications",
                        " The faculty member should be required to purchase the device"
                    ]
                },
                "📂 Data Ownership Scenario": {
                    "question": '''A research professor who used their personal smartphone to access university email for 5 years is leaving for a position at another institution. The phone contains important research communications. Based on standard MDM policies similar to WPI's, who has rights to the email data on the device?''',
                    "options": [
                        " The professor retains full ownership of all data",
                        " The emails belong to the university but the device belongs to the professor",
                        " The university has rights to wipe university-related data only",
                        " Ownership is determined by the department chair"
                    ]
                },
                "🔒 Security Breach Protocol": {
                    "question": '''A university employee reports their smartphone containing university email was stolen while traveling. According to principles in policies like WPI's, what is the correct sequence of actions?''',
                    "options": [
                        " Report to local police, then IT security, wait for instructions",
                        " Contact IT security immediately, remote wipe will be initiated",
                        " Purchase a replacement device, restore from backup, then report",
                        " Change passwords for all accounts, then report to department head"
                    ]
                }
            }

            # Display all scenarios (Always Visible)
            selected_scenarios = []
            for index, (scenario, content) in enumerate(scenarios.items()):
                # Checkbox for scenario selection
                selected = st.checkbox(f"**{scenario}**", key=f"select_{index}")
                
                # Store selected scenarios
                if selected:
                    selected_scenarios.append(scenario)

                # Always Display Scenario Details
                st.markdown(f"**Question:** {content['question']}")
                
                # Display options as bullet points
                for option in content["options"]:
                    st.markdown(f"- {option}")
                
                # Add a divider (except after the last scenario)
                if index < len(scenarios) - 1:
                    st.divider()

            # Check if the user selected more scenarios than allowed
            if len(selected_scenarios) > max_scenarios_to_select:
                st.warning(
                    f"⚠️ You have selected {len(selected_scenarios)} scenarios, "
                    f"but the maximum allowed is {max_scenarios_to_select}. \n\n"
                    "**→ Either:** \n"
                    "1. Deselect extra scenarios to fit within the limit. \n"
                    "2. **Increase the 'Max Scenarios to Select' in Model Settings.**"
                )
                # Automatically limit selection to allowed number
                selected_scenarios = selected_scenarios[:max_scenarios_to_select]

            # Store final selected scenarios in session state
            st.session_state.selected_scenarios = selected_scenarios

            # Ensure one selected scenario is loaded as prompt in tab 2
            if selected_scenarios:
                st.session_state.selected_scenario = scenarios[selected_scenarios[0]]["question"]
            else:
                st.session_state.selected_scenario = None  # Reset if none selected

        with tab2:
            # Display Warning if No Model Selected
            if not st.session_state.selected_models:
                st.warning("Please select models and settings in the 'Model Selection' tab first.")
            else:
                # Title
                st.subheader("Compare Model Responses Side by Side")

                # Dynamically Create Columns for Each Model
                cols = st.columns(len(st.session_state.selected_models))

                # Add a "Compare Now" button
                if st.button("Compare Now"):
                    # Reset model responses
                    st.session_state.model_responses = {}
                    
                    # Loop through models and assign them to columns
                    for i, model in enumerate(st.session_state.selected_models):
                        with cols[i]:  # Assign each model to a column
                            st.markdown(f"### {model}")  # Model Name in Bold

                            # Initialize the Groq client
                            client = Groq(api_key=GROQ_API_KEY)

                            # Prepare the initial prompt (uploaded document)
                            if st.session_state.uploaded_file_content:
                                initial_prompt = st.session_state.uploaded_file_content
                                st.session_state.chatHist[model].append({"role": "user", "content": initial_prompt})
                                with st.chat_message("user"):
                                    st.markdown(initial_prompt)

                            # Handle Scenario Input
                            if st.session_state.selected_scenario:
                                st.session_state.chatHist[model].append({"role": "user", "content": st.session_state.selected_scenario})
                                with st.chat_message("user"):
                                    st.markdown(st.session_state.selected_scenario)

                                # Fetch response from Groq API
                                try:
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=st.session_state.chatHist[model],
                                        max_tokens=st.session_state.max_tokens,
                                        temperature=st.session_state.temperature
                                    )
                                    assistant_response = response.choices[0].message.content

                                    # Store the response
                                    st.session_state.model_responses[model] = assistant_response

                                    # Append the response to the chat history
                                    st.session_state.chatHist[model].append({"role": "assistant", "content": assistant_response})
                                    with st.chat_message("assistant"):
                                        st.markdown(assistant_response)
                                except Exception as e:
                                    st.error(f"Error: {e}", icon="🚨")
                                    st.session_state.model_responses[model] = f"Error: {e}"

                    # Save responses to CSV file if the user is authenticated and not a guest
                    if st.session_state.username != "guest" and st.session_state.model_responses and st.session_state.selected_scenario:
                        saved_file = user_db.save_user_interaction(
                            st.session_state.username,
                            st.session_state.selected_scenario,
                            st.session_state.model_responses
                        )
                        if saved_file:
                            st.sidebar.success(f"Responses saved to {saved_file}")
                        else:
                            st.sidebar.error("Failed to save responses")

#if __name__ == "__main__":
   # exercise1()
