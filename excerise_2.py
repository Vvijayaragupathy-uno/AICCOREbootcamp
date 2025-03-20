from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from groq import Groq
import os
from database import UserDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.getenv("API_KEY")


def exercise2():
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
    st.markdown("<h1 style='text-align: center; font-size: 20px;'>AI BOOTCAMP Lab - Exercise 2</h1>", unsafe_allow_html=True)
    
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
        
        # Define LLM models
        llm_models = [
            "llama3-70b-8192", "gemma2-9b-it", "mixtral-8x7b-32768", "qwen-2.5-32b",
            "deepseek-r1-distill-qwen-32b", "allam-2-7b"
        ]

        # Define prompt techniques
        prompt_techniques = {
            "Zero Shot": "Answer the following question without any examples.",
            "Few Shot": {
                "📱 Mobile Security Policy": [
                    '''Here are some examples to guide your response: 
                    Question: I have a Nokia 2017 that I want to use as a personal device and connect to university devices. Would using this device violate the mobile-device policy? 
                    Answer: Yes.''',

                    '''Question: My department plans to use personal mobile devices for accessing confidential university data. These devices are running outdated software and do not support the latest security updates. Would using these devices violate the mobile-device policy? 
                    Answer: Yes.''',

                    '''Question: I want to use my personal tablet to access university resources. The device has up-to-date security patches, strong passwords, and mobile-device management installed as per policy. Would using this device violate the mobile-device policy? 
                    Answer: No.''',

                    "Now, answer the following question."
                ],
                "📂 Data Ownership Scenario": [
                    '''Here are some examples to guide your response: 
                    Question 1: I'd like my department to retain confidential university documents on their personal devices for one year after projects end. Is this permitted? 
                    Answer: No.''',

                    '''Question 2: Our research team wants to store project data collected through university apps on their devices for six months. Would this comply with the policy? 
                    Answer: No.''',

                    '''Question 3: We need to keep a temporary copy of publicly available university policies on personal devices for reference during a conference. Would this violate the data-ownership policy? 
                    Answer: No.''',

                    "Now, answer the following question."
                ],
                "🔒 Security Breach Protocol": [
                    '''Here are some examples to guide your response: 
                    Question: An employee's mobile device starts exhibiting unusual behavior. The battery drains quickly even when not in use. The device connects repeatedly to IP addresses in countries where the company doesn't operate. Apps the employee never installed appear in the app drawer, including one that requests unusual permissions. Does this device show clear evidence of being compromised? 
                    Answer: Yes.''',

                    '''Question: An employee's mobile device starts exhibiting unusual behavior. The battery depletes unusually fast. Network logs show the device connecting to unrecognized servers with suspicious domain names. The security scan reveals hidden processes running in the background and unauthorized modifications to system files. Several unfamiliar applications with administrator privileges appear to be installed without the employee's knowledge. Does this device show clear evidence of being compromised? 
                    Answer: Yes.''',

                    '''Question: A user's mobile device suddenly runs slower, but no unexpected network activity, unauthorized apps, or security alerts have been observed. Could this be clear evidence of being compromised? 
                    Answer: No.''',

                    "Now, answer the following question."
                ]
            },
            "Role Based": "You are an expert in cybersecurity and mobile device management policies. You are a university IT compliance officer evaluating device requests. Please give me only yes or no as the answer to the following question.",
            "Chain of Thought": "Think step by step to solve the following problem. Break down your reasoning into clear logical steps."
        }
        # Define scenarios
        scenarios = {
                "📱 Mobile Security Policy": {
                    "question": '''Question: I have a Nokia 2017 that I want to use as a personal device and connect to university devices. Would using this device violate the mobile-device policy? Please give me a yes or no answer.''',
                    
                    
                },
                "📂 Data Ownership Scenario": {
                    "question": '''I want to allow my team to store work emails on their devices for up to three months to reference older communications as per Mobile device management policy. Can you please give yes or no answer?''',
                    
                },
                "🔒 Security Breach Protocol": {
                    "question": '''An employee’s mobile device starts exhibiting. The device shows increased battery usage. It initiates several connections to various IP addresses. Some unfamiliar apps appear to be installed. Does this device show clear evidence of being compromised? Can you give the response as only no or yes?''',
                    
                }
            }


        # Initialize session state variables
        if "chatHist" not in st.session_state:
            st.session_state.chatHist = {model: {technique: [{"role": "system", "content": "You are a helpful assistant."}] 
                                                for technique in prompt_techniques} 
                                        for model in llm_models}
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = llm_models[0]  # Default to the first model
        if "selected_prompt_techniques" not in st.session_state:
            st.session_state.selected_prompt_techniques = []
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
        

        # Create four tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Prompt Selection", "Model selection", " Analyis and visualization", "Tab 4"])

        with tab1:
            # Header with logo and title
            colHeader = st.columns([2, 2])
            with colHeader[0]:
                st.header("Prompt Engineering Comparison")
            with colHeader[1]:
                st.markdown("Upload Document")
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
                    max_value=8192,
                    value=st.session_state.max_tokens,
                    step=100,
                    help="Controls the maximum number of tokens the model will generate."
                )

                # Maximum Prompt Techniques to Compare
                max_techniques_to_compare = st.slider(
                    "Maximum Prompt Techniques to Compare",
                    min_value=1,
                    max_value=4,
                    value=4,
                    step=1,
                    help="Limit the number of prompt techniques that can be compared for better viewing experience."
                )

            # DROPDOWN FOR SINGLE MODEL SELECTION
            st.session_state.selected_model = st.selectbox(
                "Choose a model:",
                llm_models,
                index=llm_models.index(st.session_state.selected_model) if st.session_state.selected_model in llm_models else 0,
                key="model_selector"
            )

            # MULTI-SELECT FOR PROMPT TECHNIQUES
            st.session_state.selected_prompt_techniques = st.multiselect(
                "Choose prompt techniques to compare:",
                list(prompt_techniques.keys()),
                default=st.session_state.selected_prompt_techniques,
                key="prompt_technique_selector"
            )

            # Limit the number of selected techniques based on the slider value
            if len(st.session_state.selected_prompt_techniques) > max_techniques_to_compare:
                st.warning(f"You can compare a maximum of {max_techniques_to_compare} prompt techniques. Please deselect some techniques.")
                st.session_state.selected_prompt_techniques = st.session_state.selected_prompt_techniques[:max_techniques_to_compare]

            # Scenario Selection Header
            st.write("📌 Select the Scenario :")

            

            # Display all scenarios (Always Visible)
            # Use radio buttons for selecting one scenario instead of multiple checkboxes
            scenario_options = list(scenarios.keys())
            selected_scenario_name = st.radio(
                "Select a scenario:", 
                scenario_options,
                index=0 if st.session_state.selected_scenario else None
            )
            
            if selected_scenario_name:
                # Display the selected scenario details
                st.markdown(f"**Question:** {scenarios[selected_scenario_name]['question']}")
                
                # Display options as bullet points
                #for option in scenarios[selected_scenario_name]["options"]:
                   # st.markdown(f"- {option}")
                
                # Store the selected scenario in session state
                st.session_state.selected_scenario = scenarios[selected_scenario_name]["question"]
            else:
                st.session_state.selected_scenario = None

            # Comparison section
            st.subheader("Compare Prompt Techniques Side by Side")

            # Warning if no prompt techniques selected
            if not st.session_state.selected_prompt_techniques:
                st.warning("Please select at least one prompt technique above.")
            else:
                # Dynamically Create Columns for Each Prompt Technique
                cols = st.columns(len(st.session_state.selected_prompt_techniques))
                # Add "Compare Now" button
                if st.button("Compare Now"):
                    # Reset model responses
                    st.session_state.model_responses = {}
                    
                    # Loop through prompt techniques and assign them to columns
                    for i, technique in enumerate(st.session_state.selected_prompt_techniques):
                        with cols[i]:  # Assign each technique to a column
                            st.markdown(f"### {technique} Prompt")  # Technique Name in Bold

                            # Initialize the Groq client
                            client = Groq(api_key=GROQ_API_KEY)

                            # Create a complete prompt based on the technique
                            full_prompt = ""
                            
                            # Add uploaded content if available
                            if st.session_state.uploaded_file_content:
                                full_prompt = st.session_state.uploaded_file_content + "\n\n"
                            
                            # Add technique-specific instructions
                            if technique == "Few Shot":
                                # Check if the selected scenario matches a key in the Few Shot dictionary
                                if selected_scenario_name in prompt_techniques["Few Shot"]:
                                    # Get the examples for the selected scenario
                                    examples = prompt_techniques["Few Shot"][selected_scenario_name]
                                    full_prompt += "\n".join(examples) + "\n\n"
                                else:
                                    st.warning(f"No examples found for the selected scenario: {selected_scenario_name}")
                            else:
                                full_prompt += prompt_techniques[technique] + "\n\n"
                            
                            # Add the scenario question
                            if st.session_state.selected_scenario:
                                full_prompt += st.session_state.selected_scenario
                                
                                # Display the full prompt
                                with st.chat_message("user"):
                                    # Show the full prompt for Few Shot (including examples for the selected scenario)
                                    if technique == "Few Shot":
                                        st.markdown(f"**Prompt:**\n{full_prompt}")
                                    else:
                                        st.markdown(f"**Prompt:**\n{prompt_techniques[technique]}\n\n**Question:**\n{st.session_state.selected_scenario}")
                                
                                # Update chat history
                                st.session_state.chatHist[st.session_state.selected_model][technique].append(
                                    {"role": "user", "content": full_prompt}
                                )

                                # Fetch response from Groq API
                                try:
                                    response = client.chat.completions.create(
                                        model=st.session_state.selected_model,
                                        messages=st.session_state.chatHist[st.session_state.selected_model][technique],
                                        max_tokens=st.session_state.max_tokens,
                                        temperature=st.session_state.temperature
                                    )
                                    assistant_response = response.choices[0].message.content

                                    # Store the response
                                    if st.session_state.selected_model not in st.session_state.model_responses:
                                        st.session_state.model_responses[st.session_state.selected_model] = {}
                                    st.session_state.model_responses[st.session_state.selected_model][technique] = assistant_response

                                    # Append the response to the chat history
                                    st.session_state.chatHist[st.session_state.selected_model][technique].append(
                                        {"role": "assistant", "content": assistant_response}
                                    )
                                    
                                    # Display the response
                                    with st.chat_message("assistant"):
                                        st.markdown(assistant_response)
                                except Exception as e:
                                    st.error(f"Error: {e}", icon="🚨")
                                    if st.session_state.selected_model not in st.session_state.model_responses:
                                        st.session_state.model_responses[st.session_state.selected_model] = {}
                                    st.session_state.model_responses[st.session_state.selected_model][technique] = f"Error: {e}"

                
                    # Save responses to CSV file if the user is authenticated and not a guest
                    if (st.session_state.username != "guest" and 
                        st.session_state.model_responses and 
                        st.session_state.selected_scenario):
                        
                        # Flatten the responses structure for saving
                        flattened_responses = {}
                        for technique in st.session_state.selected_prompt_techniques:
                            if (st.session_state.selected_model in st.session_state.model_responses and 
                                technique in st.session_state.model_responses[st.session_state.selected_model]):
                                flattened_responses[f"{technique}_{st.session_state.selected_model}"] = (
                                    st.session_state.model_responses[st.session_state.selected_model][technique]
                                )
                        
                        saved_file = user_db.save_user_interaction(
                            st.session_state.username,
                            f"{st.session_state.selected_model} - {st.session_state.selected_scenario}",
                            flattened_responses
                        )
                        if saved_file:
                            st.sidebar.success(f"Responses saved to {saved_file}")
                        else:
                            st.sidebar.error("Failed to save responses")

        #tab2
        with tab2:
            st.header("Model Comparison with Single Prompt Technique")   

            # Model Settings in a popover
            # Model Settings in a popover
            with st.popover("⚙️ Model Settings"):
                # Temperature Slider with a unique key
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls the randomness of the model's responses. Lower values make the model more deterministic.",
                    key="temperature_slider_tab2"  # Unique key for Tab 2
                )

                # Max Tokens Slider with a unique key
                st.session_state.max_tokens = st.slider(
                    "Max Tokens",
                    min_value=200,
                    max_value=8192,
                    value=st.session_state.max_tokens,
                    step=100,
                    help="Controls the maximum number of tokens the model will generate.",
                    key="max_tokens_slider_tab2"  # Unique key for Tab 2
                )
            # Allow the user to select up to four models
            selected_models = st.multiselect(
                "Choose up to four models to compare:",
                llm_models,
                default=[llm_models[0]],  # Default to the first model
                key="model_selector_tab2"
            )

            # Limit the number of selected models to four
            if len(selected_models) > 4:
                st.warning("You can compare a maximum of four models. Please deselect some models.")
                selected_models = selected_models[:4]

            # Allow the user to select one prompt technique
            selected_prompt_technique = st.selectbox(
                "Choose a prompt technique:",
                list(prompt_techniques.keys()),
                key="prompt_technique_selector_tab2"
            )

            # Allow the user to select a scenario
            selected_scenario_name = st.radio(
                "Select a scenario:",
                list(scenarios.keys()),
                key="scenario_selector_tab2"
            )

            # Display the selected scenario question
            if selected_scenario_name:
                st.markdown(f"**Selected Scenario:** {scenarios[selected_scenario_name]['question']}")

            # Add "Compare Now" button
            if st.button("Compare Now", key="compare_now_tab2"):
                # Reset model responses for Tab 2
                st.session_state.model_responses_tab2 = {}
                
                # Loop through selected models and compare their responses
                cols = st.columns(len(selected_models))
                for i, model in enumerate(selected_models):
                    with cols[i]:  # Assign each model to a column
                        st.markdown(f"### {model}")  # Model Name in Bold

                        # Initialize the Groq client
                        client = Groq(api_key=GROQ_API_KEY)

                        # Create a complete prompt based on the selected technique and scenario
                        full_prompt = ""
                        
                        # Add technique-specific instructions
                        if selected_prompt_technique == "Few Shot":
                            # Check if the selected scenario matches a key in the Few Shot dictionary
                            if selected_scenario_name in prompt_techniques["Few Shot"]:
                                # Get the examples for the selected scenario
                                examples = prompt_techniques["Few Shot"][selected_scenario_name]
                                full_prompt += "\n".join(examples) + "\n\n"
                            else:
                                st.warning(f"No examples found for the selected scenario: {selected_scenario_name}")
                        else:
                            full_prompt += prompt_techniques[selected_prompt_technique] + "\n\n"
                        
                        # Add the scenario question
                        if selected_scenario_name:
                            full_prompt += scenarios[selected_scenario_name]["question"]
                            
                            # Display the full prompt
                            with st.chat_message("user"):
                                # Show the full prompt for Few Shot (including examples for the selected scenario)
                                if selected_prompt_technique == "Few Shot":
                                    st.markdown(f"**Prompt:**\n{full_prompt}")
                                else:
                                    st.markdown(f"**Prompt:**\n{prompt_techniques[selected_prompt_technique]}\n\n**Question:**\n{scenarios[selected_scenario_name]['question']}")
                            
                            # Fetch response from Groq API
                            try:
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": full_prompt}],
                                    max_tokens=st.session_state.max_tokens,
                                    temperature=st.session_state.temperature
                                )
                                assistant_response = response.choices[0].message.content

                                # Store the response
                                if model not in st.session_state.model_responses_tab2:
                                    st.session_state.model_responses_tab2[model] = {}
                                st.session_state.model_responses_tab2[model][selected_prompt_technique] = assistant_response

                                # Display the response
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                            except Exception as e:
                                st.error(f"Error: {e}", icon="🚨")
                                if model not in st.session_state.model_responses_tab2:
                                    st.session_state.model_responses_tab2[model] = {}
                                st.session_state.model_responses_tab2[model][selected_prompt_technique] = f"Error: {e}"

                # Save responses to CSV file if the user is authenticated and not a guest
                if (st.session_state.username != "guest" and 
                    st.session_state.model_responses_tab2 and 
                    selected_scenario_name):
                    
                    # Flatten the responses structure for saving
                    flattened_responses = {}
                    for model in selected_models:
                        if model in st.session_state.model_responses_tab2:
                            flattened_responses[f"{selected_prompt_technique}_{model}"] = (
                                st.session_state.model_responses_tab2[model][selected_prompt_technique]
                            )
                    
                    saved_file = user_db.save_user_interaction(
                        st.session_state.username,
                        f"Tab2 - {selected_prompt_technique} - {selected_scenario_name}",
                        flattened_responses
                    )
                    if saved_file:
                        st.sidebar.success(f"Responses saved to {saved_file}")
                    else:
                        st.sidebar.error("Failed to save responses")
        #tab3   
        with tab3:
            
            st.header("Model Response Evaluation")

            # Define the rubric
            rubric = {
                "Accuracy": {
                    "1": "Answer is incorrect or contradicts policy",
                    "3": "Answer is partially correct with minor misinterpretations",
                    "5": "Answer is completely correct with precise policy interpretation"
                },
                "Reasoning Quality": {
                    "1": "Lacks logical progression; conclusions don't follow from premises",
                    "3": "Basic reasoning supports the answer but has gaps or inconsistencies",
                    "5": "Step-by-step reasoning that expertly connects policy principles to conclusion"
                },
                "Completeness": {
                    "1": "Major policy concepts or considerations are missing",
                    "3": "Covers essential policy points but lacks some nuance",
                    "5": "Comprehensive coverage of all relevant policy elements, exceptions, and implications"
                },
                "Clarity": {
                    "1": "Confusing, ambiguous, or difficult to follow",
                    "3": "Understandable but with some structural issues",
                    "5": "Exceptionally clear, well-organized, and accessible explanation"
                },
                "Response Time": {
                    "1": "> 30 seconds",
                    "3": "11-25 seconds",
                    "5": "< 10 seconds"
                }
            }

            # Display the rubric table
            st.subheader("Rubric for Evaluation")
            st.write("Please evaluate the model responses based on the following rubric:")
            st.table(pd.DataFrame(rubric))

            # Create a form for user input
            with st.form("evaluation_form"):
                st.subheader("Evaluate Model Responses")
                evaluations = {}

                # Get all models from the evaluation files in the Exercise directory
                all_models = set()
                user_folders = [f"Exercise/{user}" for user in os.listdir("Exercise") if os.path.isdir(f"Exercise/{user}")]
                for user_folder in user_folders:
                    evaluation_files = [f for f in os.listdir(user_folder) if f.endswith("_evaluation.csv")]  # Changed to _evaluation.csv
                    for file in evaluation_files:
                        try:
                            df = pd.read_csv(os.path.join(user_folder, file))
                            # Extract models from the columns (e.g., model_llama, model_gpt)
                            models = [col for col in df.columns if col.startswith("model_")]
                            all_models.update(models)
                        except Exception as e:
                            st.error(f"Error reading file {file}: {e}")

                # Convert set to list for iteration
                all_models = list(all_models)

                # If no models are found, display a warning
                if not all_models:
                    st.warning("No models found in evaluation files. Please ensure evaluations have been saved.")
                    # Add some default models for demonstration
                    all_models = ["model_llama3-70b-8192", "model_mixtral-8x7b-32768"]

                # Loop through each model and create input fields for evaluation
                eval_data = {}
                for model in all_models:
                    st.markdown(f"### {model.replace('model_', '')}")
                    model_data = {}

                    # Create input fields for each criterion
                    for criterion, levels in rubric.items():
                        score = st.number_input(
                            f"{criterion} (0-5)",
                            min_value=0,
                            max_value=5,
                            value=0,  # Default value is 0
                            step=1,
                            key=f"{model}_{criterion}"
                        )
                        # Store the score in the model_data dictionary
                        model_data[criterion] = score
                    
                    # Store the model data in the eval_data dictionary
                    eval_data[model] = model_data

                # Add a submit button
                submitted = st.form_submit_button("Submit Evaluation")

                # Validate and save the evaluation
                if submitted:
                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    # Prepare data for the DataFrame
                    df_data = {}
                    for model, criteria in eval_data.items():
                        df_data[model] = criteria
                    
                    # Add timestamp to the data
                    df_data["metadata"] = {"timestamp": timestamp}
                    
                    # Create DataFrame with proper index
                    df = pd.DataFrame.from_dict(df_data, orient='index')
                    
                    # Save to CSV
                    if st.session_state.username != "guest":
                        user_folder = f"Exercise/{st.session_state.username}"
                        if not os.path.exists(user_folder):
                            os.makedirs(user_folder)

                        # Save the evaluation to a CSV file
                        filename = f"{user_folder}/{st.session_state.username}_evaluation_{timestamp}.csv"
                        
                        # Save the DataFrame to CSV
                        df.to_csv(filename)
                        st.success(f"Evaluation saved to {filename}")
                    else:
                        st.warning("You are using the app as a guest. Evaluations will not be saved.")

            # Visualization Section
            st.subheader("Visualization of Evaluations")

            # Add a button to trigger visualization
            if st.button("Click to Visualize"):
                # Gather all evaluation files from all user folders
                all_user_folders = [f"Exercise/{user}" for user in os.listdir("Exercise") if os.path.isdir(f"Exercise/{user}")]
                evaluation_files = []

                for user_folder in all_user_folders:
                    user_files = [os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith("_evaluation.csv")]
                    evaluation_files.extend(user_files)

                if evaluation_files:
                    # Load all evaluation data
                    all_data = []
                    for file in evaluation_files:
                        try:
                            df = pd.read_csv(file)
                            df["file"] = file
                            all_data.append(df)
                        except Exception as e:
                            st.error(f"Error reading file {file}: {e}")

                    if all_data:
                        # Combine all data into a single DataFrame
                        combined_data = pd.concat(all_data, ignore_index=True)
                        
                        # Ensure timestamp column exists
                        if "timestamp" in combined_data.columns:
                            # Convert timestamp to datetime
                            combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"], errors='coerce')
                        
                            # Display the combined data
                            st.write("### Combined Evaluations from All Users")
                            st.dataframe(combined_data)
                        
                            # Check if 'criterion' and 'score' columns exist
                            if "criterion" in combined_data.columns and "score" in combined_data.columns:
                                # Visualization: Bar chart for average scores by criterion
                                st.write("### Average Scores by Criterion")
                                avg_scores = combined_data.groupby("criterion")["score"].mean().reset_index()
                                st.bar_chart(avg_scores, x="criterion", y="score")
                        
                                # Visualization: Line chart for response time over time
                                st.write("### Response Time Over Time")
                                response_time_data = combined_data[combined_data["criterion"] == "Response Time"]
                                st.line_chart(response_time_data, x="timestamp", y="score")
                            else:
                                st.warning("Data format doesn't include 'criterion' and 'score' columns. Cannot create visualizations.")
                        else:
                            st.warning("Combined data doesn't have a 'timestamp' column. Cannot create time-based visualizations.")
                    else:
                        st.warning("No evaluation data could be loaded for visualization.")
                else:
                    st.warning("No evaluation files found for visualization.")
        with tab4:
            st.header("Tab 4 Content")
            st.write("This tab will be implemented as requested in a follow-up.")

#if __name__ == "__main__":
   # exercise2()                                
