# Import necessary libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Import necessary classes for other LLMs
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from the .env file
load_dotenv()

# Define the reset function
def clear_state():
    """
    Function to reset the Streamlit session state.
    This clears the user input and the enhanced prompt, effectively
    reseting the entire application to its initial state.
    """
    st.session_state.user_prompt = ""
    st.session_state.enhanced_prompt = None

def main():
    """
    Main function to run the Streamlit application.
    This app serves as a prompt enhancer and a multi-LLM generator.
    It takes a user's natural language input, enhances it using a selected LLM,
    and then allows the user to run the enhanced prompt through
    a different selected LLM (Gemini, Groq, or Hugging Face) to get a final output.
    """

    # --- Page Configuration ---
    st.set_page_config(
        page_title="Prompt Enhancer & Generator",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Custom Styling ---
    # Add custom CSS styles to the application
    st.markdown("""
        <style>
            body { color: #333; }
            .stApp { background-color: #f0f2f6; }
            .stButton > button {
                background-color: #4CAF50; color: white; border-radius: 20px;
                border: none; padding: 10px 20px; transition: background-color 0.3s;
            }
            .stButton > button:hover { background-color: #45a049; }
            .stTextArea textarea {
                border-radius: 10px; border: 1px solid #ddd; background-color: #fff;
            }
            .stMarkdown h1, .stMarkdown h2 { color: #2c3e50; }
            .response-container {
                background-color: #ffffff; border-left: 5px solid #4CAF50;
                padding: 20px; border-radius: 10px; margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .llm-output-container {
                background-color: #f9f9f9; border-left: 5px solid #007bff;
                padding: 20px; border-radius: 10px; margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            /* Custom CSS to style the selectbox for prompt enhancement */
            div[data-baseweb="select"] > div {
                background-color: white !important;
                color: black !important;
            }
            /* This targets the text within the selectbox after a selection is made */
            div[data-baseweb="select"] > div > div > div > div > div {
                color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.title("🚀 AI Prompt Enhancer & Generator")
    st.markdown("Enter a simple idea, and let AI transform it into a **concise and powerful** prompt.")

    # --- API Key Check ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # --- Prompt Enhancement Section ---
    with st.form(key='prompt_enhancer_form'):
        enhancement_llm_options = ["Google Gemini", "Groq"]
        selected_enhancement_llm = st.selectbox(
            "Select an LLM for Prompt Enhancement:",
            options=enhancement_llm_options,
            key='enhancement_llm_select'
        )
        
        user_prompt = st.text_area(
            "Enter your prompt idea here:",
            height=150,
            key='user_prompt',
            placeholder="e.g., 'I want to test a landing page, and I need a few test cases'"
        )

        # Create columns to place buttons side by side
        col1, col2 = st.columns([1, 1])

        with col1:
            submit_enhance_button = st.form_submit_button(label="✨ Enhance My Prompt")
        with col2:
            st.form_submit_button(label="🔄 Clear", on_click=clear_state)

    # --- Logic for Prompt Enhancement ---
    if submit_enhance_button and user_prompt:
        try:
            if selected_enhancement_llm == "Google Gemini":
                if not google_api_key:
                    st.error("🚨 Google API Key not found!")
                    st.info("Please add your key to a `.env` file: `GOOGLE_API_KEY='Your-API-Key-Here'`")
                    st.stop()
                llm_enhancer = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
            elif selected_enhancement_llm == "Groq":
                if not groq_api_key:
                    st.error("🚨 Groq API Key not found!")
                    st.info("Please add your key to a `.env` file: `GROQ_API_KEY='Your-Groq-Key-Here'`")
                    st.stop()
                llm_enhancer = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
            else:
                st.error("Invalid LLM selection.")
                st.stop()

            template = """
            As an expert prompt engineering assistant, your task is to refine a user's simple prompt into a concise yet powerful one.
            Your goal is to be brief but effective. Enhance the user's request by adding only the most essential context, a clear persona, and a specific desired output format.
            Avoid verbosity and filler words. The final prompt should be direct and to the point.

            Your response should begin with a friendly greeting: "Hello friend, here is a better prompt for you:"

            User's Prompt:
            '{user_prompt}'

            Enhanced Prompt:
            """
            prompt_template = PromptTemplate(input_variables=['user_prompt'], template=template)
            output_parser = StrOutputParser()
            enhancement_chain = prompt_template | llm_enhancer | output_parser

            with st.spinner("Crafting the perfect prompt..."):
                response = enhancement_chain.invoke({'user_prompt': user_prompt})
            
            st.session_state.enhanced_prompt = response
            st.markdown("## 🌟 Your Enhanced Prompt")
            st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please check your API key and network connection. Also, ensure the LLM API is enabled for your project.")

    # --- Final Output Generation Section ---
    st.markdown("---")
    st.markdown("## 🤖 Generate Final Output")
    st.markdown("Use the enhanced prompt to get a real result from the LLM of your choice.")

    if 'enhanced_prompt' in st.session_state and st.session_state.enhanced_prompt:
        with st.form(key='llm_selection_form'):
            llm_options = {
                "Google Gemini": "gemini",
                "Groq": "groq",
                "Hugging Face": "hugging_face",
            }
            
            selected_llm = st.selectbox(
                "Select an LLM to generate the final result:",
                options=list(llm_options.keys()),
            )
            
            submit_generate_button = st.form_submit_button(label=f"🚀 Generate with {selected_llm}")
            
        if submit_generate_button:
            enhanced_prompt_text = st.session_state.enhanced_prompt
            final_prompt_template = PromptTemplate(input_variables=[], template=enhanced_prompt_text)
            
            try:
                with st.spinner(f"Generating result using {selected_llm}..."):
                    final_response = ""
                    llm_model = None

                    if selected_llm == "Google Gemini":
                        if google_api_key:
                            llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
                        else:
                            st.error("Google API Key not found.")
                            st.stop()
                    
                    elif selected_llm == "Groq":
                        if groq_api_key:
                            llm_model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
                        else:
                            st.error("🚨 Groq API Key not found!")
                            st.info("Please add your key to your `.env` file: `GROQ_API_KEY='Your-Groq-Key-Here'`")
                            st.stop()
                    
                    elif selected_llm == "Hugging Face":
                        if huggingface_api_key:
                            try:
                                llm_model = HuggingFaceEndpoint(
                                    repo_id="tiiuae/falcon-7b-instruct",
                                    task="text-generation",
                                    huggingfacehub_api_token=huggingface_api_key,
                                    temperature=0.5,
                                    max_new_tokens=500
                                )
                            except Exception as e:
                                st.error(f"Failed to initialize Hugging Face model: {e}")
                                st.info("Please check your API token, model ID, or network connection.")
                                st.stop()
                        else:
                            st.error("🚨 Hugging Face API Token not found!")
                            st.info("Please add your token to your `.env` file: `HUGGINGFACEHUB_API_TOKEN='Your-Token-Here'`")
                            st.stop()

                    if llm_model:
                        chain = final_prompt_template | llm_model | StrOutputParser()
                        final_response = chain.invoke({})
                        st.markdown("### ✨ Final Result")
                        st.markdown(f'<div class="llm-output-container">{final_response}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please check your API key, model ID, or network connection.")
                
    else:
        st.info("Please enhance your prompt first before generating a final result.")

if __name__ == "__main__":
    if 'enhanced_prompt' not in st.session_state:
        st.session_state.enhanced_prompt = None
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    
    main()