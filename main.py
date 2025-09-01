import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Import necessary classes for other LLMs
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceHub

# Load environment variables from the .env file.
# This ensures that API keys are not hardcoded in the script,
# which is a best practice for security.
load_dotenv()

def main():
    """
    Main function to run the Streamlit application.
    This app serves as a prompt enhancer and a multi-LLM generator.
    It takes a user's natural language input, enhances it using Gemini,
    and then allows the user to run the enhanced prompt through
    a selected LLM (Gemini, Groq, or Hugging Face) to get a final output.
    """

    # --- Page Configuration ---
    # Sets up the basic page layout, title, and icon.
    st.set_page_config(
        page_title="Prompt Enhancer & Generator",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Custom Styling ---
    # Applies custom CSS to improve the app's visual appearance,
    # making it more user-friendly and aesthetically pleasing.
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
        </style>
    """, unsafe_allow_html=True)


    # --- Header ---
    st.title("🚀 AI Prompt Enhancer & Generator")
    st.markdown("Enter a simple idea, and let AI transform it into a **concise and powerful** prompt.")

    # --- API Key Check ---
    # Checks for the presence of required API keys. This prevents the app
    # from running into errors if a key is missing and provides clear instructions
    # to the user on how to set it up.
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not google_api_key:
        st.error("🚨 Google API Key not found!")
        st.info("Please add your key to a `.env` file: `GOOGLE_API_KEY='Your-API-Key-Here'`")
        st.stop()


    # --- Prompt Enhancement Section ---
    # Form to take the user's initial prompt idea.
    enhanced_prompt = None
    with st.form(key='prompt_enhancer_form'):
        user_prompt = st.text_area(
            "Enter your prompt idea here:",
            height=150,
            placeholder="e.g., 'I want to test a landing page, and I need a few test cases'"
        )
        submit_enhance_button = st.form_submit_button(label="✨ Enhance My Prompt")


    # --- Logic for Prompt Enhancement ---
    # This block executes when the "Enhance My Prompt" button is clicked.
    if submit_enhance_button and user_prompt:
        try:
            # Initialize the Gemini LLM for prompt enhancement.
            llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

            # Create a sophisticated prompt template for the enhancement task.
            template = """
            As an expert prompt engineering assistant, your task is to refine a user's simple prompt into a concise yet powerful one.
            Your goal is to be brief but effective. Enhance the user's request by adding only the most essential context, a clear persona, and a specific desired output format.
            Avoid verbosity and filler words. The final prompt should be direct and to the point.

            Your response should begin with a friendly greeting: "Hello friend, here is a better prompt for you:"

            User's Prompt:
            '{user_prompt}'

            Enhanced Prompt:
            """

            prompt_template = PromptTemplate(
                input_variables=['user_prompt'],
                template=template
            )

            # Use LangChain Expression Language (LCEL) to create the chain.
            output_parser = StrOutputParser()
            enhancement_chain = prompt_template | llm_gemini | output_parser

            # Run the chain and store the response in Streamlit's session state.
            with st.spinner("Crafting the perfect prompt..."):
                response = enhancement_chain.invoke({'user_prompt': user_prompt})
            
            # Use session state to persist the enhanced prompt across reruns.
            st.session_state.enhanced_prompt = response

            # Display the result to the user.
            st.markdown("## 🌟 Your Enhanced Prompt")
            st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please check your API key and network connection. Also, ensure the Gemini API is enabled for your project.")


    # --- Final Output Generation Section ---
    # This new section allows the user to choose an LLM for the final task.
    st.markdown("---")
    st.markdown("## 🤖 Generate Final Output")
    st.markdown("Use the enhanced prompt to get a real result from the LLM of your choice.")
    
    # Check if an enhanced prompt exists before showing this form.
    # This prevents the user from trying to generate output before
    # a prompt has been enhanced.
    if 'enhanced_prompt' in st.session_state and st.session_state.enhanced_prompt:
        with st.form(key='llm_selection_form'):
            
            llm_options = {
                "Google Gemini": "gemini",
                "Groq": "groq",
                "Hugging Face": "hugging_face",
            }
            
            # Create a dropdown menu for LLM selection.
            selected_llm = st.selectbox(
                "Select an LLM to generate the final result:",
                options=list(llm_options.keys()),
            )
            
            submit_generate_button = st.form_submit_button(label=f"🚀 Generate with {selected_llm}")
            
        if submit_generate_button:
            
            # Retrieve the enhanced prompt from session state.
            enhanced_prompt_text = st.session_state.enhanced_prompt
            
            # The enhanced prompt is already a complete, ready-to-use prompt.
            # We just need a simple template to pass it to the final LLM.
            final_prompt_template = PromptTemplate(
                input_variables=[],
                template=enhanced_prompt_text
            )
            
            try:
                # Conditional logic to select and run the chosen LLM.
                with st.spinner(f"Generating result using {selected_llm}..."):
                    
                    final_response = ""
                    llm_model = None

                    # Gemini Model
                    if selected_llm == "Google Gemini":
                        if google_api_key:
                            llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
                        else:
                            st.error("Google API Key not found.")
                            st.stop()
                    
                    # Groq Model
                    elif selected_llm == "Groq":
                        if groq_api_key:
                            # Use a current, supported Groq model like 'llama-3.1-8b-instant'
                            llm_model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
                        else:
                            st.error("🚨 Groq API Key not found!")
                            st.info("Please add your key to your `.env` file: `GROQ_API_KEY='Your-Groq-Key-Here'`")
                            st.stop()
                    
                    # Hugging Face Model
                    elif selected_llm == "Hugging Face":
                        if huggingface_api_key:
                            # Use a Hugging Face model from the Hub, for example, "mistralai/Mixtral-8x7B-Instruct-v0.1".
                            # The 'task' parameter is REQUIRED for this model.
                            llm_model = HuggingFaceHub(
                                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                huggingfacehub_api_token=huggingface_api_key,
                                task="text-generation",
                                model_kwargs={"temperature": 0.5, "max_length": 500}
                            )
                        else:
                            st.error("🚨 Hugging Face API Token not found!")
                            st.info("Please add your token to your `.env` file: `HUGGINGFACEHUB_API_TOKEN='Your-Token-Here'`")
                            st.stop()

                    # Run the final chain if a model was successfully initialized.
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


    # --- Footer ---
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, LangChain, and various LLMs.")


if __name__ == "__main__":
    # Initialize session state for the enhanced prompt.
    # This is a crucial step to ensure the state is not lost
    # between Streamlit app reruns.
    if 'enhanced_prompt' not in st.session_state:
        st.session_state.enhanced_prompt = None
    
    main()