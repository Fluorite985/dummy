import streamlit as st
from typing import Generator, List, Dict, Any

# Import the actual functions from your RAG system file
from rag_system import load_model_and_db, query_rag_stream

# --- Streamlit App ---

st.set_page_config(page_title="DocBot", layout="centered")

st.title("💬 DocBot")
st.write("I can answer any question from the documentation! Ask away.")

# The @st.cache_resource decorator is great for loading models and data
@st.cache_resource
def setup():
    """Load the database and model from the RAG system."""
    # This function now calls your actual loading function
    return load_model_and_db()

# Load the resources
# This might take a moment on first run.
with st.spinner("Loading knowledge base and model..."):
    db, model = setup()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box
if query := st.chat_input("Ask your question..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response
    with st.chat_message("assistant"):
        # Use a placeholder for the streaming response
        response_placeholder = st.empty()
        full_response = ""
        sources = []
        
        # Use a spinner while the initial search and prompt generation happens
        with st.spinner("Searching documents and thinking..."):
            # Get the generator for the response
            response_generator = query_rag_stream(query, db, model)
            
            # Manually iterate through the generator to handle structured output
            for item in response_generator:
                if "token" in item:
                    # Append token to the full response and update the placeholder
                    full_response += str(item.get("token", ""))
                    # Add a blinking cursor to simulate typing
                    response_placeholder.markdown(full_response + "▌")
                elif "sources" in item:
                    # Collect the sources
                    sources = item["sources"]
        
        # Update the placeholder with the final response without the cursor
        response_placeholder.markdown(full_response)

        # Display the sources in an expander if they exist
        if sources:
            with st.expander("View Sources"):
                for i, source in enumerate(sources):
                    # The source object from LangChain is a Document's metadata dict
                    source_info = f"**Source {i+1}:** `{source}`"
                    st.markdown(source_info)
                    
    # Add the full assistant response (text part only) to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
