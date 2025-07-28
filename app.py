import streamlit as st
# You'll need to make sure this function is a generator
from rag_system import load_model_and_db, query_rag_stream
import time # Added for the example generator

# --- Example `rag_system.py` modification ---
# The user's provided `rag_system.py` is assumed to be correct.
# It should look something like this:
#
# def query_rag_stream(query_text: str, db, model):
#     # ... (search logic) ...
#     # Stream the response from the model
#     for token in model.stream(prompt):
#         yield {"token": token}
#
#     # After streaming the response, yield the sources
#     sources = [doc.metadata for doc, _score in results]
#     yield {"sources": sources}
#
# -------------------------------------------------


# --- Streamlit App ---

st.set_page_config(page_title="DocBot", layout="centered")

st.title("💬 DocBot")
st.write("Go ahead. Ask me anything from the documentation!")

# The @st.cache_resource decorator is great for loading models and data
@st.cache_resource
def setup():
    """Load the database and model."""
    # This function should return your actual database and model
    return load_model_and_db()

# Load the resources
db, model = setup()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box
if query := st.chat_input("Ask away!"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response
    with st.chat_message("assistant"):
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        # The `query_rag_stream` function acts as a generator
        response_generator = query_rag_stream(query, db, model)
        
        # Manually iterate through the generator to handle structured output
        for item in response_generator:
            if "token" in item:
                # Append token to the full response and update the placeholder
                full_response += str(item["token"])
                response_placeholder.markdown(full_response + " ")
            elif "sources" in item:
                # Collect the sources
                sources = item["sources"]
        
        # Update the placeholder with the final response without the cursor
        response_placeholder.markdown(full_response)

        # Display the sources if they exist
        if sources:
            st.markdown("---")
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"- {source}")


    # Add the full assistant response (without sources) to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

