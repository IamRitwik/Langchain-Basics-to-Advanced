import streamlit as st
import utils

st.title("Chat with your code!")

user_repo = st.text_input("Github Link to your public codebase...")
if user_repo:
    st.write("You entered:", user_repo)

    # Load the GitHub Repo
    embedder = utils.Embedder(user_repo)
    embedder.clone_repo()
    st.write("Trying to understand your code! This may take some time")

    # Chunk and Create DB
    st.write("Your repo has been analyzed!")
    embedder.load_db()
    st.write("Done. Ready to take your questions!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your question here."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(embedder.retrieve_results(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
