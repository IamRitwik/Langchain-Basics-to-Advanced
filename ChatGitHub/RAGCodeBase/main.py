import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import utils

st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖")
st.title("Chat with your code base!")

user_repo = st.text_input(
        "Please paste your github repository link here 👇:",
        placeholder="Github URL")

if user_repo:
    st.write("Your codebase:", user_repo)

    # Load the GitHub Repo
    embedder = utils.Embedder(user_repo)
    embedder.clone_repo()

    with st.status("Reading your codebase..."):
        st.write("Analyzing your repository! This may take some time")
        # Chunk and Create DB
        st.write("Trying to understand your code!")
        embedder.create_retriever()
        st.write("Done. Ready to take your questions!")

    # Initialize chat history
    chat_history = [
        AIMessage(content="I am here to assist you, ask me anything from Your codebase!"),
    ]

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Accept user input
    if user_input := st.chat_input("Type your question here."):
        # Add user message to chat history
        chat_history.append(HumanMessage(content=user_input))
        # Display user message in chat message container
        with st.chat_message("Human"):
            st.markdown(user_input)
        # Display assistant response in chat message container
        with st.chat_message("AI"):
            response = st.write_stream(embedder.process_chat(user_input, chat_history))
        # Add assistant response to chat history
        chat_history.append(AIMessage(content=response))
        st.session_state.chat_history = chat_history
