import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks


def get_vector_store(text_chunks):
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

    return conversation_chain


def handle_user_input(user_question):
    # store response in session state
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    # All odd message is bot and even are human
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs.", page_icon="::books::")

    # Add CSS styling
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation sessions
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Initialize chat_history sessions
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Setup page
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your doc:")

    if user_question:
        handle_user_input(user_question)

    # Sidebar setup
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            'Upload your PDFs here and click on "Process"', accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
