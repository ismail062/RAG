import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the raw text into chunks.
    Chunk size: 1000 characters
    Overlap: 200 characters
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    """
    Creates a persistent vector store using OpenAI Embeddings and FAISS.
    """
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
        return None
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """
    Creates a conversational retrieval chain using the vector store and ChatOpenAI.
    """
    if not api_key:
        return None
        
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=api_key)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """
    Handles the user's question: sends it to the conversation chain and updates the chat history.
    """
    if st.session_state.conversation is None:
        st.warning("Please process documents first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

def main():
    st.set_page_config(page_title="Agentic Onboarding", page_icon=":busts_in_silhouette:")
    st.header("Agentic Onboarding :busts_in_silhouette:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main Chat Interface
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar Configuration
    with st.sidebar:
        st.subheader("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not api_key:
                st.error("Please enter your OpenAI API Key first.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # 1. Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create vector store
                    vectorstore = get_vectorstore(text_chunks, api_key)
                    
                    if vectorstore:
                        # 4. Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                        st.success("Processing Done!")

if __name__ == '__main__':
    main()
