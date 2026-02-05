import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents and returns text + page count.
    """
    text = ""
    total_pages = 0
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text, total_pages

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
    Note: Always uses OpenAI Embeddings for now to maintain retrieval quality.
    """
    if not api_key:
        st.error("OpenAI API Key is required for processing documents (Embeddings).")
        return None
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_llm(provider, model_name, api_key=None, base_url=None):
    """
    Factory function to get the appropriate LLM based on provider.
    """
    if provider == "OpenAI":
        return ChatOpenAI(
            temperature=0, 
            model_name=model_name, 
            openai_api_key=api_key
        )
    elif provider == "Grok (xAI)":
        return ChatOpenAI(
            temperature=0, 
            model_name=model_name, 
            openai_api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    elif provider == "Ollama":
        return ChatOllama(
            model=model_name,
            base_url=base_url if base_url else "http://host.docker.internal:11434"
        )
    elif provider == "Groq":
        return ChatGroq(
            temperature=0,
            model_name=model_name,
            groq_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_conversation_chain(vectorstore, llm):
    """
    Creates a conversational retrieval chain using the vector store and the chosen LLM.
    """
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True # Optional: useful for debugging
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

    # Custom CSS for sidebar styling and visibility
    st.markdown("""
        <style>
            /* Sidebar Background & Text */
            [data-testid="stSidebar"] {
                color: white !important;
                background-color: #7d3cff; /* Ensure fallback/reinforcement */
            }
            /* Sidebar Typography */
            [data-testid="stSidebar"] .stMarkdown, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] .stCaption {
                color: white !important;
            }
            
            /* Input Fields (Text Input, Select Box) - Fix White on White */
            [data-testid="stSidebar"] input {
                color: #333333 !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] span {
                color: #333333 !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] > div {
                background-color: white !important;
                color: #333333 !important;
            }
             /* Config for Dropdown options */
             ul[data-testid="stSelectboxVirtualDropdown"] li {
                 color: #333333 !important;
             }

            /* Buttons (Green Theme) */
            .stButton > button {
                background-color: #71dc99 !important;
                color: #004d26 !important; /* Darker green text for contrast */
                border: none;
                border-radius: 8px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #5ec785 !important;
                color: white !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                transform: translateY(-2px);
            }

            /* File Uploader Button */
            [data-testid="stFileUploader"] button {
                background-color: #71dc99 !important;
                color: #004d26 !important;
                border: none;
                border-radius: 8px;
            }
            [data-testid="stFileUploader"] button:hover {
                background-color: #5ec785 !important;
                color: white !important;
            }
            
            /* Modernize Sidebar Inputs */
            [data-testid="stSidebar"] .stTextInput > div > div {
                background-color: white !important;
                border-radius: 8px;
                border: none;
            }
            [data-testid="stSidebar"] .stSelectbox > div > div {
                background-color: white !important;
                border-radius: 8px;
                border: none;
            }
            
            /* Chat Input Styling */
            /* Remove purple background from chat input container if present */
            [data-testid="stBottom"] {
                background-color: white !important;
            }
            [data-testid="stBottom"] > div {
                background-color: white !important;
            }
            
            /* Input box styling - Specific Request */
            [data-testid="stChatInput"] {
                background-color: transparent !important;
            }
            [data-testid="stChatInput"] div[data-baseweb="input"] {
                background-color: white !important;
                border: 2px solid #7d3cff !important; /* Keep purple border */
                border-radius: 20px !important;
            }
            [data-testid="stChatInput"] textarea {
                background-color: white !important;
                color: #333333 !important;
                caret-color: #333333 !important;
            }
            /* Make key focus white and remove any other backgrounds */
            [data-testid="stChatInput"] div[data-baseweb="base-input"] {
                background-color: white !important;
            }

            /* Send Button Styling */
            [data-testid="stChatInput"] button[kind="primary"] {
                background-color: transparent !important; /* Usually transparent in chat input */
                color: #71dc99 !important; /* Green Icon */
                border: none;
            }
            [data-testid="stChatInput"] button[kind="primary"]:hover {
                color: #218838 !important; /* Darker green on hover */
                background-color: transparent !important;
            }
             /* If specific send icon SVG needs coloring */
            [data-testid="stChatInput"] button svg {
                fill: #71dc99 !important;
            }

            /* Password Visibility Toggle (Eye Icon) */
            [data-testid="stSidebar"] [data-testid="stTextInput"] button {
                color: #71dc99 !important;
            }
        </style>
    """, unsafe_allow_html=True)

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
        
        # LLM Selection
        llm_provider = st.selectbox(
            "Select LLM Provider",
            ("OpenAI", "Grok (xAI)", "Groq", "Ollama")
        )
        
        llm_api_key = None
        ollama_base_url = None
        model_name = "gpt-3.5-turbo" # default

        if llm_provider == "OpenAI":
            llm_api_key = st.text_input("OpenAI API Key", type="password")
            model_name = st.text_input("Model Name", value="gpt-3.5-turbo")
        elif llm_provider == "Grok (xAI)":
            llm_api_key = st.text_input("xAI API Key", type="password")
            model_name = st.text_input("Model Name", value="grok-beta")
        elif llm_provider == "Groq":
            llm_api_key = st.text_input("Groq API Key", type="password")
            model_name = st.text_input("Model Name", value="llama-3.3-70b-versatile")
        elif llm_provider == "Ollama":
            ollama_base_url = st.text_input("Ollama Base URL", value="http://host.docker.internal:11434")
            model_name = st.text_input("Model Name", value="llama3")
            st.info("Ensure Ollama is running (`ollama serve`). Users inside Docker need `host.docker.internal`.")

        st.divider()
        
        # Document Processing (Always requires OpenAI Key for Embeddings currently)
        st.subheader("Document Processing")
        st.caption("Requires OpenAI API Key for Embeddings")
        embedding_api_key = st.text_input("OpenAI API Key (for Embeddings)", type="password", key="embedding_key")
        
        # Pre-fill embedding key if OpenAI LLM is selected
        if llm_provider == "OpenAI" and llm_api_key:
            embedding_api_key = llm_api_key

        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not embedding_api_key:
                st.error("Please enter an OpenAI API Key for embeddings.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # 1. Get PDF text
                    raw_text, total_pages = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("Could not extract text from the PDF(s). They might be empty or scanned images without OCR.")
                    else:
                        # 2. Get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                             st.error("No text chunks created. The document might be too short.")
                        else:
                            # 3. Create vector store
                            try:
                                vectorstore = get_vectorstore(text_chunks, embedding_api_key)
                                
                                if vectorstore:
                                    # 4. Create conversation chain (Initially with selected LLM)
                                    try:
                                        llm = get_llm(llm_provider, model_name, llm_api_key, ollama_base_url)
                                        st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                                        st.success("Processing Done!")
                                        
                                        # Save stats to session state for persistence
                                        st.session_state.processing_stats = {
                                            "doc_count": len(pdf_docs),
                                            "total_pages": total_pages,
                                            "chunk_count": len(text_chunks),
                                            "avg_chunk_size": len(raw_text)//len(text_chunks),
                                            "raw_text": raw_text,
                                            "chunk_lengths": [len(chunk) for chunk in text_chunks]
                                        }

                                    except Exception as e:
                                        st.error(f"Error initializing LLM: {e}")
                            except Exception as e:
                                st.error(f"Error creating vector store: {e}")
                                
    # Display Infographic if stats exist
    if "processing_stats" in st.session_state and st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        
        # --- INFOGRAPHIC VISUALIZATION ---
        with st.expander("Document Infographic", expanded=False):
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Documents", stats["doc_count"])
            col2.metric("Total Pages", stats["total_pages"])
            col3.metric("Text Chunks", stats["chunk_count"])
            col4.metric("Avg Chunk Size", f"{stats['avg_chunk_size']} chars")
            
            # Word Cloud
            st.write("#### Word Cloud")
            try:
                # Use cached raw text
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stats["raw_text"])
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except Exception as wc_error:
                st.warning(f"Could not generate word cloud: {wc_error}")

            # Chunk Distribution
            st.write("#### Chunk Size Distribution")
            fig_hist = px.histogram(x=stats["chunk_lengths"], nbins=20, labels={'x': 'Chunk Size (chars)', 'y': 'Count'}, title="Chunk Length Distribution")
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist)
        
        # Allow updating LLM without re-processing docs if vector store exists
        if st.session_state.conversation is not None and st.button("Update LLM Settings"):
             # We need to retrieve the vectorstore. For now, we reuse the one in the chain or reload (complex).
             # Simpler: Just warn user to re-process if they want to switch mid-stream, 
             # OR effectively we can't easily swap the LLM in the EXISTING chain without rebuilding it.
             # But we can rebuild it if we stored the vectorstore in session_state? 
             # LangChain objects often aren't pickleable for session_state.
             # We check if we can rebuild the chain.
             st.warning("To switch LLMs, please re-process the documents for now.")

if __name__ == '__main__':
    main()
