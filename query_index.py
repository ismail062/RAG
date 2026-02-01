import argparse
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    parser = argparse.ArgumentParser(description="Query the RAG vector store directly.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key (optional if set in env)", default=None)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API Key must be provided via --api_key or OPENAI_API_KEY environment variable.")
        return

    # Check if index exists
    if not os.path.exists("faiss_index"):
        print("Error: 'faiss_index' directory not found. Please process documents in the app first.")
        return

    # Load Vector Store
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # Create QA Chain
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Run Query
    print(f"\nQuestion: {args.query}\n")
    try:
        result = qa_chain.invoke({"query": args.query})
        print(f"Answer: {result['result']}")
    except Exception as e:
         print(f"Error executing query: {e}")

if __name__ == "__main__":
    main()
