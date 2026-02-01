# Agentic Onboarding 🤖📚

Welcome to **Agentic Onboarding**! This tool allows you to chat with your own PDF documents. Imagine having a smart assistant that has read all your company manuals, onboarding docs, or research papers and can answer any question about them instantly.

## 🌟 What is this?
This application uses a technology called **RAG (Retrieval-Augmented Generation)**.
1.  **Read**: You upload your PDF files. The app reads them and breaks them into small pieces.
2.  **Remember**: It saves these pieces in a special "vector database" (a way for computers to understand meanings).
3.  **Answer**: When you ask a question, the app finds the most relevant pieces of text from your PDFs and sends them to a smart AI (OpenAI's GPT). The AI uses this specific information to give you an accurate answer.

---

## 🛠️ Prerequisites
Before you start, make sure you have:
1.  **Docker Desktop**: Installed and running on your computer. [Download Docker](https://docs.docker.com/get-docker/)
2.  **OpenAI API Key**: You need a key to use the AI brain. [Get your key here](https://platform.openai.com/api-keys)

---

## 🚀 How to Run the App (The Easy Way)

### 1. Open your Terminal
Navigate to the folder where you downloaded this project.

### 2. Start the App
Run this single command in your terminal:
```bash
docker-compose up --build
```
*Wait a few moments for the building process to finish. When you see "Streamlit run app.py", it's ready!*

### 3. Open in Browser
Go to this address in your web browser (Chrome, Safari, etc.):
**http://localhost:8501**

---

## 📖 How to Use the Web Interface

1.  **Enter API Key**: On the left sidebar, paste your OpenAI API Key.
2.  **Upload Documents**: Click the "Browse files" button in the sidebar and select your PDF files.
3.  **Process**: Click the **"Process"** button.
    *   *What's happening?* The app is reading your files and organizing the information. Wait for the "Processing Done!" message.
4.  **Chat**: Type your question in the main chat box (e.g., "What is the holiday policy?"). The AI will answer based ONLY on your documents!

---

## 🤓 Advanced: Command Line Querying
If you prefer using the terminal (or want to automate things), you can ask questions directly without the web browser.

**Prerequisite**: You must have "Processed" your documents in the web interface at least once (this creates the database).

Run this command (replace `YOUR_API_KEY` with your actual key):
```bash
docker exec -it rag-rag-app-1 python query_index.py "Your question here" --api_key "YOUR_API_KEY"
```

**Example:**
```bash
docker exec -it rag-rag-app-1 python query_index.py "Who is the CEO?" --api_key "sk-proj-1234..."
```

---

## ⚠️ Troubleshooting
*   **"OPENAI_API_KEY variable is not set" warning**: You can ignore this in the terminal. You enter the key in the app sidebar.
*   **App not loading?**: Make sure Docker is running.
*   **Answers are wrong?**: Ensure you clicked "Process" after uploading new PDFs. The AI only knows what you've uploaded.

---

**Happy Chatting!** 🚀
