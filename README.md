
# Flask Application for PDF Processing and Conversational Retrieval

## Overview

This Flask application enables users to upload PDF files, process and store their content in a FAISS vector store, and interact with the content through a chat interface powered by a conversational retrieval chain. The system utilizes advanced NLP techniques for text preprocessing and embedding, ensuring accurate and efficient information retrieval.

## Video Demo

https://github.com/kashish1344/Conversational-PDF/assets/83247791/ed9e76fb-429d-4259-96f6-40d5cb7fdb8d

os/demo.mp4)


## Key Features

### 1. PDF Upload and Processing
Users can upload PDF files to the application. The system processes these files by extracting text, preprocessing it to improve accuracy, and splitting the documents into manageable chunks.

### 2. Text Preprocessing
The application performs several preprocessing steps on the extracted text:
- Converts text to lowercase.
- Removes special characters and digits.
- Tokenizes the text.
- Removes common stopwords.

These steps help in cleaning and standardizing the text for better performance during retrieval.

### 3. Embeddings and Vector Store
The system uses HuggingFace Embeddings to convert text into vector representations. These vectors are stored in a FAISS vector store, which allows for efficient similarity-based retrieval of document content.

### 4. Conversational Interface
Users can interact with the uploaded documents via a chat interface. The application uses a conversational retrieval chain that leverages the stored embeddings to provide contextually relevant responses based on user queries. The chat interface maintains session history, ensuring coherent and context-aware interactions.

### 5. Session Management
The application supports session management, allowing users to reset the conversation and clear the chat history when needed.

## Technologies Used

### 1. Flask
- A lightweight WSGI web application framework in Python.

### 2. LangChain Community Libraries
- **PyPDFLoader**: For loading and extracting text from PDF files.
- **HuggingFaceEmbeddings**: For converting text into embeddings using models from HuggingFace.
- **RecursiveCharacterTextSplitter**: For splitting documents into chunks.
- **FAISS**: For storing and retrieving text embeddings based on similarity.
- **ConversationalRetrievalChain**: For managing the conversational interface and retrieving relevant document content.
- **Ollama Llama2**: Used as the LLM (Large Language Model) to generate responses based on user queries.

### 3. NLTK (Natural Language Toolkit)
- Used for text preprocessing tasks such as tokenization and stopword removal.

## Usage

1. **Home Page**: Navigate to the home page to upload a PDF file.
2. **Upload PDF**: Use the upload feature to submit a PDF file. The system processes the file, extracts text, and creates a searchable vector database.
3. **Chat Interface**: Interact with the uploaded document through the chat interface. Ask questions and receive contextually relevant answers based on the document content.
4. **Reset Conversation**: Use the reset feature to clear the chat history and start a new session.

## Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/kashish1344/Conversational-PDF.git
   cd Conversational-PDF
   \`\`\`
2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Run the Flask application:
   \`\`\`bash
   flask run
   \`\`\`

## NLTK Data

The application uses NLTK for text preprocessing. The necessary NLTK data files are downloaded at runtime. Ensure you have an active internet connection when running the application for the first time.
