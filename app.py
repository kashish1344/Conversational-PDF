from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
import os
import tempfile
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.schema import Document 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Rejoin the words into a single string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text

# Initialize session state
messages = []
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Initialize db_retriever with a default value (None or a placeholder retriever)
db_retriever = None

# Prompt template
prompt_template = """This is a chat template. As a chat bot, your primary objective is to provide accurate and concise information based on the user's questions about the uploaded document. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the uploaded document. Do not give any other information or note.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the Ollama Llama2 model
llama_model = Ollama(model="llama2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global db_retriever

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Preprocess the text for better accuracy
        preprocessed_documents = [Document(page_content=preprocess_text(doc.page_content), metadata=doc.metadata) for doc in documents]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        texts = text_splitter.split_documents(preprocessed_documents)

        faiss_db = FAISS.from_documents(texts, embeddings)
        faiss_db.save_local("ipc_vector_db")

        db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        return jsonify({'message': 'PDF processed and database created successfully!'})

@app.route('/chat', methods=['POST'])
def chat():
    global db_retriever

    if db_retriever is None:
        return jsonify({'message': 'Please upload a PDF file to create the database before asking questions.'})

    user_input = request.json.get('input')

    messages.append({"role": "user", "content": user_input})

    qa = ConversationalRetrievalChain.from_llm(
        llm=llama_model,
        memory=memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    result = qa.invoke(input=user_input)

    messages.append({"role": "assistant", "content": result["answer"]})

    return jsonify({'response': result["answer"]})

@app.route('/reset', methods=['POST'])
def reset():
    global messages, memory
    messages = []
    memory.clear()
    return jsonify({'message': 'Conversation reset successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
