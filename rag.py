import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding Model
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import re

# ✅ Step 1: Initialize Embedding Model for FAISS
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# ✅ Step 2: Initialize Hugging Face LLM (GPT-2 for response generation)
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
hf = HuggingFacePipeline(pipeline=pipe)

# Function to Create FAISS Vector Store
def create_vector_store_from_txt(file_path, vector_db_path):
    """Creates a FAISS vector store from JSON data (one object per line) in a file and saves it."""
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_data = json.loads(line)  # Parse JSON object
                    content = f"Question: {json_data['question']}\nContext: {json_data['context']}\nAnswer: {json_data['answer']}"
                    document = Document(page_content=content, metadata={"question": json_data["question"]})
                    documents.append(document)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format on line: {line.strip()}")
                    continue  # Skip to the next line if there's an error
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)  # ✅ Use correct embedding model

    vector_store.save_local(vector_db_path)
    return vector_store

# Function to Load FAISS Vector Store
def load_vector_store(vector_db_path):
    """Loads a FAISS vector store from a file."""
    try:
        vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# Function for Retrieval Augmented Generation (RAG)
def rag_query(vector_store, query):
    """Performs a Retrieval Augmented Generation (RAG) query."""
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in relevant_docs])  # Combine relevant passages
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = hf.invoke(prompt)  # ✅ Generate response using GPT-2
    return response

def extract_last_answer(llm_output):
    """Extracts only the last answer from the LLM output."""
    matches = re.findall(r"Answer:\s*(.+)", llm_output)  # Find all 'Answer:' occurrences
    return matches[-1] if matches else None  # Return the last one

if __name__ == "__main__":
    # File paths
    diabetes_file_path = "PIMA/diabetes_dataset.txt"
    diabetes_vector_db_path = "PIMA/vector_db"

    retinopathy_file_path = "Two classes/diabetic_retinopathy_dataset.txt"
    retinopathy_vector_db_path = "Two classes/vector_db"

    # Create or Load Vector Stores
    if not os.path.exists(diabetes_vector_db_path):
        diabetes_vector_db = create_vector_store_from_txt(diabetes_file_path, diabetes_vector_db_path)
    else:
        diabetes_vector_db = load_vector_store(diabetes_vector_db_path)

    if not os.path.exists(retinopathy_vector_db_path):
        retinopathy_vector_db = create_vector_store_from_txt(retinopathy_file_path, retinopathy_vector_db_path)
    else:
        retinopathy_vector_db = load_vector_store(retinopathy_vector_db_path)

    # Queries and Responses
    if diabetes_vector_db:
        diabetes_query = "What are healthy fats for diabetics?"
        diabetes_response = rag_query(diabetes_vector_db, diabetes_query)
        print("Diabetes Response:", extract_last_answer(diabetes_response))

    if retinopathy_vector_db:
        retinopathy_query = "What are the treatment options for gestational diabetes?"
        retinopathy_response = rag_query(retinopathy_vector_db, retinopathy_query)
        print("Retinopathy Response:", extract_last_answer(retinopathy_response))
