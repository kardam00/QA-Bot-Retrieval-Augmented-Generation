import streamlit as st
from PyPDF2 import PdfReader
from io import StringIO
from pinecone import Pinecone
import cohere
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import sent_tokenize
import nltk  # Import NLTK

nltk.download('punkt')  # This downloads the basic punkt tokenizer.
# nltk.download('punkt_tab')  # Uncomment if 'punkt_tab' is needed.

# Initialize Pinecone
pc = Pinecone(
    api_key="API key"
)
index_name = "qa-bot-index"

# Delete existing index if it exists
if index_name in pc.list_indexes():
    pc.delete_index(index_name)
    print(f"Deleted existing index: {index_name}")

# Create a new index with the correct dimension
pc.create_index(index_name, dimension=384)
print(f"Created new index: {index_name} with dimension 384")

# Connect to the new index
index = pc.Index(index_name)

# Initialize Cohere
co = cohere.Client("API key")

# Load pre-trained model (e.g., Sentence-BERT)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

# PDF processing function
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split the text into chunks
def split_into_chunks(context, max_length=100):
    sentences = sent_tokenize(context)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Generate embeddings for the document chunks
def generate_embeddings_batch(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):  
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings.cpu().numpy())
    return torch.cat([torch.tensor(batch) for batch in embeddings])

# Pinecone query function
def query_pinecone(query, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    results = index.query(vector=query_embedding.tolist(), top_k=top_k)
    return results['matches']

# Cohere for answering queries
def generate_answer(contexts, question):
    context_text = " ".join(contexts)
    response = co.generate(
        model="command",
        prompt=f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer:",
        max_tokens=150,
        temperature=0.5
    )
    return response.generations[0].text.strip()

# Streamlit App
def main():
    st.title("Interactive QA Bot")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
        st.write("Document uploaded successfully!")
        
        # Step 2: Split the document into chunks
        document_chunks = split_into_chunks(text)
        st.write(f"Total chunks created: {len(document_chunks)}")
        
        # Generate embeddings for the chunks and store in Pinecone
        doc_embeddings = generate_embeddings_batch(document_chunks)
        vectors_to_upsert = [(str(i), emb.numpy().tolist()) for i, emb in enumerate(doc_embeddings)]
        
        # Log upserted IDs
        for vector in vectors_to_upsert:
            print(f"Upserting Vector ID: {vector[0]}")
        
        # Batch upsert
        index.upsert(vectors=vectors_to_upsert)
        st.write("Document embeddings stored in Pinecone.")
        print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone.")
    
    # Step 3: Question Input
    user_query = st.text_input("Ask a question based on the uploaded document:")
    
    if user_query:
        # Query Pinecone and generate answer
        results = query_pinecone(user_query)
        st.write(f"Query results: {results}")
        
        # Log returned IDs
        for res in results:
            print(f"Returned Vector ID: {res['id']}")
        
        # Safely retrieve relevant chunks
        relevant_chunks = []
        for res in results:
            try:
                idx = int(res['id'])
                if 0 <= idx < len(document_chunks):
                    relevant_chunks.append(document_chunks[idx])
                else:
                    st.warning(f"Received invalid ID {res['id']} from Pinecone.")
            except ValueError:
                st.warning(f"Received non-integer ID {res['id']} from Pinecone.")
        
        if relevant_chunks:
            answer = generate_answer(relevant_chunks, user_query)
            st.write("Answer:", answer)
            st.write("Relevant Document Sections:")
            for chunk in relevant_chunks:
                st.write(chunk)
        else:
            st.write("No relevant sections found.")

if __name__ == "__main__":
    main()
