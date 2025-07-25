import fitz #PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from ollama import Client

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf):
            page_text = page.get_text()
            text += page_text + "\n"
            print(f"✅ Extracted Page {page_num + 1}")
    return text

pdf_path = r"media\DBMS_Syllabus.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

with open("syllabus.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)
    
def chunk_text(text, max_length=500):
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = chunk_text(extracted_text, max_length=500)



model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings Shape: \n {np.array(embeddings).shape}")


# FAISS requires data in numpy float32 format to build its index.
embeddings_np = np.array(embeddings).astype("float32")

# IndexFlatL2 creates a flat (brute-force) index using L2 (Euclidean) distance for similarity search.
index = faiss.IndexFlatL2(embeddings_np.shape[1])

index.add(embeddings_np)

print(f"Index created and {index.ntotal} items added.")

faiss.write_index(index, "syllabus_index.faiss")
print("✅ FAISS index saved to syllabus_index.faiss")



with open("syllabus_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Chunks saved to syllabus_chunks.pkl")

# Retrieve top - k embeddings dependent on the query.
def retrieve(query, k = 3):
    #Encode the query into an embedding
    query_embedding = model.encode([query]).astype("float32")
    
    #Search the FAISS index for k-similar chunks
    distances, indices = index.search(query_embedding, k)
    
    result = [chunks[i] for i in indices[0]]
    
    return result


client = Client(host='http://localhost:11434')

def generate_answer_ollama(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    print(f"✅ I have recieved the context and it is: {context}")

    prompt = f"""Answer the question using ONLY the context below. If unsure, say "I don't know".

            Question: {query}

            Context: \n{context}

            Answer:"""

    response = client.chat(
        model='llama2:7b',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']

query = "What is Architecture of DBMS"
results = retrieve(query, k=2)
answer = generate_answer_ollama(query, results)

print(answer)