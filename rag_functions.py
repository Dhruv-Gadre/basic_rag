import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from ollama import Client

class RAGPipeline:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Client(host='http://localhost:11434')
        self.index = None
        self.chunks = None
    
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                text += page_text + "\n"
        return text
    
    def chunk_text(self, text, max_length=500):
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
    
    def create_index(self, chunks):
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        return index, chunks
    
    def save_index(self, index, chunks, index_path="syllabus_index.faiss", chunks_path="syllabus_chunks.pkl"):
        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
    
    def load_index(self, index_path="syllabus_index.faiss", chunks_path="syllabus_chunks.pkl"):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
    
    def retrieve(self, query, k=3):
        if self.index is None or self.chunks is None:
            raise ValueError("Index and chunks must be loaded first")
        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]
    
    def generate_answer(self, query, retrieved_chunks):
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question using ONLY the context below. If unsure, say "I don't know".
        
        Question: {query}

        Context: \n{context}

        Answer:"""
        response = self.client.chat(
            model='llama2:7b',
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']