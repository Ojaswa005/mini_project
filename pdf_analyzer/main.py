import os
import streamlit as st
import PyPDF2
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
import base64
import openai
from transformers import pipeline
import json

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document"""
    text: str
    page_num: int
    section: str
    doc_id: str
    metadata: Dict[str, Any]

@dataclass
class ImageData:
    """Represents an extracted image from PDF"""
    image: Image.Image
    page_num: int
    bbox: Tuple[float, float, float, float]
    description: str = ""

class PDFProcessor:
    """Handles PDF text extraction, chunking, and image extraction"""
    
    def __init__(self):
        self.section_patterns = {
            'abstract': r'abstract|summary',
            'introduction': r'introduction|background',
            'methodology': r'method|approach|technique|implementation',
            'results': r'results|findings|evaluation|experiment',
            'conclusion': r'conclusion|discussion|future work'
        }
    
    def extract_text_and_images(self, pdf_file) -> Tuple[List[DocumentChunk], List[ImageData]]:
        """Extract text chunks and images from PDF"""
        chunks = []
        images = []
        
        # Open PDF with PyMuPDF for better image extraction
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(BytesIO(img_data))
                        bbox = page.get_image_bbox(img)
                        images.append(ImageData(
                            image=img_pil,
                            page_num=page_num + 1,
                            bbox=bbox
                        ))
                    pix = None
                except Exception as e:
                    st.warning(f"Could not extract image from page {page_num + 1}: {str(e)}")
            
            # Process text into chunks
            if text.strip():
                section = self._identify_section(text)
                chunk = DocumentChunk(
                    text=text.strip(),
                    page_num=page_num + 1,
                    section=section,
                    doc_id=pdf_file.name,
                    metadata={'word_count': len(text.split())}
                )
                chunks.append(chunk)
        
        pdf_document.close()
        return chunks, images
    
    def _identify_section(self, text: str) -> str:
        """Identify which section of the paper this text belongs to"""
        text_lower = text.lower()
        for section, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower):
                return section
        return 'other'
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        return chunks

class VectorStore:
    """Handles embeddings and vector search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        self.embeddings = self.model.encode(texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        if self.index:
            faiss.write_index(self.index, filepath.replace('.pkl', '.index'))
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        
        # Rebuild index
        if self.embeddings is not None:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

class ResearchAssistant:
    """Main research assistant class"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.documents = {}
        self.images = {}
    
    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Process a PDF file and return summary"""
        try:
            chunks, images = self.pdf_processor.extract_text_and_images(pdf_file)
            
            # Store documents
            doc_id = pdf_file.name
            self.documents[doc_id] = chunks
            self.images[doc_id] = images
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Generate summary
            summary = self._generate_summary(chunks)
            
            return {
                'doc_id': doc_id,
                'summary': summary,
                'chunks_count': len(chunks),
                'images_count': len(images),
                'sections': list(set([chunk.section for chunk in chunks]))
            }
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def _generate_summary(self, chunks: List[DocumentChunk]) -> Dict[str, str]:
        """Generate structured summary of the document"""
        sections = {}
        
        for chunk in chunks:
            if chunk.section not in sections:
                sections[chunk.section] = []
            sections[chunk.section].append(chunk.text)
        
        summary = {}
        for section, texts in sections.items():
            combined_text = ' '.join(texts)
            # Truncate to reasonable length
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000] + "..."
            summary[section] = combined_text
        
        return summary
    
    def answer_query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Answer a query using RAG"""
        # Search for relevant chunks
        results = self.vector_store.search(query, k)
        
        if not results:
            return {
                'answer': "No relevant information found in the documents.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Combine relevant chunks
        context_chunks = []
        sources = []
        
        for chunk, score in results:
            context_chunks.append(chunk.text)
            sources.append({
                'doc_id': chunk.doc_id,
                'page': chunk.page_num,
                'section': chunk.section,
                'score': score
            })
        
        context = '\n\n'.join(context_chunks)
        
        # Generate answer (simplified - in production use LLM)
        answer = self._generate_answer(query, context)
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'confidence': max([s['score'] for s in sources]) if sources else 0.0
        }
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context (simplified version)"""
        # This is a simplified answer generation
        # In production, you'd use OpenAI, Anthropic, or local LLM
        
        sentences = context.split('.')
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if query_words & sentence_words:  # If there's overlap
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]) + '.'
        else:
            return "Based on the available context, I cannot find a specific answer to your query."
    
    def analyze_image(self, doc_id: str, image_index: int) -> str:
        """Analyze an image from the document"""
        if doc_id not in self.images or image_index >= len(self.images[doc_id]):
            return "Image not found."
        
        image_data = self.images[doc_id][image_index]
        
        # Basic image analysis (in production, use vision models)
        width, height = image_data.image.size
        
        analysis = f"""
        Image Analysis (Page {image_data.page_num}):
        - Dimensions: {width} x {height} pixels
        - Location: {image_data.bbox}
        - Type: Figure/Chart/Graph
        
        Note: For detailed image analysis, integrate with vision models like GPT-4V or LLaVA.
        """
        
        return analysis

def main():
    st.set_page_config(page_title="PDF Research Assistant", layout="wide")
    
    st.title("Advanced PDF Research Assistant")
    st.markdown("Upload research papers and get AI-powered insights, summaries, and answers!")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload academic papers, research documents, or any PDF files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = st.session_state.assistant.process_pdf(uploaded_file)
                        if result:
                            st.success(f"Processed {result['doc_id']}")
                            st.json(result)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Questions")
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., What methodology was used in this study?"
        )
        
        if st.button("🔍 Search") and query:
            with st.spinner("Searching documents..."):
                response = st.session_state.assistant.answer_query(query)
                
                st.subheader("Answer")
                st.write(response['answer'])
                
                st.subheader("Sources")
                for i, source in enumerate(response['sources']):
                    st.write(f"**Source {i+1}:**")
                    st.write(f"- Document: {source['doc_id']}")
                    st.write(f"- Page: {source['page']}")
                    st.write(f"- Section: {source['section']}")
                    st.write(f"- Relevance Score: {source['score']:.3f}")
                    st.write("---")
                
                with st.expander("View Context"):
                    st.text(response['context'])
    
    with col2:
        st.header("Document Analysis")
        
        if st.session_state.assistant.documents:
            selected_doc = st.selectbox(
                "Select Document:",
                list(st.session_state.assistant.documents.keys())
            )
            
            if selected_doc:
                chunks = st.session_state.assistant.documents[selected_doc]
                images = st.session_state.assistant.images.get(selected_doc, [])
                
                st.metric("Total Chunks", len(chunks))
                st.metric("Images Found", len(images))
                
                # Show sections
                sections = list(set([chunk.section for chunk in chunks]))
                st.write("**Sections Found:**")
                for section in sections:
                    st.write(f"- {section.title()}")
                
                # Image analysis
                if images:
                    st.subheader("Images")
                    for i, img_data in enumerate(images):
                        st.write(f"**Image {i+1} (Page {img_data.page_num}):**")
                        st.image(img_data.image, width=200)
                        
                        if st.button(f"Analyze Image {i+1}"):
                            analysis = st.session_state.assistant.analyze_image(selected_doc, i)
                            st.text(analysis)
        
        else:
            st.info("Upload and process PDFs to see analysis options.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, SentenceTransformers, FAISS, and PyMuPDF")

if __name__ == "__main__":
    main()