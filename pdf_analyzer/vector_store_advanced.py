# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Tuple, Any, Optional
# import pickle
# import json
# from dataclasses import dataclass, asdict
# from collections import defaultdict
# import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import hashlib

# @dataclass
# class DocumentChunk:
#     """Enhanced document chunk with metadata"""
#     text: str
#     doc_id: str
#     chunk_id: str
#     page_num: int
#     section_type: str
#     word_count: int
#     key_terms: List[str]
#     embedding: Optional[np.ndarray] = None
#     metadata: Optional[Dict[str, Any]] = None

# @dataclass
# class SearchResult:
#     """Search result with detailed information"""
#     chunk: DocumentChunk
#     score: float
#     rank: int
#     search_type: str
#     explanation: str

# class MultiModalVectorStore:
#     """Advanced vector store supporting multiple embedding models and search strategies"""
    
#     def __init__(self, models_config: Dict[str, str] = None):
#         """
#         Initialize with multiple embedding models
        
#         Args:
#             models_config: Dict mapping model names to model identifiers
#         """
#         if models_config is None:
#             models_config = {
#                 'general': 'all-MiniLM-L6-v2',
#                 'scientific': 'allenai/scibert_scivocab_uncased',
#                 'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'
#             }
        
#         self.models = {}
#         self.indices = {}
#         self.chunks = []
#         self.chunk_map = {}  # chunk_id -> chunk
#         self.doc_map = defaultdict(list)  # doc_id -> chunks
#         self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
#         self.tfidf_matrix = None
        
#         # Load embedding models
#         for model_name, model_id in models_config.items():
#             try:
#                 print(f"Loading model: {model_name} ({model_id})")
#                 self.models[model_name] = SentenceTransformer(model_id)
#             except Exception as e:
#                 print(f"Failed to load model {model_name}: {e}")
    
#     def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 32):
#         """Add documents to the vector store with batch processing"""
#         print(f"Adding {len(chunks)} chunks to vector store...")
        
#         # Generate unique IDs if not provided
#         for i, chunk in enumerate(chunks):
#             if not chunk.chunk_id:
#                 chunk.chunk_id = self._generate_chunk_id(chunk.text, chunk.doc_id, i)
        
#         # Store chunks
#         self.chunks.extend(chunks)
#         for chunk in chunks:
#             self.chunk_map[chunk.chunk_id] = chunk
#             self.doc_map[chunk.doc_id].append(chunk)
        
#         # Generate embeddings for each model
#         for model_name, model in self.models.items():
#             print(f"Generating embeddings with {model_name} model...")
#             embeddings = self._generate_embeddings_batch(
#                 [chunk.text for chunk in chunks], 
#                 model, 
#                 batch_size
#             )
            
#             # Create or update FAISS index
#             if model_name not in self.indices:
#                 dimension = embeddings.shape[1]
#                 self.indices[model_name] = faiss.IndexFlatIP(dimension)
            
#             # Normalize embeddings for cosine similarity
#             faiss.normalize_L2(embeddings)
#             self.indices[model_name].add(embeddings)
        
#         # Generate TF-IDF vectors
#         print("Generating TF-IDF vectors...")
#         all_texts = [chunk.text for chunk in self.chunks]
#         self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
#         print(f"Vector store updated with {len(self.chunks)} total chunks")
    
#     def _generate_embeddings_batch(self, texts: List[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
#         """Generate embeddings in batches to handle memory efficiently"""
#         embeddings = []
        
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             batch_embeddings = model.encode(batch, convert_to_numpy=True)
#             embeddings.append(batch_embeddings)
        
#         return np.vstack(embeddings)
    
#     def _generate_chunk_id(self, text: str, doc_id: str, index: int) -> str:
#         """Generate unique chunk ID"""
#         content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
#         return f"{doc_id}_{index}_{content_hash}"
    
#     def search(self, 
#                query: str, 
#                k: int = 5, 
#                search_strategy: str = 'hybrid',
#                model_weights: Dict[str, float] = None,
#                filters: Dict[str, Any] = None) -> List[SearchResult]:
#         """
#         Advanced search with multiple strategies
        
#         Args:
#             query: Search query
#             k: Number of results to return
#             search_strategy: 'semantic', 'tfidf', 'hybrid', 'ensemble'
#             model_weights: Weights for different models in ensemble search
#             filters: Filters to apply (doc_id, section_type, etc.)
#         """
#         if not self.chunks:
#             return []
        
#         if search_strategy == 'semantic':
#             return self._semantic_search(query, k, filters)
#         elif search_strategy == 'tfidf':
#             return self._tfidf_search(query, k, filters)
#         elif search_strategy == 'hybrid':
#             return self._hybrid_search(query, k, filters)
#         elif search_strategy == 'ensemble':
#             return self._ensemble_search(query, k, model_weights, filters)
#         else:
#             raise ValueError(f"Unknown search strategy: {search_strategy}")
    
#     def _semantic_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[SearchResult]:
#         """Semantic search using default model"""
#         if 'general' not in self.models:
#             return []
        
#         model = self.models['general']
#         index = self.indices['general']
        
#         # Encode query
#         query_embedding = model.encode([query])
#         faiss.normalize_L2(query_embedding)
        
#         # Search
#         scores, indices = index.search(query_embedding, min(k * 2, len(self.chunks)))
        
#         results = []
#         for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
#             if idx >= 0 and idx < len(self.chunks):
#                 chunk = self.chunks[idx]
#                 if self._apply_filters(chunk, filters):
#                     result = SearchResult(
#                         chunk=chunk,
#                         score=float(score),
#                         rank=len(results) + 1,
#                         search_type='semantic',
#                         explanation=f'Semantic similarity: {score:.3f}'
#                     )
#                     results.append(result)
#                     if len(results) >= k:
#                         break
        
#         return results
    
#     def _tfidf_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[SearchResult]:
#         """TF-IDF based search"""
#         if self.tfidf_matrix is None:
#             return []
        
#         # Transform query
#         query_vector = self.tfidf_vectorizer.transform([query])
        
#         # Calculate similarities
#         similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
#         # Get top k indices
#         top_indices = np.argsort(similarities)[::-1]
        
#         results = []
#         for i, idx in enumerate(top_indices):
#             if len(results) >= k:
#                 break
            
#             chunk = self.chunks[idx]
#             if similarities[idx] > 0 and self._apply_filters(chunk, filters):
#                 result = SearchResult(
#                     chunk=chunk,
#                     score=float(similarities[idx]),
#                     rank=len(results) + 1,
#                     search_type='tfidf',
#                     explanation=f'TF-IDF similarity: {similarities[idx]:.3f}'
#                 )
#                 results.append(result)
        
#         return results
    
#     def _hybrid_search(self, query: str, k: int, filters: Dict[str, Any] = None, 
#                       semantic_weight: float = 0.7, tfidf_weight: float = 0.3) -> List[SearchResult]:
#         """Hybrid search combining semantic and TF-IDF"""
#         # Get results from both methods
#         semantic_results = self._semantic_search(query, k * 2, filters)
#         tfidf_results = self._tfidf_search(query, k * 2, filters)
        
#         # Combine scores
#         score_map = {}
        
#         # Add semantic scores
#         for result in semantic_results:
#             chunk_id = result.chunk.chunk_id
#             score_map[chunk_id] = score_map.get(chunk_id, 0) + result.score * semantic_weight
        
#         # Add TF-IDF scores
#         for result in tfidf_results:
#             chunk_id = result.chunk.chunk_id
#             score_map[chunk_id] = score_map.get(chunk_id, 0) + result.score * tfidf_weight
        
#         # Sort by combined score
#         sorted_chunks = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
#         results = []
#         for i, (chunk_id, combined_score) in enumerate(sorted_chunks[:k]):
#             chunk = self.chunk_map[chunk_id]
#             result = SearchResult(
#                 chunk=chunk,
#                 score=combined_score,
#                 rank=i + 1,
#                 search_type='hybrid',
#                 explanation=f'Hybrid score (semantic: {semantic_weight}, tfidf: {tfidf_weight}): {combined_score:.3f}'
#             )
#             results.append(result)
        
#         return results
    
#     def _ensemble_search(self, query: str, k: int, model_weights: Dict[str, float] = None, 
#                         filters: Dict[str, Any] = None) -> List[SearchResult]:
#         """Ensemble search using multiple models"""
#         if model_weights is None:
#             model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
#         score_map = {}
        
#         # Get results from each model
#         for model_name, weight in model_weights.items():
#             if model_name not in self.models or model_name not in self.indices:
#                 continue
            
#             model = self.models[model_name]
#             index = self.indices[model_name]
            
#             # Encode query
#             query_embedding = model.encode([query])
#             faiss.normalize_L2(query_embedding)
            
#             # Search
#             scores, indices = index.search(query_embedding, min(k * 2, len(self.chunks)))
            
#             for score, idx in zip(scores[0], indices[0]):
#                 if idx >= 0 and idx < len(self.chunks):
#                     chunk = self.chunks[idx]
#                     if self._apply_filters(chunk, filters):
#                         chunk_id = chunk.chunk_id
#                         score_map[chunk_id] = score_map.get(chunk_id, 0) + float(score) * weight
        
#         # Sort by combined score
#         sorted_chunks = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
#         results = []
#         for i, (chunk_id, combined_score) in enumerate(sorted_chunks[:k]):
#             chunk = self.chunk_map[chunk_id]
#             result = SearchResult(
#                 chunk=chunk,
#                 score=combined_score,
#                 rank=i + 1,
#                 search_type='ensemble',
#                 explanation=f'Ensemble score from {len(model_weights)} models: {combined_score:.3f}'
#             )
#             results.append(result)
        
#         return results
    
#     def _apply_filters(self, chunk: DocumentChunk, filters: Dict[str, Any] = None) -> bool:
#         """Apply filters to search results"""
#         if not filters:
#             return True
        
#         for filter_key, filter_value in filters.items():
#             if filter_key == 'doc_id' and chunk.doc_id != filter_value:
#                 return False
#             elif filter_key == 'section_type' and chunk.section_type != filter_value:
#                 return False
#             elif filter_key == 'min_words' and chunk.word_count < filter_value:
#                 return False
#             elif filter_key == 'max_words' and chunk.word_count > filter_value:
#                 return False
#             elif filter_key == 'page_range':
#                 if not (filter_value[0] <= chunk.page_num <= filter_value[1]):
#                     return False
        
#         return True
    
#     def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[SearchResult]:
#         """Find chunks similar to a given chunk"""
#         if chunk_id not in self.chunk_map:
#             return []
        
#         chunk = self.chunk_map[chunk_id]
#         return self.search(chunk.text, k + 1)[1:]  # Exclude the chunk itself
    
#     def get_document_summary(self, doc_id: str) -> Dict[str, Any]:
#         """Get summary of a specific document"""
#         if doc_id not in self.doc_map:
#             return {}
        
#         chunks = self.doc_map[doc_id]
        
#         # Group by section type
#         sections = defaultdict(list)
#         for chunk in chunks:
#             sections[chunk.section_type].append(chunk)
        
#         summary = {
#             'doc_id': doc_id,
#             'total_chunks': len(chunks),
#             'total_words': sum(chunk.word_count for chunk in chunks),
#             'sections': {
#                 section_type: {
#                     'chunk_count': len(section_chunks),
#                     'word_count': sum(chunk.word_count for chunk in section_chunks),
#                     'key_terms': list(set(term for chunk in section_chunks for term in chunk.key_terms))[:20]
#                 }
#                 for section_type, section_chunks in sections.items()
#             },
#             'page_range': (
#                 min(chunk.page_num for chunk in chunks),
#                 max(chunk.page_num for chunk in chunks)
#             )
#         }
        
#         return summary
    
#     def save_to_disk(self, base_path: str):
#         """Save vector store to disk"""
#         # Save chunks and metadata
#         chunks_data = [asdict(chunk) for chunk in self.chunks]
#         with open(f"{base_path}_chunks.json", 'w') as f:
#             json.dump(chunks_data, f, indent=2)
        
#         # Save FAISS indices
#         for model_name, index in self.indices.items():
#             faiss.write_index(index, f"{base_path}_{model_name}.index")
        
#         # Save TF-IDF vectorizer and matrix
#         if self.tfidf_vectorizer and self.tfidf_matrix is not None:
#             with open(f"{base_path}_tfidf.pkl", 'wb') as f:
#                 pickle.dump({
#                     'vectorizer': self.tfidf_vectorizer,
#                     'matrix': self.tfidf_matrix
#                 }, f)
        
#         # Save configuration
#         config = {
#             'models': list(self.models.keys()),
#             'chunk_count': len(self.chunks)
#         }
#         with open(f"{base_path}_config.json", 'w') as f:
#             json.dump(config, f, indent=2)
        
#         print(f"Vector store saved to {base_path}")
    
#     def load_from_disk(self, base_path: str):
#         """Load vector store from disk"""
#         # Load chunks
#         with open(f"{base_path}_chunks.json", 'r') as f:
#             chunks_data = json.load(f)
        
#         self.chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
        
#         # Rebuild maps
#         self.chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
#         self.doc_map = defaultdict(list)
#         for chunk in self.chunks:
#             self.doc_map[chunk.doc_id].append(chunk)
        
#         # Load FAISS indices
#         for model_name in self.models.keys():
#             index_path = f"{base_path}_{model_name}.index"
#             try:
#                 self.indices[model_name] = faiss.read_index(index_path)
#             except Exception as e:
#                 print(f"Could not load index for {model_name}: {e}")
        
#         # Load TF-IDF
#         tfidf_path = f"{base_path}_tfidf.pkl"
#         try:
#             with open(tfidf_path, 'rb') as f:
#                 tfidf_data = pickle.load(f)
#                 self.tfidf_vectorizer = tfidf_data['vectorizer']
#                 self.tfidf_matrix = tfidf_data['matrix']
#         except Exception as e:
#             print(f"Could not load TF-IDF data: {e}")
        
#         print(f"Vector store loaded from {base_path} with {len(self.chunks)} chunks")
    
#     def get_statistics(self) -> Dict[str, Any]:
#         """Get vector store statistics"""
#         if not self.chunks:
#             return {}
        
#         # Document statistics
#         doc_stats = {}
#         for doc_id, chunks in self.doc_map.items():
#             doc_stats[doc_id] = {
#                 'chunks': len(chunks),
#                 'words': sum(chunk.word_count for chunk in chunks),
#                 'sections': len(set(chunk.section_type for chunk in chunks))
#             }
        
#         # Section statistics
#         section_stats = defaultdict(int)
#         for chunk in self.chunks:
#             section_stats[chunk.section_type] += 1
        
#         stats = {
#             'total_chunks': len(self.chunks),
#             'total_documents': len(self.doc_map),
#             'total_words': sum(chunk.word_count for chunk in self.chunks),
#             'avg_words_per_chunk': sum(chunk.word_count for chunk in self.chunks) / len(self.chunks),
#             'models_loaded': list(self.models.keys()),
#             'indices_available': list(self.indices.keys()),
#             'documents': doc_stats,
#             'section_distribution': dict(section_stats),
#             'has_tfidf': self.tfidf_matrix is not None
#         }
        
#         return stats


import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional
import pickle
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Enhanced document chunk with metadata"""
    text: str
    doc_id: str
    chunk_id: str
    page_num: int
    section_type: str
    word_count: int
    key_terms: List[str]
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """Search result with detailed information"""
    chunk: DocumentChunk
    score: float
    rank: int
    search_type: str
    explanation: str

class MultiModalVectorStore:
    """Advanced vector store supporting multiple embedding models and search strategies"""

    def __init__(self, models_config: Dict[str, str] = None):
        """
        Initialize with multiple embedding models

        Args:
            models_config: Dict mapping model names to sentence-transformers model identifiers
        """
        if models_config is None:
            models_config = {
                "general": "all-MiniLM-L6-v2",
                "scientific": "allenai-specter",  # strong for papers
                "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
            }

        self.models: Dict[str, SentenceTransformer] = {}
        self.indices: Dict[str, faiss.IndexIDMap2] = {}
        self._id_maps: Dict[str, Dict[int, int]] = {}  # faiss_id -> global chunk idx per model
        self.chunks: List[DocumentChunk] = []
        self.chunk_map: Dict[str, DocumentChunk] = {}  # chunk_id -> chunk
        self.doc_map: Dict[str, List[DocumentChunk]] = defaultdict(list)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
        self.tfidf_matrix = None

        # Load embedding models
        for name, ident in models_config.items():
            try:
                logger.info(f"Loading embedding model: {name} ({ident})")
                self.models[name] = SentenceTransformer(ident)
            except Exception as e:
                logger.warning(f"Failed to load model {name} ({ident}): {e}")

        # internal running FAISS id counter per model
        self._next_faiss_id: Dict[str, int] = {name: 0 for name in self.models}

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------
    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 32):
        """Add documents to the vector store with batch processing and stable FAISS IDs."""
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        # Generate chunk IDs if missing
        for i, c in enumerate(chunks):
            if not c.chunk_id:
                c.chunk_id = self._generate_chunk_id(c.text, c.doc_id, i)

        start_index = len(self.chunks)
        self.chunks.extend(chunks)
        for c in chunks:
            self.chunk_map[c.chunk_id] = c
            self.doc_map[c.doc_id].append(c)

        # Embeddings & FAISS (ID-mapped)
        for model_name, model in self.models.items():
            texts = [c.text for c in chunks]
            embs = self._generate_embeddings_batch(texts, model, batch_size)  # (N, D)
            # Normalize for cosine similarity
            faiss.normalize_L2(embs)

            # Build or append to Index
            if model_name not in self.indices:
                dim = embs.shape[1]
                self.indices[model_name] = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
                self._id_maps[model_name] = {}

            # Assign stable ids for these vectors
            ids = np.arange(self._next_faiss_id[model_name],
                            self._next_faiss_id[model_name] + len(chunks)).astype("int64")
            self._next_faiss_id[model_name] += len(chunks)

            self.indices[model_name].add_with_ids(embs, ids)

            # Map faiss id -> global chunk idx
            for local_i, fid in enumerate(ids):
                self._id_maps[model_name][int(fid)] = start_index + local_i

        # TF-IDF global matrix (re-fit on all texts)
        logger.info("Fitting TF-IDF on corpus...")
        all_texts = [c.text for c in self.chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        logger.info(f"Vector store now has {len(self.chunks)} total chunks.")

    def _generate_embeddings_batch(self, texts: List[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
        outs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            outs.append(model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False))
        return np.vstack(outs)

    def _generate_chunk_id(self, text: str, doc_id: str, index: int) -> str:
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index}_{content_hash}"

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        search_strategy: str = "hybrid",
        model_weights: Dict[str, float] = None,
        filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        if not self.chunks:
            return []

        if search_strategy == "semantic":
            return self._semantic_search(query, k, filters)
        if search_strategy == "tfidf":
            return self._tfidf_search(query, k, filters)
        if search_strategy == "hybrid":
            return self._hybrid_search(query, k, filters)
        if search_strategy == "ensemble":
            return self._ensemble_search(query, k, model_weights, filters)
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    def _semantic_search(self, query: str, k: int, filters: Dict[str, Any] = None, model_name: str = "general") -> List[SearchResult]:
        if model_name not in self.models or model_name not in self.indices:
            return []
        model = self.models[model_name]
        index = self.indices[model_name]

        q = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        scores, ids = index.search(q, min(k * 4, max(1, index.ntotal)))
        results: List[SearchResult] = []
        for score, fid in zip(scores[0], ids[0]):
            if int(fid) == -1:
                continue
            chunk_idx = self._id_maps[model_name].get(int(fid))
            if chunk_idx is None:
                continue
            c = self.chunks[chunk_idx]
            if self._apply_filters(c, filters):
                results.append(
                    SearchResult(
                        chunk=c,
                        score=float(score),  # already inner-product on normalized vectors ~ cosine
                        rank=len(results) + 1,
                        search_type=f"semantic:{model_name}",
                        explanation=f"Semantic similarity ({model_name}): {float(score):.3f}",
                    )
                )
                if len(results) >= k:
                    break
        return results

    def _tfidf_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[SearchResult]:
        if self.tfidf_matrix is None:
            return []
        qv = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf_matrix).flatten()
        order = np.argsort(sims)[::-1]
        results: List[SearchResult] = []
        for idx in order:
            if len(results) >= k:
                break
            score = float(sims[idx])
            if score <= 0:
                break
            c = self.chunks[idx]
            if self._apply_filters(c, filters):
                results.append(
                    SearchResult(
                        chunk=c,
                        score=score,
                        rank=len(results) + 1,
                        search_type="tfidf",
                        explanation=f"TF-IDF similarity: {score:.3f}",
                    )
                )
        return results

    def _hybrid_search(self, query: str, k: int, filters: Dict[str, Any] = None, semantic_weight: float = 0.7, tfidf_weight: float = 0.3) -> List[SearchResult]:
        sem = self._semantic_search(query, k * 4, filters, model_name="general")
        tf = self._tfidf_search(query, k * 4, filters)

        comb: Dict[str, float] = {}  # chunk_id -> score
        for r in sem:
            comb[r.chunk.chunk_id] = comb.get(r.chunk.chunk_id, 0.0) + r.score * semantic_weight
        for r in tf:
            comb[r.chunk.chunk_id] = comb.get(r.chunk.chunk_id, 0.0) + r.score * tfidf_weight

        ranked = sorted(comb.items(), key=lambda x: x[1], reverse=True)
        out: List[SearchResult] = []
        for i, (cid, sc) in enumerate(ranked[:k]):
            c = self.chunk_map[cid]
            out.append(
                SearchResult(
                    chunk=c,
                    score=float(sc),
                    rank=i + 1,
                    search_type="hybrid",
                    explanation=f"Hybrid score (semantic {semantic_weight:.2f} + tfidf {tfidf_weight:.2f}): {sc:.3f}",
                )
            )
        return out

    def _ensemble_search(self, query: str, k: int, model_weights: Dict[str, float] = None, filters: Dict[str, Any] = None) -> List[SearchResult]:
        if model_weights is None:
            # equal weights
            model_weights = {name: 1.0 / max(1, len(self.models)) for name in self.models.keys()}
        comb: Dict[str, float] = {}
        for model_name, w in model_weights.items():
            res = self._semantic_search(query, k * 3, filters, model_name=model_name)
            for r in res:
                comb[r.chunk.chunk_id] = comb.get(r.chunk.chunk_id, 0.0) + w * r.score
        ranked = sorted(comb.items(), key=lambda x: x[1], reverse=True)
        out: List[SearchResult] = []
        for i, (cid, sc) in enumerate(ranked[:k]):
            out.append(
                SearchResult(
                    chunk=self.chunk_map[cid],
                    score=float(sc),
                    rank=i + 1,
                    search_type="ensemble",
                    explanation=f"Ensemble ({len(model_weights)} models) score: {sc:.3f}",
                )
            )
        return out

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _apply_filters(self, chunk: DocumentChunk, filters: Dict[str, Any] = None) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if k == "doc_id" and chunk.doc_id != v:
                return False
            if k == "section_type" and chunk.section_type != v:
                return False
            if k == "min_words" and chunk.word_count < v:
                return False
            if k == "max_words" and chunk.word_count > v:
                return False
            if k == "page_range":
                lo, hi = v
                if not (lo <= chunk.page_num <= hi):
                    return False
        return True

    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[SearchResult]:
        if chunk_id not in self.chunk_map:
            return []
        # use the chunk's own text to search; exclude itself
        q = self.chunk_map[chunk_id].text
        res = self.search(q, k + 1, "hybrid")
        return [r for r in res if r.chunk.chunk_id != chunk_id][:k]

    def get_document_summary(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.doc_map:
            return {}
        chunks = self.doc_map[doc_id]
        from collections import defaultdict
        sections = defaultdict(list)
        for c in chunks:
            sections[c.section_type].append(c)
        summary = {
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "total_words": sum(c.word_count for c in chunks),
            "sections": {
                s: {
                    "chunk_count": len(cs),
                    "word_count": sum(c.word_count for c in cs),
                    "key_terms": list({t for c in cs for t in c.key_terms})[:20],
                } for s, cs in sections.items()
            },
            "page_range": (min(c.page_num for c in chunks), max(c.page_num for c in chunks)),
        }
        return summary

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save_to_disk(self, base_path: str):
        chunks_data = [asdict(c) for c in self.chunks]
        with open(f"{base_path}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        for model_name, index in self.indices.items():
            faiss.write_index(index, f"{base_path}_{model_name}.index")
            # save id map and counter
            with open(f"{base_path}_{model_name}_ids.json", "w") as f:
                json.dump({
                    "id_map": self._id_maps.get(model_name, {}),
                    "next_id": self._next_faiss_id.get(model_name, 0)
                }, f)

        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            with open(f"{base_path}_tfidf.pkl", "wb") as f:
                pickle.dump({"vectorizer": self.tfidf_vectorizer, "matrix": self.tfidf_matrix}, f)

        with open(f"{base_path}_config.json", "w") as f:
            json.dump({"models": list(self.models.keys()), "chunk_count": len(self.chunks)}, f, indent=2)
        logger.info(f"Vector store saved to {base_path}")

    def load_from_disk(self, base_path: str):
        with open(f"{base_path}_chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [DocumentChunk(**cd) for cd in chunks_data]
        self.chunk_map = {c.chunk_id: c for c in self.chunks}
        self.doc_map = defaultdict(list)
        for c in self.chunks:
            self.doc_map[c.doc_id].append(c)

        # Rebuild FAISS indices & id maps
        for model_name in self.models.keys():
            try:
                self.indices[model_name] = faiss.read_index(f"{base_path}_{model_name}.index")
                with open(f"{base_path}_{model_name}_ids.json", "r") as f:
                    meta = json.load(f)
                # keys in json are strings, cast to int
                self._id_maps[model_name] = {int(k): int(v) for k, v in meta.get("id_map", {}).items()}
                self._next_faiss_id[model_name] = int(meta.get("next_id", 0))
            except Exception as e:
                logger.warning(f"Could not load FAISS/index metadata for {model_name}: {e}")

        try:
            with open(f"{base_path}_tfidf.pkl", "rb") as f:
                data = pickle.load(f)
                self.tfidf_vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
        except Exception as e:
            logger.warning(f"Could not load TF-IDF data: {e}")

        logger.info(f"Vector store loaded from {base_path} with {len(self.chunks)} chunks")

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        if not self.chunks:
            return {}
        doc_stats = {}
        for d, cs in self.doc_map.items():
            doc_stats[d] = {
                "chunks": len(cs),
                "words": sum(c.word_count for c in cs),
                "sections": len(set(c.section_type for c in cs)),
            }
        section_stats = defaultdict(int)
        for c in self.chunks:
            section_stats[c.section_type] += 1

        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.doc_map),
            "total_words": sum(c.word_count for c in self.chunks),
            "avg_words_per_chunk": (sum(c.word_count for c in self.chunks) / len(self.chunks)) if self.chunks else 0.0,
            "models_loaded": list(self.models.keys()),
            "indices_available": list(self.indices.keys()),
            "documents": doc_stats,
            "section_distribution": dict(section_stats),
            "has_tfidf": self.tfidf_matrix is not None,
        }
