# import openai
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# import re
# from datetime import datetime
# import json
# import logging
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# from sentence_transformers import SentenceTransformer, util
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class RAGResponse:
#     """RAG response with detailed metadata"""
#     answer: str
#     sources: List[Dict[str, Any]]
#     confidence: float
#     context_used: str
#     reasoning: str
#     search_strategy: str
#     timestamp: datetime
#     tokens_used: Optional[int] = None
#     response_time: Optional[float] = None

# class AdvancedRAGEngine:
#     """Advanced RAG engine with multiple LLM backends and reasoning"""
    
#     def __init__(self, 
#                  vector_store,
#                  llm_provider: str = "local",  # "openai", "local", "huggingface"
#                  model_config: Dict[str, Any] = None):
#         """
#         Initialize RAG engine
        
#         Args:
#             vector_store: MultiModalVectorStore instance
#             llm_provider: LLM provider to use
#             model_config: Configuration for the LLM
#         """
#         self.vector_store = vector_store
#         self.llm_provider = llm_provider
#         self.model_config = model_config or {}
        
#         # Initialize LLM based on provider
#         self._initialize_llm()
        
#         # Prompt templates
#         self.prompt_templates = {
#             'qa': """Based on the following context from research papers, answer the question accurately and cite your sources.

# Context:
# {context}

# Question: {question}

# Instructions:
# 1. Provide a comprehensive answer based on the context
# 2. Cite specific sources using [Source: Document_name, Page X, Section Y] format
# 3. If information is insufficient, clearly state what's missing
# 4. Maintain scientific accuracy and objectivity

# Answer:""",

#             'summarize': """Summarize the key findings from the following research context:

# Context:
# {context}

# Instructions:
# 1. Extract main findings and conclusions
# 2. Organize information by themes or sections
# 3. Highlight significant results, methodologies, or implications
# 4. Cite sources for major claims

# Summary:""",

#             'analyze': """Analyze the following research content and provide insights:

# Context:
# {context}

# Question: {question}

# Instructions:
# 1. Provide detailed analysis addressing the question
# 2. Compare different findings if multiple sources are present
# 3. Identify patterns, contradictions, or gaps
# 4. Suggest implications or future research directions
# 5. Cite all sources used

# Analysis:""",

#             'methodology': """Extract and explain the methodology from the following research context:

# Context:
# {context}

# Instructions:
# 1. Describe the research methods used
# 2. Explain the experimental design or approach
# 3. List tools, techniques, or software mentioned
# 4. Identify sample sizes and data collection methods
# 5. Note any limitations mentioned by authors

# Methodology Summary:"""
#         }
    
#     def _initialize_llm(self):
#         """Initialize the language model based on provider"""
#         if self.llm_provider == "openai":
#             # OpenAI API configuration
#             self.client = openai.OpenAI(
#                 api_key=self.model_config.get('api_key', ''),
#             )
#             self.model_name = self.model_config.get('model', 'gpt-3.5-turbo')
            
#         elif self.llm_provider == "local":
#             # Local model using transformers
#             model_name = self.model_config.get('model', 'microsoft/DialoGPT-medium')
#             try:
#                 self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#                 self.model = AutoModelForCausalLM.from_pretrained(model_name)
#                 if self.tokenizer.pad_token is None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
#             except Exception as e:
#                 logger.warning(f"Could not load local model {model_name}: {e}")
#                 # Fallback to pipeline
#                 try:
#                     self.pipeline = pipeline(
#                         "text-generation", 
#                         model="microsoft/DialoGPT-small",
#                         device=0 if torch.cuda.is_available() else -1
#                     )
#                 except Exception as e2:
#                     logger.error(f"Could not initialize any local model: {e2}")
#                     self.pipeline = None
                    
#         elif self.llm_provider == "huggingface":
#             # Hugging Face pipeline
#             model_name = self.model_config.get('model', 'google/flan-t5-base')
#             try:
#                 self.pipeline = pipeline(
#                     "text2text-generation",
#                     model=model_name,
#                     device=0 if torch.cuda.is_available() else -1
#                 )
#             except Exception as e:
#                 logger.error(f"Could not load Hugging Face model {model_name}: {e}")
#                 self.pipeline = None
    
#     def query(self, 
#               question: str,
#               query_type: str = "qa",
#               search_strategy: str = "hybrid",
#               k: int = 5,
#               filters: Dict[str, Any] = None,
#               context_length: int = 4000) -> RAGResponse:
#         """
#         Process a query using RAG
        
#         Args:
#             question: The user's question
#             query_type: Type of query (qa, summarize, analyze, methodology)
#             search_strategy: Search strategy for vector store
#             k: Number of chunks to retrieve
#             filters: Filters for search
#             context_length: Maximum context length in characters
#         """
#         start_time = datetime.now()
        
#         try:
#             # Step 1: Retrieve relevant chunks
#             search_results = self.vector_store.search(
#                 query=question,
#                 k=k,
#                 search_strategy=search_strategy,
#                 filters=filters
#             )
            
#             if not search_results:
#                 return RAGResponse(
#                     answer="I couldn't find any relevant information in the documents to answer your question.",
#                     sources=[],
#                     confidence=0.0,
#                     context_used="",
#                     reasoning="No relevant chunks found in vector search",
#                     search_strategy=search_strategy,
#                     timestamp=start_time
#                 )
            
#             # Step 2: Prepare context
#             context, sources_info = self._prepare_context(search_results, context_length)
            
#             # Step 3: Generate prompt
#             prompt = self._generate_prompt(question, context, query_type)
            
#             # Step 4: Generate answer
#             answer, tokens_used = self._generate_answer(prompt, query_type)
            
#             # Step 5: Calculate confidence
#             confidence = self._calculate_confidence(search_results, answer)
            
#             # Step 6: Generate reasoning
#             reasoning = self._generate_reasoning(search_results, answer)
            
#             end_time = datetime.now()
#             response_time = (end_time - start_time).total_seconds()
            
#             return RAGResponse(
#                 answer=answer,
#                 sources=sources_info,
#                 confidence=confidence,
#                 context_used=context,
#                 reasoning=reasoning,
#                 search_strategy=search_strategy,
#                 timestamp=start_time,
#                 tokens_used=tokens_used,
#                 response_time=response_time
#             )
            
#         except Exception as e:
#             logger.error(f"Error in RAG query: {e}")
#             return RAGResponse(
#                 answer=f"An error occurred while processing your question: {str(e)}",
#                 sources=[],
#                 confidence=0.0,
#                 context_used="",
#                 reasoning="Error in processing",
#                 search_strategy=search_strategy,
#                 timestamp=start_time
#             )
    
#     def _prepare_context(self, search_results, max_length: int) -> Tuple[str, List[Dict]]:
#         """Prepare context from search results"""
#         context_parts = []
#         sources_info = []
#         current_length = 0
        
#         for i, result in enumerate(search_results):
#             chunk = result.chunk
            
#             # Create source info
#             source_info = {
#                 'doc_id': chunk.doc_id,
#                 'page_num': chunk.page_num,
#                 'section_type': chunk.section_type,
#                 'score': result.score,
#                 'rank': result.rank,
#                 'search_type': result.search_type
#             }
#             sources_info.append(source_info)
            
#             # Add to context if within length limit
#             chunk_text = f"\n\n[Source {i+1}: {chunk.doc_id}, Page {chunk.page_num}, {chunk.section_type.title()}]\n{chunk.text}"
            
#             if current_length + len(chunk_text) <= max_length:
#                 context_parts.append(chunk_text)
#                 current_length += len(chunk_text)
#             else:
#                 # Truncate if necessary
#                 remaining_space = max_length - current_length
#                 if remaining_space > 200:  # Only add if meaningful space left
#                     truncated_text = chunk_text[:remaining_space] + "..."
#                     context_parts.append(truncated_text)
#                 break
        
#         context = "".join(context_parts)
#         return context, sources_info
    
#     def _generate_prompt(self, question: str, context: str, query_type: str) -> str:
#         """Generate prompt based on query type"""
#         template = self.prompt_templates.get(query_type, self.prompt_templates['qa'])
#         return template.format(question=question, context=context)
    
#     def _generate_answer(self, prompt: str, query_type: str) -> Tuple[str, Optional[int]]:
#         """Generate answer using the configured LLM"""
#         tokens_used = None
        
#         if self.llm_provider == "openai":
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful research assistant that provides accurate, well-sourced answers based on academic papers."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=self.model_config.get('max_tokens', 1000),
#                     temperature=self.model_config.get('temperature', 0.1)
#                 )
#                 answer = response.choices[0].message.content
#                 tokens_used = response.usage.total_tokens
                
#             except Exception as e:
#                 logger.error(f"OpenAI API error: {e}")
#                 answer = "Error generating response with OpenAI API."
                
#         elif self.llm_provider == "local":
#             try:
#                 if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
#                     # Use direct model inference
#                     inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=512)
                    
#                     with torch.no_grad():
#                         outputs = self.model.generate(
#                             inputs,
#                             max_length=inputs.shape[1] + 200,
#                             temperature=0.1,
#                             do_sample=True,
#                             pad_token_id=self.tokenizer.eos_token_id
#                         )
                    
#                     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                     answer = response[len(prompt):].strip()
                    
#                 elif hasattr(self, 'pipeline') and self.pipeline:
#                     # Use pipeline
#                     response = self.pipeline(
#                         prompt,
#                         max_length=len(prompt) + 200,
#                         temperature=0.1,
#                         do_sample=True
#                     )
#                     answer = response[0]['generated_text'][len(prompt):].strip()
                    
#                 else:
#                     answer = "Local model not available. Please configure a proper model."
                    
#             except Exception as e:
#                 logger.error(f"Local model error: {e}")
#                 answer = "Error generating response with local model."
                
#         elif self.llm_provider == "huggingface":
#             try:
#                 if hasattr(self, 'pipeline') and self.pipeline:
#                     response = self.pipeline(
#                         prompt,
#                         max_length=200,
#                         temperature=0.1
#                     )
#                     answer = response[0]['generated_text']
#                 else:
#                     answer = "Hugging Face model not available."
                    
#             except Exception as e:
#                 logger.error(f"Hugging Face model error: {e}")
#                 answer = "Error generating response with Hugging Face model."
        
#         else:
#             answer = "No valid LLM provider configured."
        
#         # Post-process answer
#         answer = self._post_process_answer(answer)
        
#         return answer, tokens_used
    
#     def _post_process_answer(self, answer: str) -> str:
#         """Post-process the generated answer"""
#         # Remove any remaining prompt text
#         answer = answer.strip()
        
#         # Clean up common artifacts
#         answer = re.sub(r'\n+', '\n', answer)  # Multiple newlines
#         answer = re.sub(r'^\s*Answer:\s*', '', answer, flags=re.IGNORECASE)  # Remove "Answer:" prefix
        
#         # Ensure proper citation format
#         answer = re.sub(r'\[Source:([^\]]+)\]', r'[Source:\1]', answer)
        
#         return answer
    
#     def _calculate_confidence(self, search_results, answer: str) -> float:
#         """Calculate confidence score for the answer"""
#         if not search_results:
#             return 0.0
        
#         # Base confidence on search result scores
#         avg_search_score = sum(result.score for result in search_results) / len(search_results)
        
#         # Adjust based on answer quality indicators
#         quality_score = 1.0
        
#         # Check if answer contains citations
#         citation_count = len(re.findall(r'\[Source:', answer))
#         if citation_count > 0:
#             quality_score += 0.1 * min(citation_count, 3)  # Bonus for citations, capped
        
#         # Check answer length (not too short, not too long)
#         answer_length = len(answer.split())
#         if 20 <= answer_length <= 300:
#             quality_score += 0.1
#         elif answer_length < 10:
#             quality_score -= 0.2
        
#         # Check for uncertainty indicators
#         uncertainty_phrases = ['not sure', 'unclear', 'insufficient information', 'cannot determine']
#         if any(phrase in answer.lower() for phrase in uncertainty_phrases):
#             quality_score -= 0.2
        
#         # Calculate final confidence
#         confidence = min(avg_search_score * quality_score, 1.0)
#         return max(confidence, 0.0)  # Ensure non-negative
    
#     def _generate_reasoning(self, search_results, answer: str) -> str:
#         """Generate reasoning explanation for the answer"""
#         reasoning_parts = []
        
#         # Explain search results
#         reasoning_parts.append(f"Found {len(search_results)} relevant chunks from the document corpus.")
        
#         # Top sources
#         if search_results:
#             top_source = search_results[0]
#             reasoning_parts.append(f"Most relevant source: {top_source.chunk.doc_id} (page {top_source.chunk.page_num}, {top_source.chunk.section_type}) with similarity score {top_source.score:.3f}")
        
#         # Document diversity
#         unique_docs = len(set(result.chunk.doc_id for result in search_results))
#         if unique_docs > 1:
#             reasoning_parts.append(f"Answer synthesized from {unique_docs} different documents.")
        
#         # Section types used
#         section_types = list(set(result.chunk.section_type for result in search_results))
#         if len(section_types) > 1:
#             reasoning_parts.append(f"Information drawn from sections: {', '.join(section_types)}")
        
#         return " ".join(reasoning_parts)
    
#     def batch_process(self, queries: List[Dict[str, Any]]) -> List[RAGResponse]:
#         """Process multiple queries in batch"""
#         responses = []
        
#         for query_config in queries:
#             question = query_config.get('question', '')
#             query_type = query_config.get('type', 'qa')
#             search_strategy = query_config.get('search_strategy', 'hybrid')
            
#             response = self.query(
#                 question=question,
#                 query_type=query_type,
#                 search_strategy=search_strategy,
#                 k=query_config.get('k', 5),
#                 filters=query_config.get('filters'),
#                 context_length=query_config.get('context_length', 4000)
#             )
            
#             responses.append(response)
        
#         return responses
    
#     def explain_answer(self, response: RAGResponse) -> Dict[str, Any]:
#         """Provide detailed explanation of how the answer was generated"""
#         explanation = {
#             'search_strategy': response.search_strategy,
#             'sources_used': len(response.sources),
#             'confidence_score': response.confidence,
#             'reasoning': response.reasoning,
#             'context_length': len(response.context_used),
#             'response_time': response.response_time,
#             'tokens_used': response.tokens_used,
#             'source_breakdown': {},
#             'quality_metrics': {}
#         }
        
#         # Source breakdown
#         doc_counts = {}
#         section_counts = {}
        
#         for source in response.sources:
#             doc_id = source['doc_id']
#             section = source['section_type']
            
#             doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
#             section_counts[section] = section_counts.get(section, 0) + 1
        
#         explanation['source_breakdown'] = {
#             'documents': doc_counts,
#             'sections': section_counts
#         }
        
#         # Quality metrics
#         answer_words = len(response.answer.split())
#         citation_count = len(re.findall(r'\[Source:', response.answer))
        
#         explanation['quality_metrics'] = {
#             'answer_length_words': answer_words,
#             'citations_found': citation_count,
#             'avg_source_score': sum(s['score'] for s in response.sources) / len(response.sources) if response.sources else 0
#         }
        
#         return explanation
    
#     def suggest_followup_questions(self, response: RAGResponse) -> List[str]:
#         """Suggest follow-up questions based on the response"""
#         suggestions = []
        
#         # Based on sources used
#         unique_docs = list(set(source['doc_id'] for source in response.sources))
#         if len(unique_docs) > 1:
#             suggestions.append(f"Can you compare the findings between {unique_docs[0]} and {unique_docs[1]}?")
        
#         # Based on sections
#         sections_used = list(set(source['section_type'] for source in response.sources))
#         if 'methodology' in sections_used:
#             suggestions.append("What methodology was used in this research?")
#         if 'results' in sections_used:
#             suggestions.append("What were the key results and findings?")
#         if 'conclusion' in sections_used:
#             suggestions.append("What are the main conclusions and implications?")
        
#         # Generic suggestions
#         suggestions.extend([
#             "Can you provide more details about the experimental design?",
#             "What are the limitations of this research?",
#             "Are there any related studies mentioned?",
#             "What future research directions are suggested?"
#         ])
        
#         return suggestions[:5]  # Return top 5 suggestions




from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import json
import logging
import time

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from sentence_transformers import util  # optional, not strictly needed
import numpy as np

# OpenAI v1 client (with v0 fallback)
try:
    from openai import OpenAI
    _OPENAI_V1 = True
except Exception:
    import openai as openai_v0  # type: ignore
    _OPENAI_V1 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """RAG response with detailed metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    context_used: str
    reasoning: str
    search_strategy: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None

class AdvancedRAGEngine:
    """Advanced RAG engine with multi-backend LLMs and strict grounding"""

    def __init__(self, vector_store, llm_provider: str = "local", model_config: Dict[str, Any] = None):
        """
        Initialize RAG engine

        Args:
            vector_store: MultiModalVectorStore instance
            llm_provider: "openai", "local", or "huggingface"
            model_config: provider-specific config (e.g., model, api_key, temperature)
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.model_config = model_config or {}
        self._initialize_llm()
        self.prompt_templates = self._build_prompts()

    # -------------------------------------------------------------------------
    # LLM init
    # -------------------------------------------------------------------------
    def _initialize_llm(self):
        if self.llm_provider == "openai":
            if _OPENAI_V1:
                self.client = OpenAI(api_key=self.model_config.get("api_key"))
            else:
                openai_v0.api_key = self.model_config.get("api_key")
                self.client = openai_v0
            self.model_name = self.model_config.get("model", "gpt-4o-mini")
            self.max_tokens = int(self.model_config.get("max_tokens", 800))
            self.temperature = float(self.model_config.get("temperature", 0.1))

        elif self.llm_provider == "local":
            model_name = self.model_config.get("model", "microsoft/DialoGPT-medium")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.pipeline = None
            except Exception as e:
                logger.warning(f"Could not load local model {model_name}: {e}")
                try:
                    self.pipeline = pipeline(
                        "text-generation",
                        model="microsoft/DialoGPT-small",
                        device=0 if torch.cuda.is_available() else -1,
                    )
                except Exception as e2:
                    logger.error(f"Could not initialize any local model: {e2}")
                    self.pipeline = None
                self.model = None
                self.tokenizer = None

        elif self.llm_provider == "huggingface":
            model_name = self.model_config.get("model", "google/flan-t5-base")
            try:
                self.pipeline = pipeline(
                    "text2text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.error(f"Could not load Hugging Face model {model_name}: {e}")
                self.pipeline = None
            self.model = None
            self.tokenizer = None

        else:
            raise ValueError(f"Unknown llm_provider: {self.llm_provider}")

    def _build_prompts(self) -> Dict[str, str]:
        base_header = (
            "You are a careful research assistant. Use ONLY the provided context.\n"
            "If the answer is not in the context, say you don't have enough information.\n"
            "Cite sources strictly in the format [Source i] where i is the bracket number shown in the context headers.\n"
        )
        return {
            "qa": base_header
            + (
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Instructions:\n"
                "1) Answer only with facts present in the context.\n"
                "2) When stating a fact, include the citation like [Source 1].\n"
                "3) If multiple sources support a point, you may include multiple citations.\n\n"
                "Answer:\n"
            ),
            "summarize": base_header
            + (
                "Context:\n{context}\n\n"
                "Instructions:\n- Summarize key findings and conclusions.\n- Group by theme.\n- Include citations.\n\n"
                "Summary:\n"
            ),
            "analyze": base_header
            + (
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Instructions:\n- Provide a reasoned analysis using only the context.\n- Note contradictions/gaps.\n- Cite sources for each claim.\n\n"
                "Analysis:\n"
            ),
            "methodology": base_header
            + (
                "Context:\n{context}\n\n"
                "Instructions:\n- Extract methods, sample sizes, tools, and limitations (only if present in context).\n- Cite sources.\n\n"
                "Methodology Summary:\n"
            ),
        }

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------
    def query(
        self,
        question: str,
        query_type: str = "qa",
        search_strategy: str = "hybrid",
        k: int = 5,
        filters: Dict[str, Any] = None,
        context_length: int = 6000,
    ) -> RAGResponse:
        t0 = time.time()
        ts = datetime.now()
        try:
            # 1) retrieve
            search_results = self.vector_store.search(query=question, k=k, search_strategy=search_strategy, filters=filters)
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find relevant information in the indexed documents to answer this question.",
                    sources=[],
                    confidence=0.0,
                    context_used="",
                    reasoning="Vector search returned no results.",
                    search_strategy=search_strategy,
                    timestamp=ts,
                )

            # 2) prepare context w/ bracketed numbered headers
            context, sources_info = self._prepare_context(search_results, context_length)

            # 3) prompt
            prompt = self._generate_prompt(question, context, query_type)

            # 4) LLM answer
            answer, tokens_used = self._generate_answer(prompt, query_type)

            # 5) post-process (strip hallucinated citations / enforce format)
            answer = self._post_process_answer(answer, len(sources_info))

            # 6) confidence
            confidence = self._calculate_confidence(search_results, answer)

            # 7) reasoning metadata
            reasoning = self._generate_reasoning(search_results, answer)

            return RAGResponse(
                answer=answer,
                sources=sources_info,
                confidence=confidence,
                context_used=context,
                reasoning=reasoning,
                search_strategy=search_strategy,
                timestamp=ts,
                tokens_used=tokens_used,
                response_time=round(time.time() - t0, 3),
            )
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your question: {e}",
                sources=[],
                confidence=0.0,
                context_used="",
                reasoning="Unhandled exception",
                search_strategy=search_strategy,
                timestamp=ts,
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _prepare_context(self, search_results, max_length: int) -> Tuple[str, List[Dict]]:
        parts: List[str] = []
        sources_info: List[Dict[str, Any]] = []
        cur = 0
        for i, r in enumerate(search_results, 1):
            c = r.chunk
            header = f"[Source {i}: {c.doc_id}, Page {c.page_num}, {c.section_type}]\n"
            body = c.text.strip()
            seg = header + body + "\n\n"
            if cur + len(seg) <= max_length:
                parts.append(seg)
                cur += len(seg)
                sources_info.append({
                    "doc_id": c.doc_id,
                    "page_num": c.page_num,
                    "section_type": c.section_type,
                    "score": float(r.score),
                    "rank": int(r.rank),
                    "search_type": r.search_type,
                })
            else:
                # Soft truncate if near limit
                if max_length - cur > 200:
                    parts.append(seg[: max_length - cur] + "…\n")
                    sources_info.append({
                        "doc_id": c.doc_id,
                        "page_num": c.page_num,
                        "section_type": c.section_type,
                        "score": float(r.score),
                        "rank": int(r.rank),
                        "search_type": r.search_type,
                    })
                break
        return "".join(parts), sources_info

    def _generate_prompt(self, question: str, context: str, query_type: str) -> str:
        template = self.prompt_templates.get(query_type, self.prompt_templates["qa"])
        return template.format(question=question, context=context)

    def _generate_answer(self, prompt: str, query_type: str) -> Tuple[str, Optional[int]]:
        tokens_used = None
        if self.llm_provider == "openai":
            try:
                if _OPENAI_V1:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful research assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    out = resp.choices[0].message.content
                    tokens_used = getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else None
                else:
                    resp = self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful research assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.model_config.get("max_tokens", 800),
                        temperature=self.model_config.get("temperature", 0.1),
                    )
                    out = resp["choices"][0]["message"]["content"]
                    tokens_used = resp.get("usage", {}).get("total_tokens")
                return out, tokens_used
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                return "I couldn't generate a response with the OpenAI backend.", None

        if self.llm_provider == "local":
            try:
                if getattr(self, "model", None) and getattr(self, "tokenizer", None):
                    inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_length=min(1500, inputs.shape[1] + 512),
                            do_sample=True,
                            temperature=0.1,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return text[len(prompt):].strip(), None
                if getattr(self, "pipeline", None):
                    res = self.pipeline(prompt, max_length=min(len(prompt) + 512, 1500), temperature=0.1, do_sample=True)
                    return res[0]["generated_text"][len(prompt):].strip(), None
                return "Local model is not available. Please configure a proper local model.", None
            except Exception as e:
                logger.error(f"Local model error: {e}")
                return "I couldn't generate a response with the local model.", None

        if self.llm_provider == "huggingface":
            try:
                if getattr(self, "pipeline", None):
                    res = self.pipeline(prompt, max_length=512, temperature=0.1)
                    # many seq2seq return key 'generated_text'
                    text = res[0].get("generated_text", "")
                    return text, None
                return "The Hugging Face pipeline is not available.", None
            except Exception as e:
                logger.error(f"Hugging Face pipeline error: {e}")
                return "I couldn't generate a response with the Hugging Face backend.", None

        return "No valid LLM provider configured.", None

    def _post_process_answer(self, answer: str, n_sources: int) -> str:
        ans = answer.strip()
        ans = re.sub(r"\n{2,}", "\n\n", ans)
        # Normalize citation tokens like [Source: ...] -> [Source i]
        # Also remove any citation referring to non-existent indices
        # Keep only [Source i] where 1 <= i <= n_sources
        valid = set(str(i) for i in range(1, n_sources + 1))
        # Replace variants like [Source: 1] or (Source 1)
        ans = re.sub(r"\((?i:source)\s*(\d+)\)", r"[Source \1]", ans)
        ans = re.sub(r"\[(?i:source)\s*:\s*(\d+)\]", r"[Source \1]", ans)
        # Drop citations to non-existent sources
        def _filter_cite(m):
            i = m.group(1)
            return f"[Source {i}]" if i in valid else ""
        ans = re.sub(r"\[Source\s+(\d+)\]", _filter_cite, ans)
        return ans

    def _calculate_confidence(self, search_results, answer: str) -> float:
        if not search_results:
            return 0.0
        # Base: avg score (already in [0,1] approx for cosine/IP)
        base = float(np.mean([r.score for r in search_results]))
        # Quality signals
        word_len = len(answer.split())
        cite_count = len(re.findall(r"\[Source\s+\d+\]", answer))
        q = 1.0
        if 40 <= word_len <= 350:
            q += 0.1
        if cite_count >= 1:
            q += min(0.2, 0.05 * cite_count)
        if any(s in answer.lower() for s in ["i don't have enough", "not enough information", "cannot answer"]):
            q -= 0.1
        conf = min(max(base * q, 0.0), 1.0)
        return conf

    def _generate_reasoning(self, search_results, answer: str) -> str:
        parts = [f"Used {len(search_results)} chunks from the vector store."]
        if search_results:
            top = search_results[0]
            parts.append(f"Top source: {top.chunk.doc_id} (p.{top.chunk.page_num}, {top.chunk.section_type}) score={top.score:.3f}.")
        uniq_docs = len(set(r.chunk.doc_id for r in search_results))
        if uniq_docs > 1:
            parts.append(f"Evidence comes from {uniq_docs} documents.")
        sec_types = sorted(list(set(r.chunk.section_type for r in search_results)))
        parts.append("Sections leveraged: " + ", ".join(sec_types) + ".")
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Batch/explain/suggest
    # -------------------------------------------------------------------------
    def batch_process(self, queries: List[Dict[str, Any]]) -> List[RAGResponse]:
        out = []
        for q in queries:
            out.append(self.query(
                question=q.get("question", ""),
                query_type=q.get("type", "qa"),
                search_strategy=q.get("search_strategy", "hybrid"),
                k=q.get("k", 5),
                filters=q.get("filters"),
                context_length=q.get("context_length", 6000),
            ))
        return out

    def explain_answer(self, response: RAGResponse) -> Dict[str, Any]:
        doc_counts: Dict[str, int] = {}
        section_counts: Dict[str, int] = {}
        for s in response.sources:
            doc_counts[s["doc_id"]] = doc_counts.get(s["doc_id"], 0) + 1
            section_counts[s["section_type"]] = section_counts.get(s["section_type"], 0) + 1
        return {
            "search_strategy": response.search_strategy,
            "sources_used": len(response.sources),
            "confidence_score": response.confidence,
            "reasoning": response.reasoning,
            "context_length": len(response.context_used),
            "response_time": response.response_time,
            "tokens_used": response.tokens_used,
            "source_breakdown": {"documents": doc_counts, "sections": section_counts},
            "quality_metrics": {
                "answer_length_words": len(response.answer.split()),
                "citations_found": len(re.findall(r"\[Source\s+\d+\]", response.answer)),
                "avg_source_score": float(np.mean([s["score"] for s in response.sources])) if response.sources else 0.0,
            },
        }

    def suggest_followup_questions(self, response: RAGResponse) -> List[str]:
        suggestions: List[str] = []
        uniq_docs = list({s["doc_id"] for s in response.sources})
        if len(uniq_docs) >= 2:
            suggestions.append(f"Compare the findings between {uniq_docs[0]} and {uniq_docs[1]}.")
        secs = list({s["section_type"] for s in response.sources})
        if "methodology" in secs:
            suggestions.append("Detail the experimental design and controls mentioned.")
        if "results" in secs:
            suggestions.append("List the key quantitative results with their significance values.")
        if "conclusion" in secs:
            suggestions.append("Summarize the main implications and limitations discussed.")
        suggestions.extend([
            "Are there any contradictions between sources on this topic?",
            "What future directions are proposed by the authors?",
        ])
        # top 5 unique
        out, seen = [], set()
        for s in suggestions:
            if s not in seen:
                out.append(s)
                seen.add(s)
            if len(out) >= 5:
                break
        return out
