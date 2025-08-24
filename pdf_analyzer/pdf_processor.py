# import fitz  # PyMuPDF
# import PyPDF2
# from PIL import Image
# from io import BytesIO
# import re
# from typing import List, Dict, Tuple, Any
# import numpy as np
# from dataclasses import dataclass
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from sentence_transformers import SentenceTransformer
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# import spacy

# # Download required NLTK data
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass

# @dataclass
# class ExtractedFigure:
#     """Represents a figure/chart extracted from PDF"""
#     image: Image.Image
#     page_num: int
#     bbox: Tuple[float, float, float, float]
#     caption: str
#     figure_type: str  # 'chart', 'graph', 'diagram', 'table', 'unknown'
#     metadata: Dict[str, Any]

# @dataclass
# class TextSection:
#     """Represents a section of text with metadata"""
#     content: str
#     section_type: str
#     page_num: int
#     confidence: float
#     word_count: int
#     key_terms: List[str]

# class AdvancedPDFProcessor:
#     """Enhanced PDF processor with better text extraction and image analysis"""
    
#     def __init__(self):
#         self.section_patterns = {
#             'title': r'^[A-Z][A-Za-z\s:]+$',
#             'abstract': r'(?i)(abstract|summary)\s*:?\s*',
#             'introduction': r'(?i)(introduction|background)\s*:?\s*',
#             'methodology': r'(?i)(method|methodology|approach|technique|implementation|design)\s*:?\s*',
#             'results': r'(?i)(result|finding|evaluation|experiment|analysis)\s*:?\s*',
#             'discussion': r'(?i)(discussion|interpretation)\s*:?\s*',
#             'conclusion': r'(?i)(conclusion|summary|future\s+work)\s*:?\s*',
#             'references': r'(?i)(reference|bibliography|citation)\s*:?\s*',
#             'acknowledgment': r'(?i)(acknowledgment|acknowledge)\s*:?\s*'
#         }
        
#         self.figure_keywords = [
#             'figure', 'fig.', 'chart', 'graph', 'plot', 'diagram', 
#             'table', 'image', 'illustration', 'schema'
#         ]
        
#         # Initialize NLP models
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             self.nlp = None
#             print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
#     def extract_comprehensive_data(self, pdf_file) -> Dict[str, Any]:
#         """Extract all data from PDF with advanced processing"""
#         pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
#         result = {
#             'text_sections': [],
#             'figures': [],
#             'tables': [],
#             'metadata': {},
#             'structure': {},
#             'statistics': {}
#         }
        
#         try:
#             # Extract metadata
#             result['metadata'] = self._extract_metadata(pdf_document)
            
#             # Process each page
#             for page_num in range(pdf_document.page_count):
#                 page = pdf_document[page_num]
                
#                 # Extract text with structure
#                 text_data = self._extract_structured_text(page, page_num)
#                 result['text_sections'].extend(text_data)
                
#                 # Extract figures and images
#                 figures = self._extract_figures_with_analysis(page, page_num)
#                 result['figures'].extend(figures)
                
#                 # Extract tables
#                 tables = self._extract_tables(page, page_num)
#                 result['tables'].extend(tables)
            
#             # Analyze document structure
#             result['structure'] = self._analyze_document_structure(result['text_sections'])
            
#             # Generate statistics
#             result['statistics'] = self._generate_statistics(result)
            
#         finally:
#             pdf_document.close()
        
#         return result
    
#     def _extract_metadata(self, pdf_document) -> Dict[str, Any]:
#         """Extract PDF metadata"""
#         metadata = pdf_document.metadata
#         return {
#             'title': metadata.get('title', ''),
#             'author': metadata.get('author', ''),
#             'subject': metadata.get('subject', ''),
#             'creator': metadata.get('creator', ''),
#             'producer': metadata.get('producer', ''),
#             'creation_date': metadata.get('creationDate', ''),
#             'modification_date': metadata.get('modDate', ''),
#             'page_count': pdf_document.page_count
#         }
    
#     def _extract_structured_text(self, page, page_num: int) -> List[TextSection]:
#         """Extract text with structure analysis"""
#         text = page.get_text()
#         blocks = page.get_text("dict")
        
#         sections = []
        
#         # Split text into paragraphs
#         paragraphs = text.split('\n\n')
        
#         for para in paragraphs:
#             if len(para.strip()) < 20:  # Skip very short paragraphs
#                 continue
            
#             section_type = self._classify_section(para)
#             key_terms = self._extract_key_terms(para)
            
#             section = TextSection(
#                 content=para.strip(),
#                 section_type=section_type,
#                 page_num=page_num + 1,
#                 confidence=self._calculate_section_confidence(para, section_type),
#                 word_count=len(para.split()),
#                 key_terms=key_terms
#             )
#             sections.append(section)
        
#         return sections
    
#     def _classify_section(self, text: str) -> str:
#         """Classify text section type"""
#         text_lower = text.lower().strip()
        
#         # Check for section headers
#         for section_type, pattern in self.section_patterns.items():
#             if re.search(pattern, text_lower[:100]):  # Check first 100 chars
#                 return section_type
        
#         # Content-based classification
#         if len(text) < 200:
#             return 'short_text'
#         elif any(keyword in text_lower for keyword in ['method', 'approach', 'technique']):
#             return 'methodology'
#         elif any(keyword in text_lower for keyword in ['result', 'finding', 'show']):
#             return 'results'
#         elif any(keyword in text_lower for keyword in ['discuss', 'interpret', 'implication']):
#             return 'discussion'
#         else:
#             return 'body_text'
    
#     def _calculate_section_confidence(self, text: str, section_type: str) -> float:
#         """Calculate confidence score for section classification"""
#         if section_type in self.section_patterns:
#             pattern = self.section_patterns[section_type]
#             if re.search(pattern, text.lower()[:100]):
#                 return 0.9
        
#         return 0.6  # Default confidence
    
#     def _extract_key_terms(self, text: str) -> List[str]:
#         """Extract key terms from text using NLP"""
#         if not self.nlp:
#             # Fallback to simple keyword extraction
#             words = word_tokenize(text.lower())
#             stop_words = set(stopwords.words('english') if 'stopwords' in dir(nltk.corpus) else [])
#             return [w for w in words if w.isalpha() and len(w) > 3 and w not in stop_words][:10]
        
#         doc = self.nlp(text)
        
#         # Extract named entities and important terms
#         entities = [ent.text for ent in doc.ents]
#         noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        
#         # Combine and deduplicate
#         key_terms = list(set(entities + noun_phrases))
#         return key_terms[:15]
    
#     def _extract_figures_with_analysis(self, page, page_num: int) -> List[ExtractedFigure]:
#         """Extract figures with enhanced analysis"""
#         figures = []
        
#         # Get images from page
#         image_list = page.get_images()
#         text = page.get_text().lower()
        
#         for img_index, img in enumerate(image_list):
#             try:
#                 xref = img[0]
#                 pix = fitz.Pixmap(page.get_document(), xref)
                
#                 if pix.n - pix.alpha < 4:  # GRAY or RGB
#                     img_data = pix.tobytes("png")
#                     img_pil = Image.open(BytesIO(img_data))
                    
#                     # Get image bounding box
#                     bbox = page.get_image_bbox(img)
                    
#                     # Find caption
#                     caption = self._find_figure_caption(text, img_index)
                    
#                     # Classify figure type
#                     figure_type = self._classify_figure_type(img_pil, caption)
                    
#                     # Extract metadata
#                     metadata = {
#                         'size': img_pil.size,
#                         'mode': img_pil.mode,
#                         'format': img_pil.format,
#                         'has_transparency': 'transparency' in img_pil.info
#                     }
                    
#                     figure = ExtractedFigure(
#                         image=img_pil,
#                         page_num=page_num + 1,
#                         bbox=bbox,
#                         caption=caption,
#                         figure_type=figure_type,
#                         metadata=metadata
#                     )
#                     figures.append(figure)
                
#                 pix = None
                
#             except Exception as e:
#                 print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
        
#         return figures
    
#     def _find_figure_caption(self, page_text: str, img_index: int) -> str:
#         """Find caption for a figure"""
#         # Look for figure references
#         patterns = [
#             rf'figure\s+{img_index + 1}[:\.]?\s*([^\.]+\.)',
#             rf'fig\.\s*{img_index + 1}[:\.]?\s*([^\.]+\.)',
#             rf'figure\s+\d+[:\.]?\s*([^\.]+\.)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, page_text, re.IGNORECASE)
#             if match:
#                 return match.group(1).strip()
        
#         return f"Figure {img_index + 1}"
    
#     def _classify_figure_type(self, image: Image.Image, caption: str) -> str:
#         """Classify the type of figure"""
#         caption_lower = caption.lower()
        
#         # Check caption for keywords
#         if any(keyword in caption_lower for keyword in ['chart', 'bar', 'pie', 'histogram']):
#             return 'chart'
#         elif any(keyword in caption_lower for keyword in ['graph', 'plot', 'curve', 'line']):
#             return 'graph'
#         elif any(keyword in caption_lower for keyword in ['table', 'matrix']):
#             return 'table'
#         elif any(keyword in caption_lower for keyword in ['diagram', 'schema', 'flow']):
#             return 'diagram'
#         elif any(keyword in caption_lower for keyword in ['photo', 'image', 'picture']):
#             return 'photograph'
#         else:
#             # Basic image analysis
#             width, height = image.size
#             aspect_ratio = width / height
            
#             if aspect_ratio > 1.5:  # Wide image, likely a chart or graph
#                 return 'chart'
#             elif 0.7 <= aspect_ratio <= 1.3:  # Square-ish, could be diagram
#                 return 'diagram'
#             else:
#                 return 'unknown'
    
#     def _extract_tables(self, page, page_num: int) -> List[Dict[str, Any]]:
#         """Extract tables from page"""
#         tables = []
        
#         try:
#             # Use PyMuPDF's table detection
#             tabs = page.find_tables()
            
#             for tab_index, tab in enumerate(tabs):
#                 table_data = tab.extract()
                
#                 table_info = {
#                     'data': table_data,
#                     'page_num': page_num + 1,
#                     'bbox': tab.bbox,
#                     'rows': len(table_data),
#                     'cols': len(table_data[0]) if table_data else 0,
#                     'table_index': tab_index
#                 }
#                 tables.append(table_info)
        
#         except Exception as e:
#             print(f"Error extracting tables from page {page_num + 1}: {e}")
        
#         return tables
    
#     def _analyze_document_structure(self, sections: List[TextSection]) -> Dict[str, Any]:
#         """Analyze the overall document structure"""
#         structure = {
#             'sections_found': [],
#             'section_order': [],
#             'has_abstract': False,
#             'has_conclusion': False,
#             'has_references': False,
#             'total_sections': len(sections)
#         }
        
#         section_types = [section.section_type for section in sections]
#         structure['sections_found'] = list(set(section_types))
#         structure['section_order'] = section_types
        
#         structure['has_abstract'] = 'abstract' in section_types
#         structure['has_conclusion'] = 'conclusion' in section_types
#         structure['has_references'] = 'references' in section_types
        
#         return structure
    
#     def _generate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         """Generate document statistics"""
#         sections = data['text_sections']
#         figures = data['figures']
#         tables = data['tables']
        
#         total_words = sum(section.word_count for section in sections)
#         avg_words_per_section = total_words / len(sections) if sections else 0
        
#         stats = {
#             'total_words': total_words,
#             'total_sections': len(sections),
#             'avg_words_per_section': avg_words_per_section,
#             'total_figures': len(figures),
#             'total_tables': len(tables),
#             'pages': data['metadata']['page_count'],
#             'words_per_page': total_words / data['metadata']['page_count'] if data['metadata']['page_count'] > 0 else 0,
#             'figure_types': {fig.figure_type: sum(1 for f in figures if f.figure_type == fig.figure_type) for fig in figures},
#             'section_distribution': {sec_type: sum(1 for s in sections if s.section_type == sec_type) for sec_type in set(s.section_type for s in sections)}
#         }
        
#         return stats

#     def create_document_summary(self, data: Dict[str, Any]) -> Dict[str, str]:
#         """Create a structured summary of the document"""
#         sections = data['text_sections']
#         structure = data['structure']
        
#         summary = {
#             'abstract': '',
#             'introduction': '',
#             'methodology': '',
#             'results': '',
#             'discussion': '',
#             'conclusion': ''
#         }
        
#         # Group sections by type
#         section_groups = {}
#         for section in sections:
#             if section.section_type not in section_groups:
#                 section_groups[section.section_type] = []
#             section_groups[section.section_type].append(section)
        
#         # Create summaries for each section type
#         for section_type in summary.keys():
#             if section_type in section_groups:
#                 # Combine all text from this section type
#                 combined_text = ' '.join([s.content for s in section_groups[section_type]])
                
#                 # Truncate to reasonable length for summary
#                 if len(combined_text) > 1500:
#                     sentences = sent_tokenize(combined_text)
#                     summary_text = ''
#                     for sentence in sentences:
#                         if len(summary_text + sentence) <= 1500:
#                             summary_text += sentence + ' '
#                         else:
#                             break
#                     summary[section_type] = summary_text.strip()
#                 else:
#                     summary[section_type] = combined_text
        
#         return summary

#     def extract_key_insights(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Extract key insights from the document"""
#         sections = data['text_sections']
#         insights = []
        
#         # Look for methodology insights
#         method_sections = [s for s in sections if s.section_type in ['methodology', 'method']]
#         if method_sections:
#             method_text = ' '.join([s.content for s in method_sections])
#             insights.append({
#                 'type': 'methodology',
#                 'content': self._extract_methodology_insights(method_text),
#                 'confidence': 0.8
#             })
        
#         # Look for results insights
#         result_sections = [s for s in sections if s.section_type in ['results', 'findings']]
#         if result_sections:
#             result_text = ' '.join([s.content for s in result_sections])
#             insights.append({
#                 'type': 'results',
#                 'content': self._extract_result_insights(result_text),
#                 'confidence': 0.8
#             })
        
#         # Look for key terms across all sections
#         all_key_terms = []
#         for section in sections:
#             all_key_terms.extend(section.key_terms)
        
#         # Count frequency and get top terms
#         from collections import Counter
#         term_counts = Counter(all_key_terms)
#         top_terms = term_counts.most_common(10)
        
#         insights.append({
#             'type': 'key_terms',
#             'content': [{'term': term, 'frequency': count} for term, count in top_terms],
#             'confidence': 0.9
#         })
        
#         return insights
    
#     def _extract_methodology_insights(self, text: str) -> Dict[str, Any]:
#         """Extract insights from methodology section"""
#         method_keywords = {
#             'experimental': ['experiment', 'test', 'trial', 'controlled'],
#             'survey': ['survey', 'questionnaire', 'interview', 'participant'],
#             'analysis': ['analyze', 'analysis', 'statistical', 'regression'],
#             'simulation': ['simulate', 'model', 'simulation', 'computational'],
#             'review': ['review', 'systematic', 'meta-analysis', 'literature']
#         }
        
#         text_lower = text.lower()
#         detected_methods = []
        
#         for method_type, keywords in method_keywords.items():
#             if any(keyword in text_lower for keyword in keywords):
#                 detected_methods.append(method_type)
        
#         return {
#             'detected_methods': detected_methods,
#             'sample_size': self._extract_sample_size(text),
#             'tools_mentioned': self._extract_tools(text)
#         }
    
#     def _extract_result_insights(self, text: str) -> Dict[str, Any]:
#         """Extract insights from results section"""
#         # Look for statistical significance
#         significance_patterns = [
#             r'p\s*[<>=]\s*0\.\d+',
#             r'significant',
#             r'correlation',
#             r'r\s*=\s*0\.\d+',
#             r'\d+%'
#         ]
        
#         statistical_findings = []
#         for pattern in significance_patterns:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             statistical_findings.extend(matches)
        
#         return {
#             'statistical_findings': statistical_findings,
#             'has_significance_testing': any('p' in finding.lower() for finding in statistical_findings),
#             'has_correlation': any('correlation' in finding.lower() for finding in statistical_findings)
#         }
    
#     def _extract_sample_size(self, text: str) -> str:
#         """Extract sample size from methodology text"""
#         patterns = [
#             r'n\s*=\s*(\d+)',
#             r'sample\s+size\s+of\s+(\d+)',
#             r'(\d+)\s+participants',
#             r'(\d+)\s+subjects'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 return match.group(1)
        
#         return "Not specified"
    
#     def _extract_tools(self, text: str) -> List[str]:
#         """Extract tools and software mentioned in methodology"""
#         tool_patterns = [
#             r'SPSS', r'R\s+software', r'Python', r'MATLAB', r'SAS',
#             r'Excel', r'Stata', r'NVIVO', r'Atlas\.ti', r'AMOS'
#         ]
        
#         tools = []
#         for pattern in tool_patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 tools.append(pattern.replace(r'\.', '.').replace(r'\s+', ' '))
        
#         return tools





import fitz  # PyMuPDF
import PyPDF2
from PIL import Image
from io import BytesIO
import re
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Optional OCR if PDFs are scanned; safe-import
try:
    import pytesseract  # noqa: F401
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download minimal NLTK data (no error if offline)
for pkg in ("punkt", "stopwords"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass
class ExtractedFigure:
    """Represents a figure/chart extracted from PDF"""
    image: Image.Image
    page_num: int
    bbox: Tuple[float, float, float, float]
    caption: str
    figure_type: str  # 'chart', 'graph', 'diagram', 'table', 'photograph', 'unknown'
    metadata: Dict[str, Any]

@dataclass
class TextSection:
    """Represents a section of text with metadata"""
    content: str
    section_type: str
    page_num: int
    confidence: float
    word_count: int
    key_terms: List[str]
    title: Optional[str] = None  # nearest heading
    bbox: Optional[Tuple[float, float, float, float]] = None

# -----------------------------------------------------------------------------
# Processor
# -----------------------------------------------------------------------------
class AdvancedPDFProcessor:
    """Enhanced PDF processor with robust text/figure/table extraction and light NLP"""

    def __init__(self):
        self.section_patterns = {
            "title": r"^[A-Z][A-Za-z0-9\s:,\-\(\)]+$",
            "abstract": r"(?i)^(abstract|summary)\s*:?\s*$",
            "introduction": r"(?i)^(introduction|background)\s*:?\s*$",
            "methodology": r"(?i)^(method|methods|methodology|approach|technique|implementation|design)\s*:?\s*$",
            "results": r"(?i)^(results?|findings?|evaluation|experiments?|analysis)\s*:?\s*$",
            "discussion": r"(?i)^(discussion|interpretation)\s*:?\s*$",
            "conclusion": r"(?i)^(conclusion|conclusions|summary|future\s+work)\s*:?\s*$",
            "references": r"(?i)^(references?|bibliography|citations?)\s*:?\s*$",
            "acknowledgment": r"(?i)^(acknowledg(e)?ments?)\s*:?\s*$",
        }

        self.figure_keywords = [
            "figure", "fig.", "chart", "graph", "plot", "diagram", "table", "image", "illustration", "schema"
        ]

        # spaCy (optional)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")

        # Stopwords fallback
        try:
            self._stopwords = set(stopwords.words("english"))
        except Exception:
            self._stopwords = set()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def extract_comprehensive_data(self, pdf_file) -> Dict[str, Any]:
        """Extract structured text, figures, tables, and compute stats."""
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        result: Dict[str, Any] = {
            "text_sections": [],
            "figures": [],
            "tables": [],
            "metadata": {},
            "structure": {},
            "statistics": {},
        }

        try:
            result["metadata"] = self._extract_metadata(pdf_document)

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]

                # Text (structured)
                text_sections = self._extract_structured_text(page, page_num)
                result["text_sections"].extend(text_sections)

                # Figures/images (+ captions)
                figures = self._extract_figures_with_analysis(pdf_document, page, page_num)
                result["figures"].extend(figures)

                # Tables
                tables = self._extract_tables(page, page_num)
                result["tables"].extend(tables)

            # Document-level analysis
            result["structure"] = self._analyze_document_structure(result["text_sections"])
            result["statistics"] = self._generate_statistics(result)
        finally:
            pdf_document.close()

        return result

    def create_document_summary(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create a structured summary of the document by section types."""
        sections = data.get("text_sections", [])
        summary = {k: "" for k in ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]}

        by_type: Dict[str, List[TextSection]] = {}
        for sec in sections:
            by_type.setdefault(sec.section_type, []).append(sec)

        # Concatenate and truncate to ~1500 chars per section
        for key in summary.keys():
            if key in by_type:
                combined = " ".join(s.content for s in by_type[key])
                if len(combined) > 1500:
                    sentences = sent_tokenize(combined)
                    out = []
                    cur = 0
                    for sent in sentences:
                        if cur + len(sent) + 1 > 1500:
                            break
                        out.append(sent)
                        cur += len(sent) + 1
                    summary[key] = " ".join(out)
                else:
                    summary[key] = combined

        return summary

    def extract_key_insights(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract basic insights (methods/results/terms) with regex + NLP fallback."""
        sections: List[TextSection] = data.get("text_sections", [])
        insights = []

        # Methodology
        methods_text = " ".join(s.content for s in sections if s.section_type in ["methodology", "method"])
        if methods_text:
            insights.append({
                "type": "methodology",
                "content": self._extract_methodology_insights(methods_text),
                "confidence": 0.8
            })

        # Results
        results_text = " ".join(s.content for s in sections if s.section_type in ["results", "findings"])
        if results_text:
            insights.append({
                "type": "results",
                "content": self._extract_result_insights(results_text),
                "confidence": 0.8
            })

        # Key terms across all sections
        all_terms: List[str] = []
        for sec in sections:
            all_terms.extend(sec.key_terms)
        from collections import Counter
        top = Counter(all_terms).most_common(10)
        insights.append({"type": "key_terms", "content": [{"term": t, "frequency": c} for t, c in top], "confidence": 0.9})
        return insights

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    def _extract_metadata(self, pdf_document) -> Dict[str, Any]:
        meta = pdf_document.metadata or {}
        return {
            "title": meta.get("title", "") or meta.get("Title", ""),
            "author": meta.get("author", "") or meta.get("Author", ""),
            "subject": meta.get("subject", "") or meta.get("Subject", ""),
            "creator": meta.get("creator", "") or meta.get("Creator", ""),
            "producer": meta.get("producer", "") or meta.get("Producer", ""),
            "creation_date": meta.get("creationDate", "") or meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", "") or meta.get("modDate", ""),
            "page_count": pdf_document.page_count,
        }

    # -------------------------------------------------------------------------
    # Text extraction & sectioning
    # -------------------------------------------------------------------------
    def _extract_structured_text(self, page: fitz.Page, page_num: int) -> List[TextSection]:
        """
        Use dict layout to detect headings via font size and group nearby text.
        Fallback to plain text when needed.
        """
        sections: List[TextSection] = []

        try:
            layout = page.get_text("dict")
        except Exception:
            layout = None

        # Collect candidate headings by font size
        headings: List[Tuple[str, Tuple[float, float, float, float]]] = []
        if layout and "blocks" in layout:
            for block in layout["blocks"]:
                for line in block.get("lines", []):
                    # Compute avg font size of spans in this line
                    sizes = [s.get("size", 0) for s in line.get("spans", [])]
                    text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not text:
                        continue
                    avg_size = sum(sizes) / len(sizes) if sizes else 0
                    bbox = self._line_bbox(line)
                    # Heuristic: large text and short line -> heading candidate
                    if avg_size >= 11 and len(text) <= 120:
                        if self._looks_like_heading(text):
                            headings.append((text, bbox))

        # Extract paragraphs using "blocks" to preserve structure
        full_text = page.get_text("text") or ""
        paras = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) >= 20]

        for para in paras:
            stype = self._classify_section(para)
            key_terms = self._extract_key_terms(para)
            confidence = self._calculate_section_confidence(para, stype)
            # nearest heading by y-distance
            nearest = self._nearest_heading(page, para, headings)

            sections.append(
                TextSection(
                    content=para,
                    section_type=stype,
                    page_num=page_num + 1,
                    confidence=confidence,
                    word_count=len(para.split()),
                    key_terms=key_terms,
                    title=nearest[0] if nearest else None,
                    bbox=None,
                )
            )

        # If page has almost no text and OCR is available, OCR page image
        if not paras and OCR_AVAILABLE:
            try:
                pix = page.get_pixmap(dpi=200)
                pil = Image.open(BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(pil)
                ocr_text = re.sub(r"\n{2,}", "\n\n", ocr_text).strip()
                if len(ocr_text) > 30:
                    stype = self._classify_section(ocr_text)
                    sections.append(
                        TextSection(
                            content=ocr_text,
                            section_type=stype,
                            page_num=page_num + 1,
                            confidence=0.4,
                            word_count=len(ocr_text.split()),
                            key_terms=self._extract_key_terms(ocr_text),
                        )
                    )
            except Exception as e:
                logger.debug(f"OCR failed on page {page_num+1}: {e}")

        return sections

    def _looks_like_heading(self, text: str) -> bool:
        t = text.strip()
        if len(t) > 140:
            return False
        # match any section regex or ALL CAPS short titles
        is_pattern = any(re.search(pat, t, flags=re.IGNORECASE) for pat in self.section_patterns.values())
        is_caps = (t == t.upper() and len(t.split()) <= 12)
        return is_pattern or is_caps

    def _nearest_heading(self, page: fitz.Page, para_text: str, headings) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
        """Very light heuristic: prefer last heading seen earlier on the page."""
        # In absence of exact text bbox, return last heading in list
        if headings:
            return headings[-1]
        return None

    def _line_bbox(self, line: Dict[str, Any]) -> Tuple[float, float, float, float]:
        x0 = min(s["bbox"][0] for s in line.get("spans", []) if "bbox" in s)
        y0 = min(s["bbox"][1] for s in line.get("spans", []) if "bbox" in s)
        x1 = max(s["bbox"][2] for s in line.get("spans", []) if "bbox" in s)
        y1 = max(s["bbox"][3] for s in line.get("spans", []) if "bbox" in s)
        return (x0, y0, x1, y1)

    def _classify_section(self, text: str) -> str:
        low = text.lower().strip()
        # headers in first 120 chars
        head = low[:120]
        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, head, flags=re.IGNORECASE):
                return section_type

        # lightweight content heuristics
        if len(text) < 200:
            return "short_text"
        if any(k in low for k in ["method", "approach", "technique", "implementation"]):
            return "methodology"
        if any(k in low for k in ["result", "finding", "show", "we observe", "we find"]):
            return "results"
        if any(k in low for k in ["discuss", "interpret", "implication", "we argue"]):
            return "discussion"
        if any(k in low for k in ["conclusion", "future work", "limitations"]):
            return "conclusion"
        return "body_text"

    def _calculate_section_confidence(self, text: str, section_type: str) -> float:
        if section_type in self.section_patterns:
            pat = self.section_patterns[section_type]
            if re.search(pat, text[:120], flags=re.IGNORECASE):
                return 0.9
        return 0.6

    def _extract_key_terms(self, text: str) -> List[str]:
        if self.nlp:
            try:
                doc = self.nlp(text)
                ents = [ent.text for ent in doc.ents]
                noun_chunks = [nc.text for nc in doc.noun_chunks if 1 <= len(nc.text.split()) <= 3]
                key = list(dict.fromkeys([*ents, *noun_chunks]))  # preserve order & dedupe
                return key[:15]
            except Exception:
                pass
        # Fallback
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and len(w) > 3 and w not in self._stopwords]
        # light dedupe while preserving order
        seen = set()
        out = []
        for w in words:
            if w not in seen:
                out.append(w)
                seen.add(w)
            if len(out) >= 10:
                break
        return out

    # -------------------------------------------------------------------------
    # Figures & captions
    # -------------------------------------------------------------------------
    def _extract_figures_with_analysis(self, doc: fitz.Document, page: fitz.Page, page_num: int) -> List[ExtractedFigure]:
        figures: List[ExtractedFigure] = []
        # PyMuPDF: page.get_images(full=True) then locate rects via get_image_rects(xref)
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]  # most images have one rect
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha < 4:  # GRAY/RGB
                    img_bytes = pix.tobytes("png")
                    pil = Image.open(BytesIO(img_bytes))
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    pil = Image.open(BytesIO(img_bytes))

                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                caption = self._find_caption_near(page, bbox) or "Figure"
                ftype = self._classify_figure_type(pil, caption)
                meta = {
                    "size": pil.size,
                    "mode": pil.mode,
                    "has_transparency": bool(pil.info.get("transparency")) or (pil.mode in ("RGBA", "LA")),
                    "xref": xref,
                }
                figures.append(
                    ExtractedFigure(
                        image=pil, page_num=page_num + 1, bbox=bbox, caption=caption, figure_type=ftype, metadata=meta
                    )
                )
            except Exception as e:
                logger.debug(f"Image extraction error on page {page_num+1}: {e}")
        return figures

    def _find_caption_near(self, page: fitz.Page, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """
        Look for 'Figure/ Fig.' lines within a small rectangle under/above the image.
        """
        x0, y0, x1, y1 = bbox
        page_h = page.rect.height
        # search region below, then above
        regions = [
            fitz.Rect(x0, min(y1 + 2, page_h - 1), x1, min(y1 + 80, page_h)),
            fitz.Rect(x0, max(0, y0 - 80), x1, max(0, y0 - 2)),
        ]
        for rect in regions:
            text = page.get_textbox(rect) or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines[:3]:
                if re.search(r"(?i)\b(fig(?:\.|ure)?)\s*\d+[:.\)]?\s*", line):
                    # return first sentence-ish
                    m = re.match(r"(?i)\b(fig(?:\.|ure)?)\s*\d+[:.\)]?\s*(.*)", line)
                    desc = m.group(2).strip() if m else line
                    if not desc:
                        desc = line
                    if not desc.endswith("."):
                        desc += "."
                    return desc
        return None

    def _classify_figure_type(self, image: Image.Image, caption: str) -> str:
        c = caption.lower()
        if any(k in c for k in ["chart", "bar", "pie", "histogram", "scatter"]):
            return "chart"
        if any(k in c for k in ["graph", "plot", "curve", "line"]):
            return "graph"
        if any(k in c for k in ["table", "matrix"]):
            return "table"
        if any(k in c for k in ["diagram", "schema", "flow", "architecture"]):
            return "diagram"
        if any(k in c for k in ["photo", "image", "picture", "microscopy", "microscope"]):
            return "photograph"

        # image heuristic
        w, h = image.size
        ar = (w / max(h, 1))
        if ar > 1.6:
            return "chart"
        if 0.75 <= ar <= 1.33:
            return "diagram"
        return "unknown"

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        try:
            tabs = page.find_tables()
            for idx, tab in enumerate(tabs.tables or []):
                data = tab.extract()
                tables.append({
                    "data": data,
                    "page_num": page_num + 1,
                    "bbox": tuple(tab.bbox),
                    "rows": len(data),
                    "cols": len(data[0]) if data else 0,
                    "table_index": idx,
                })
        except Exception as e:
            logger.debug(f"Table extraction error p{page_num+1}: {e}")
        return tables

    # -------------------------------------------------------------------------
    # Structure + stats
    # -------------------------------------------------------------------------
    def _analyze_document_structure(self, sections: List[TextSection]) -> Dict[str, Any]:
        types = [s.section_type for s in sections]
        return {
            "sections_found": sorted(list(set(types))),
            "section_order": types,
            "has_abstract": "abstract" in types,
            "has_conclusion": "conclusion" in types,
            "has_references": "references" in types,
            "total_sections": len(sections),
        }

    def _generate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sections: List[TextSection] = data.get("text_sections", [])
        figures: List[ExtractedFigure] = data.get("figures", [])
        tables: List[Dict[str, Any]] = data.get("tables", [])

        total_words = sum(s.word_count for s in sections)
        page_count = max(1, data.get("metadata", {}).get("page_count", 1))

        fig_types: Dict[str, int] = {}
        for f in figures:
            fig_types[f.figure_type] = fig_types.get(f.figure_type, 0) + 1

        section_distribution: Dict[str, int] = {}
        for s in sections:
            section_distribution[s.section_type] = section_distribution.get(s.section_type, 0) + 1

        return {
            "total_words": total_words,
            "total_sections": len(sections),
            "avg_words_per_section": (total_words / len(sections)) if sections else 0.0,
            "total_figures": len(figures),
            "total_tables": len(tables),
            "pages": page_count,
            "words_per_page": total_words / page_count if page_count else 0.0,
            "figure_types": fig_types,
            "section_distribution": section_distribution,
        }

    # -------------------------------------------------------------------------
    # Lightweight NLP helpers
    # -------------------------------------------------------------------------
    def _extract_methodology_insights(self, text: str) -> Dict[str, Any]:
        method_keywords = {
            "experimental": ["experiment", "test", "trial", "controlled", "randomized"],
            "survey": ["survey", "questionnaire", "interview", "participant"],
            "analysis": ["analyze", "analysis", "statistical", "regression", "anova"],
            "simulation": ["simulate", "simulation", "model", "computational"],
            "review": ["review", "systematic", "meta-analysis", "literature"],
        }
        low = text.lower()
        detected = [m for m, kws in method_keywords.items() if any(k in low for k in kws)]
        return {"detected_methods": sorted(list(set(detected))), "sample_size": self._extract_sample_size(text), "tools_mentioned": self._extract_tools(text)}

    def _extract_result_insights(self, text: str) -> Dict[str, Any]:
        pats = [r"p\s*[<>=]\s*0\.\d+", r"\b(significant|significance)\b", r"\bcorrelation\b", r"r\s*=\s*[-+]?\d\.\d+", r"\b\d{1,3}%\b"]
        findings = []
        for p in pats:
            findings.extend(re.findall(p, text, flags=re.IGNORECASE))
        return {
            "statistical_findings": findings,
            "has_significance_testing": any("p" in f.lower() for f in findings),
            "has_correlation": any("correlation" in f.lower() for f in findings),
        }

    def _extract_sample_size(self, text: str) -> str:
        pats = [r"n\s*=\s*(\d+)", r"sample\s+size\s+of\s+(\d+)", r"(\d+)\s+participants", r"(\d+)\s+subjects"]
        for p in pats:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return "Not specified"

    def _extract_tools(self, text: str) -> List[str]:
        tools = ["SPSS", "R", "Python", "MATLAB", "SAS", "Excel", "Stata", "NVivo", "Atlas.ti", "AMOS"]
        found = []
        for t in tools:
            if re.search(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE):
                found.append(t)
        # dedupe preserve order
        out, seen = [], set()
        for t in found:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out
