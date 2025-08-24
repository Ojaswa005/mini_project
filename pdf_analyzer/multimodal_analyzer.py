import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # (kept if you use it elsewhere)
import matplotlib.patches as patches  # (kept if you use it elsewhere)
from typing import List, Dict, Any, Tuple, Optional
import pytesseract
import re
from dataclasses import dataclass, field

# Optional / not strictly required imports retained for your workflow
import base64
from io import BytesIO
import json

# Try to import EasyOCR defensively so the module still loads if it's missing
try:
    import easyocr  # type: ignore
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None  # type: ignore
    _EASYOCR_AVAILABLE = False

# For chart detection and analysis (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    image_type: str  # 'chart', 'graph', 'diagram', 'table', 'photograph', 'equation'
    confidence: float
    description: str
    extracted_text: str
    chart_data: Optional[Dict[str, Any]] = None
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartElement:
    """Detected chart element"""
    element_type: str  # 'axis', 'legend', 'title', 'data_point', 'line', 'bar'
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    confidence: float


class MultimodalAnalyzer:
    """Advanced analyzer for images, charts, graphs, and diagrams in research papers"""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the multimodal analyzer

        Args:
            use_gpu: Whether to use GPU for OCR processing
        """
        self.use_gpu = use_gpu

        # Initialize OCR readers (optional)
        self.ocr_available = False
        self.easyocr_reader = None
        if _EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)  # type: ignore
                self.ocr_available = True
            except Exception as e:
                print(f"Warning: EasyOCR initialization failed: {e}")
                self.ocr_available = False

        # Chart type classification patterns
        self.chart_patterns = {
            'bar_chart': ['bar', 'column', 'histogram'],
            'line_chart': ['line', 'trend', 'time series', 'plot'],
            'pie_chart': ['pie', 'donut', 'circular'],
            'scatter_plot': ['scatter', 'correlation', 'distribution'],
            'heatmap': ['heatmap', 'correlation matrix', 'heat map'],
            'box_plot': ['box plot', 'boxplot', 'whisker'],
            'flow_chart': ['flowchart', 'flow chart', 'process', 'workflow'],
            'network_diagram': ['network', 'graph', 'node', 'edge'],
            'timeline': ['timeline', 'chronological', 'sequence']
        }

        # Color palettes for chart analysis
        self.common_colors = {
            'blue': [0, 0, 255], 'red': [255, 0, 0], 'green': [0, 255, 0],
            'yellow': [255, 255, 0], 'purple': [128, 0, 128], 'orange': [255, 165, 0],
            'cyan': [0, 255, 255], 'magenta': [255, 0, 255], 'black': [0, 0, 0],
            'white': [255, 255, 255], 'gray': [128, 128, 128]
        }

    def analyze_image(self, image: Image.Image, context: str = "") -> ImageAnalysisResult:
        """
        Comprehensive analysis of an image from a research paper

        Args:
            image: PIL Image object
            context: Optional context about the image (e.g., caption, surrounding text)
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Step 1: Basic image analysis
        metadata = self._extract_image_metadata(image)

        # Step 2: Classify image type
        image_type, type_confidence = self._classify_image_type(image, cv_image, context)

        # Step 3: Extract text using OCR
        extracted_text = self._extract_text_ocr(image)

        # Step 4: Detect elements based on image type
        detected_elements = self._detect_elements(cv_image, image_type)

        # Step 5: Extract chart data if applicable
        chart_data = None
        if image_type in ['bar_chart', 'line_chart', 'scatter_plot', 'pie_chart']:
            chart_data = self._extract_chart_data(cv_image, image_type, detected_elements)

        # Step 6: Generate description
        description = self._generate_description(image_type, detected_elements, extracted_text, context)

        return ImageAnalysisResult(
            image_type=image_type,
            confidence=type_confidence,
            description=description,
            extracted_text=extracted_text,
            chart_data=chart_data,
            detected_elements=detected_elements,
            metadata=metadata
        )

    def _extract_image_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic metadata from image"""
        height = image.height if image.height else 1
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'aspect_ratio': image.width / height,
            'total_pixels': image.width * image.height
        }

    def _classify_image_type(self, image: Image.Image, cv_image: np.ndarray, context: str) -> Tuple[str, float]:
        """Classify the type of image using multiple approaches"""
        scores: Dict[str, float] = {}

        # Method 1: Context-based classification
        context_lower = context.lower() if context else ""
        for chart_type, keywords in self.chart_patterns.items():
            context_score = sum(1 for keyword in keywords if keyword in context_lower)
            if context_score > 0:
                scores[chart_type] = context_score * 0.3

        # Method 2: Visual feature analysis
        visual_scores = self._analyze_visual_features(cv_image)
        for chart_type, score in visual_scores.items():
            scores[chart_type] = scores.get(chart_type, 0.0) + score * 0.4

        # Method 3: Color and structure analysis
        structure_scores = self._analyze_structure(cv_image)
        for chart_type, score in structure_scores.items():
            scores[chart_type] = scores.get(chart_type, 0.0) + score * 0.3

        # Determine best match
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = float(min(scores[best_type], 1.0))
            return best_type, confidence
        else:
            return 'diagram', 0.5  # Default classification

    def _analyze_visual_features(self, cv_image: np.ndarray) -> Dict[str, float]:
        """Analyze visual features to classify image type"""
        scores: Dict[str, float] = {}

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0

        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Rectangle / circle detection (very rough)
        rect_count = 0
        circle_count = 0

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Rectangle-ish
                rect_count += 1
            elif len(approx) > 8:  # Circular-ish
                circle_count += 1

        # Scoring based on features (heuristic)
        if rect_count > 3 and line_count > 5:
            scores['bar_chart'] = 0.8

        if line_count > 10 and rect_count < 3:
            scores['line_chart'] = 0.7

        if circle_count > 0 and rect_count < 2:
            scores['pie_chart'] = 0.6

        if edge_density > 0.1 and line_count > 15:
            scores['network_diagram'] = 0.5

        return scores

    def _analyze_structure(self, cv_image: np.ndarray) -> Dict[str, float]:
        """Analyze image structure for classification"""
        scores: Dict[str, float] = {}

        # Color (kept for future use)
        _ = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Horizontal and vertical line detection via morphology
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        h_line_density = float(np.sum(horizontal_lines > 0)) / float(horizontal_lines.size)
        v_line_density = float(np.sum(vertical_lines > 0)) / float(vertical_lines.size)

        # Grid suggests chart/graph
        if h_line_density > 0.01 and v_line_density > 0.01:
            scores['line_chart'] = scores.get('line_chart', 0.0) + 0.6
            scores['bar_chart'] = scores.get('bar_chart', 0.0) + 0.5
            scores['scatter_plot'] = scores.get('scatter_plot', 0.0) + 0.4

        # Text region detection (suggests labels, titles)
        text_regions = self._detect_text_regions(gray)
        if len(text_regions) > 2:
            scores['bar_chart'] = scores.get('bar_chart', 0.0) + 0.2
            scores['line_chart'] = scores.get('line_chart', 0.0) + 0.2

        return scores

    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in image using MSER"""
        text_regions: List[Tuple[int, int, int, int]] = []
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)

            for region in regions:
                if len(region) > 50:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(region)
                    # Filter by aspect ratio (text tends to be wider than tall)
                    if w > h and w > 20 and h > 5:
                        text_regions.append((x, y, w, h))
        except Exception as e:
            # If MSER unavailable for any reason, just return empty
            print(f"MSER text detection warning: {e}")
        return text_regions

    def _extract_text_ocr(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        extracted_texts: List[str] = []

        # Method 1: Tesseract OCR
        try:
            pytesseract_text = pytesseract.image_to_string(image, config='--psm 6')
            if pytesseract_text and pytesseract_text.strip():
                extracted_texts.append(pytesseract_text.strip())
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")

        # Method 2: EasyOCR (if available)
        if self.ocr_available and self.easyocr_reader is not None:
            try:
                image_array = np.array(image)
                results = self.easyocr_reader.readtext(image_array)
                easyocr_text = ' '.join([res[1] for res in results if len(res) > 2 and res[2] > 0.5])
                if easyocr_text.strip():
                    extracted_texts.append(easyocr_text.strip())
            except Exception as e:
                print(f"EasyOCR failed: {e}")

        # Combine and clean results
        combined_text = ' '.join(extracted_texts)

        # Clean up text
        combined_text = re.sub(r'\s+', ' ', combined_text)  # Multiple spaces
        combined_text = re.sub(r'[^\w\s.,;:()%-]', '', combined_text)  # Remove special chars

        return combined_text

    def _detect_elements(self, cv_image: np.ndarray, image_type: str) -> List[Dict[str, Any]]:
        """Detect specific elements based on image type"""
        elements: List[Dict[str, Any]] = []

        if image_type in ['bar_chart', 'line_chart', 'scatter_plot']:
            elements.extend(self._detect_chart_elements(cv_image))
        elif image_type in ['flow_chart', 'network_diagram']:
            elements.extend(self._detect_diagram_elements(cv_image))
        elif image_type == 'table':
            elements.extend(self._detect_table_elements(cv_image))

        return elements

    def _detect_chart_elements(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect chart-specific elements like axes, legends, data points"""
        elements: List[Dict[str, Any]] = []

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect axes (long straight lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=5)

        if lines is not None:
            # Group lines into horizontal and vertical
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                if angle < 10 or angle > 170:  # Horizontal
                    horizontal_lines.append(line[0])
                elif 80 < angle < 100:  # Vertical
                    vertical_lines.append(line[0])

            if horizontal_lines:
                elements.append({
                    'type': 'x_axis',
                    'lines': horizontal_lines,
                    'count': len(horizontal_lines),
                    'confidence': 0.8
                })

            if vertical_lines:
                elements.append({
                    'type': 'y_axis',
                    'lines': vertical_lines,
                    'count': len(vertical_lines),
                    'confidence': 0.8
                })

        # Detect data points (for scatter plots)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data_points = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Small circular regions
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius < 10:
                    data_points.append({'x': int(x), 'y': int(y), 'radius': float(radius)})

        if data_points:
            elements.append({
                'type': 'data_points',
                'points': data_points,
                'count': len(data_points),
                'confidence': 0.6
            })

        return elements

    def _detect_diagram_elements(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect flowchart and diagram elements"""
        elements: List[Dict[str, Any]] = []

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        diamonds = []
        circles = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip very small contours
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect_ratio = w / float(h)

            if len(approx) == 4:
                if 0.7 < aspect_ratio < 1.3:
                    diamonds.append({'bbox': (x, y, w, h), 'area': float(area)})
                else:
                    boxes.append({'bbox': (x, y, w, h), 'area': float(area)})
            elif len(approx) > 8:
                circles.append({'bbox': (x, y, w, h), 'area': float(area)})

        if boxes:
            elements.append({'type': 'process_boxes', 'elements': boxes, 'count': len(boxes)})
        if diamonds:
            elements.append({'type': 'decision_diamonds', 'elements': diamonds, 'count': len(diamonds)})
        if circles:
            elements.append({'type': 'terminal_circles', 'elements': circles, 'count': len(circles)})

        return elements

    def _detect_table_elements(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect table structure"""
        elements: List[Dict[str, Any]] = []

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Threshold to emphasize lines
        _, h_bin = cv2.threshold(horizontal_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, v_bin = cv2.threshold(vertical_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find line positions (very rough heuristic)
        h_lines = []
        v_lines = []

        # Detect horizontal line positions
        row_sums = np.sum(h_bin > 0, axis=1)
        for i, s in enumerate(row_sums):
            if s > h_bin.shape[1] * 0.5:
                h_lines.append(i)

        # Detect vertical line positions
        col_sums = np.sum(v_bin > 0, axis=0)
        for j, s in enumerate(col_sums):
            if s > v_bin.shape[0] * 0.5:
                v_lines.append(j)

        if len(h_lines) >= 2 and len(v_lines) >= 2:
            elements.append({
                'type': 'table_grid',
                'rows': max(0, len(h_lines) - 1),
                'cols': max(0, len(v_lines) - 1),
                'h_lines': h_lines,
                'v_lines': v_lines,
                'confidence': 0.9
            })

        return elements

    def _extract_chart_data(self, cv_image: np.ndarray, chart_type: str, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract quantitative data from charts"""
        chart_data: Dict[str, Any] = {'type': chart_type, 'data_points': [], 'axes_info': {}}

        if chart_type == 'bar_chart':
            chart_data.update(self._extract_bar_data(cv_image, elements))
        elif chart_type == 'line_chart':
            chart_data.update(self._extract_line_data(cv_image, elements))
        elif chart_type == 'scatter_plot':
            chart_data.update(self._extract_scatter_data(cv_image, elements))
        elif chart_type == 'pie_chart':
            chart_data.update(self._extract_pie_data(cv_image, elements))

        return chart_data

    def _extract_bar_data(self, cv_image: np.ndarray, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data from bar charts (heuristic)"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Binarize to get solid shapes
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Significant area
                x, y, w, h = cv2.boundingRect(contour)
                if h == 0:
                    continue
                aspect_ratio = w / float(h)

                # Filter for bar-like shapes (vertical or horizontal)
                if (aspect_ratio > 2 and h > 50) or (aspect_ratio < 0.5 and w > 50):
                    bars.append({
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'area': float(area),
                        'height': float(h if aspect_ratio < 0.5 else w),  # Effective height
                        'position': int(x if aspect_ratio < 0.5 else y)   # Position along axis
                    })

        # Sort bars by position
        bars.sort(key=lambda b: b['position'])

        return {
            'bars': bars,
            'bar_count': len(bars),
            'estimated_values': [bar['height'] for bar in bars]
        }

    def _extract_line_data(self, cv_image: np.ndarray, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data from line charts (heuristic)"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find lines
        edges = cv2.Canny(gray, 50, 150)

        # Find data lines (excluding axes)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

        data_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = float(np.hypot(x2 - x1, y2 - y1))
                angle = float(np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))

                # Filter out horizontal and vertical lines (likely axes)
                if 10 < angle < 170 and length > 50:
                    data_lines.append({
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'length': length,
                        'angle': angle
                    })

        return {
            'data_lines': data_lines,
            'line_count': len(data_lines)
        }

    def _extract_scatter_data(self, cv_image: np.ndarray, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data from scatter plots"""
        data_points_element = next((e for e in elements if e.get('type') == 'data_points'), None)

        if data_points_element:
            points = data_points_element['points']
            xs = [p['x'] for p in points]
            ys = [p['y'] for p in points]
            return {
                'points': points,
                'point_count': len(points),
                'x_range': (int(min(xs)), int(max(xs))) if xs else (0, 0),
                'y_range': (int(min(ys)), int(max(ys))) if ys else (0, 0)
            }

        return {'points': [], 'point_count': 0}

    def _extract_pie_data(self, cv_image: np.ndarray, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data from pie charts (very rough heuristic)"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Emphasize large solid regions
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Significant circular-ish area
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 50:
                    circles.append({
                        'center': (int(x), int(y)),
                        'radius': float(radius),
                        'area': float(area)
                    })

        segments = []
        if circles:
            main_circle = max(circles, key=lambda c: c['radius'])
            segments = self._analyze_pie_segments(cv_image, main_circle)

        return {
            'circles': circles,
            'segments': segments,
            'segment_count': len(segments)
        }

    def _analyze_pie_segments(self, cv_image: np.ndarray, circle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze pie chart segments using naive color uniqueness (placeholder for clustering)"""
        center_x, center_y = circle['center']
        radius = int(circle['radius'])

        # Extract circular region
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Get colors within the circle (downsample for speed)
        ys = range(max(0, center_y - radius), min(cv_image.shape[0], center_y + radius), 3)
        xs = range(max(0, center_x - radius), min(cv_image.shape[1], center_x + radius), 3)

        unique_colors = []
        seen = set()
        for y in ys:
            for x in xs:
                if mask[y, x] > 0:
                    b, g, r = cv_image[y, x]
                    color = (int(b), int(g), int(r))
                    # Skip near-black
                    if (b + g + r) <= 50:
                        continue
                    if color not in seen:
                        seen.add(color)
                        unique_colors.append(color)

        segments: List[Dict[str, Any]] = []
        if unique_colors:
            max_segments = min(8, len(unique_colors))
            estimated = 100.0 / float(max_segments)
            for i, color in enumerate(unique_colors[:max_segments]):
                segments.append({
                    'color': color,
                    'estimated_percentage': estimated,
                    'segment_id': i
                })

        return segments

    def _generate_description(self, image_type: str, elements: List[Dict[str, Any]],
                              extracted_text: str, context: str) -> str:
        """Generate a comprehensive description of the image"""
        description_parts: List[str] = []

        # Start with image type
        description_parts.append(f"This appears to be a {image_type.replace('_', ' ')}.")

        # Add element information
        if image_type in ['bar_chart', 'line_chart', 'scatter_plot']:
            axes_info = [e for e in elements if 'axis' in e.get('type', '')]
            if axes_info:
                description_parts.append("The chart has clearly defined axes with grid lines.")

            data_info = [e for e in elements if 'data' in e.get('type', '')]
            if data_info:
                data_element = data_info[0]
                count = data_element.get('count', 0)
                description_parts.append(f"Contains approximately {count} data elements.")

        elif image_type == 'flow_chart':
            process_boxes = sum(e.get('count', 0) for e in elements if e.get('type') == 'process_boxes')
            decision_diamonds = sum(e.get('count', 0) for e in elements if e.get('type') == 'decision_diamonds')
            if process_boxes or decision_diamonds:
                description_parts.append(
                    f"The flowchart contains {process_boxes} process boxes and {decision_diamonds} decision diamonds."
                )

        elif image_type == 'table':
            table_grid = next((e for e in elements if e.get('type') == 'table_grid'), None)
            if table_grid:
                rows = table_grid.get('rows', 0)
                cols = table_grid.get('cols', 0)
                description_parts.append(f"The table has approximately {rows} rows and {cols} columns.")

        # Add text information
        if extracted_text:
            text_words = extracted_text.split()
            if len(text_words) > 5:
                description_parts.append("Text elements include labels and annotations.")
                numbers = re.findall(r'\d+\.?\d*', extracted_text)
                if numbers:
                    # Convert to floats for min/max comparison
                    nums = [float(n) for n in numbers]
                    description_parts.append(f"Contains numerical values ranging from {min(nums)} to {max(nums)}.")

        # Add context if available
        if context and len(context.strip()) > 10:
            description_parts.append(f"According to the caption: {context[:100]}...")

        return ' '.join(description_parts)

    def create_interactive_visualization(self, analysis_result: ImageAnalysisResult) -> Optional[str]:
        """Create an interactive visualization of the analyzed image (if possible)"""
        if not PLOTLY_AVAILABLE:
            return None

        if analysis_result.image_type == 'bar_chart' and analysis_result.chart_data:
            return self._create_bar_chart_viz(analysis_result.chart_data)
        elif analysis_result.image_type == 'scatter_plot' and analysis_result.chart_data:
            return self._create_scatter_plot_viz(analysis_result.chart_data)
        elif analysis_result.image_type == 'line_chart' and analysis_result.chart_data:
            return self._create_line_chart_viz(analysis_result.chart_data)

        return None

    def _create_bar_chart_viz(self, chart_data: Dict[str, Any]) -> Optional[str]:
        """Create interactive bar chart from extracted data"""
        if not PLOTLY_AVAILABLE or 'estimated_values' not in chart_data:
            return None

        values = chart_data['estimated_values']
        labels = [f"Bar {i+1}" for i in range(len(values))]

        fig = px.bar(x=labels, y=values, title="Reconstructed Bar Chart")
        fig.update_layout(
            xaxis_title="Categories",
            yaxis_title="Values",
            showlegend=False
        )

        return fig.to_html()

    def _create_scatter_plot_viz(self, chart_data: Dict[str, Any]) -> Optional[str]:
        """Create interactive scatter plot from extracted data"""
        if not PLOTLY_AVAILABLE or 'points' not in chart_data or not chart_data['points']:
            return None

        points = chart_data['points']
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]

        fig = px.scatter(x=x_coords, y=y_coords, title="Reconstructed Scatter Plot")
        fig.update_layout(
            xaxis_title="X-axis",
            yaxis_title="Y-axis"
        )

        return fig.to_html()

    def _create_line_chart_viz(self, chart_data: Dict[str, Any]) -> Optional[str]:
        """Create interactive line chart from extracted data"""
        if not PLOTLY_AVAILABLE or 'data_lines' not in chart_data:
            return None

        fig = go.Figure()

        for i, line in enumerate(chart_data['data_lines']):
            start_x, start_y = line['start']
            end_x, end_y = line['end']

            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode='lines',
                name=f'Line {i+1}',
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Reconstructed Line Chart",
            xaxis_title="X-axis",
            yaxis_title="Y-axis"
        )

        return fig.to_html()

    def batch_analyze_images(self, images_with_context: List[Tuple[Image.Image, str]]) -> List[ImageAnalysisResult]:
        """Analyze multiple images in batch"""
        results: List[ImageAnalysisResult] = []

        for image, context in images_with_context:
            try:
                result = self.analyze_image(image, context)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing image: {e}")
                # Create error result
                error_result = ImageAnalysisResult(
                    image_type='unknown',
                    confidence=0.0,
                    description=f"Error analyzing image: {str(e)}",
                    extracted_text="",
                    chart_data=None,
                    detected_elements=[],
                    metadata={}
                )
                results.append(error_result)

        return results
