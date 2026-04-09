import io
import json
import os
import logging
import tempfile
import signal
import time
from collections import defaultdict
import pdfplumber
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import easyocr
import numpy as np
from datetime import datetime, timedelta
import hashlib

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------
# Configuration & Validation
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout settings (in seconds)
PDF_EXTRACTION_TIMEOUT = 60
GEMINI_TIMEOUT = 120

# Initialize OCR reader (same as desktop app)
ocr_reader = None

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        try:
            ocr_reader = easyocr.Reader(['en'])
            logger.info("✅ OCR reader initialized")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
    return ocr_reader

# ------------------------------------------------------------------
# Simple Request Cache (prevents duplicate API calls)
# ------------------------------------------------------------------
_request_cache = {}
_cache_expiry = 3600  # 1 hour

def get_cache_key(text: str) -> str:
    """ Generate cache key from text hash."""
    return hashlib.md5(text[:1000].encode()).hexdigest()

def get_cached_result(text: str):
    """Get cached parsing result if available."""
    key = get_cache_key(text)
    if key in _request_cache:
        cached_time, cached_data = _request_cache[key]
        if time.time() - cached_time < _cache_expiry:
            logger.info("✅ Using cached result")
            return cached_data
        else:
            del _request_cache[key]
    return None

def cache_result(text: str, data):
    """ Cache parsing result."""
    key = get_cache_key(text)
    _request_cache[key] = (time.time(), data)

# ------------------------------------------------------------------
# Model selection with automatic fallback & quota awareness
# ------------------------------------------------------------------
# Models ordered by quota limits (free tier most restrictive)
MODEL_CONFIGS = [
    {"name": "gemini-1.5-flash", "quota": 20, "delay": 3},  # Free tier: 20 req/day
    {"name": "gemini-2.0-flash", "quota": 20, "delay": 3},
    {"name": "gemini-2.5-flash", "quota": 20, "delay": 3},
    {"name": "gemini-pro", "quota": 50, "delay": 2},
]

model = None
MODEL_NAME = None
MODEL_QUOTA_REMAINING = 20
LAST_REQUEST_TIME = None

def select_model():
    """Select best available model with quota awareness."""
    global model, MODEL_NAME, MODEL_QUOTA_REMAINING
    
    for config in MODEL_CONFIGS:
        try:
            candidate = config["name"]
            logger.info(f"Attempting to initialize {candidate}...")
            tmp_model = genai.GenerativeModel(candidate)
            
            # Light test call
            resp = tmp_model.generate_content("OK")
            if resp and resp.text:
                model = tmp_model
                MODEL_NAME = candidate
                MODEL_QUOTA_REMAINING = config["quota"]
                logger.info(f"✅ Using model: {MODEL_NAME} (quota: {MODEL_QUOTA_REMAINING}/day)")
                return True
        except Exception as e:
            logger.warning(f"❌ {config['name']} unavailable: {str(e)[:100]}")
            continue
    
    logger.warning("⚠️  All Gemini models unavailable - using MOCK mode")
    return False

def handle_quota_error(error_str: str) -> tuple[bool, int]:
    """
    Handle quota exceeded errors.
    Returns (should_retry, wait_seconds)
    """
    if "429" in str(error_str) or "quota" in str(error_str).lower():
        # Extract retry delay if available
        if "retry" in error_str.lower():
            try:
                # Look for "retry in X seconds" pattern
                import re
                match = re.search(r'retry[^0-9]*(\d+\.?\d*)', error_str, re.IGNORECASE)
                if match:
                    wait = max(int(float(match.group(1))) + 2, 30)
                    logger.warning(f"⚠️  Rate limited. Waiting {wait}s before retry...")
                    return True, wait
            except:
                pass
        return True, 30  # Default 30 second wait
    return False, 0

# Initialize model on startup
select_model()

# ------------------------------------------------------------------
# PDF text extraction with chunking for large files
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF with coordinate-based matching and chunking."""
    pdfplumber = __import__('pdfplumber')
    structured_text = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            # Limit to first 10 pages for large PDFs (to reduce API calls)
            pages_to_process = min(total_pages, 10)
            if pages_to_process < total_pages:
                logger.info(f"⚠️  Large PDF ({total_pages} pages), limiting to first {pages_to_process}")
            
            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                logger.info(f"Reading page {page_num + 1}/{pages_to_process}...")
                structured_text.append(f"\n--- Page {page_num + 1} ---\n")
                
                # Extract words with coordinates
                words = page.extract_words(x_tolerance=3, y_tolerance=3)
                
                if not words:
                    t = page.extract_text()
                    if t:
                        structured_text.append(t)
                    continue
                
                # Group words by Y coordinate (same line)
                lines_dict = defaultdict(list)
                for word in words:
                    y = round(word['top'])
                    lines_dict[y].append(word)
                
                # Sort lines by Y position
                sorted_lines = sorted(lines_dict.items())
                
                # Process lines and match size/quantity by coordinates
                skip_next = False
                for i, (y, line_words) in enumerate(sorted_lines):
                    if skip_next:
                        skip_next = False
                        continue
                    
                    # Sort words in line by X position (left to right)
                    line_words = sorted(line_words, key=lambda w: w['x0'])
                    
                    # Check if this is a size line
                    size_tokens = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', 'XXXXL', 'XXXXXL']
                    size_words = [w for w in line_words if w['text'] in size_tokens]
                    
                    # Also check for numeric sizes
                    numeric_sizes = [w for w in line_words if w['text'].isdigit() and len(w['text']) == 2]
                    if len(numeric_sizes) >= 3:
                        size_words = numeric_sizes
                    
                    # Need at least 3 sizes to be considered a size header row
                    if len(size_words) >= 3:
                        # Look for quantities in the next line
                        if i + 1 < len(sorted_lines):
                            next_y, next_words = sorted_lines[i + 1]
                            qty_words = [w for w in next_words if w['text'].isdigit()]
                            
                            if qty_words and len(qty_words) <= len(size_words):
                                # COORDINATE-BASED MATCHING
                                pairs = []
                                
                                for qty_word in qty_words:
                                    qty_x = qty_word['x0']
                                    qty_val = qty_word['text']
                                    
                                    # Find the size whose X position is closest to this quantity
                                    best_size = min(size_words, key=lambda s: abs(s['x0'] - qty_x))
                                    best_size_val = best_size['text']
                                    
                                    pairs.append(f"{best_size_val}:{qty_val}")
                                
                                # Output structured format
                                if size_words[0]['text'].isdigit():
                                    pair_str = ' | '.join([f"Size{p}" for p in pairs])
                                else:
                                    pair_str = ' | '.join(pairs)
                                
                                structured_text.append(f"SIZEQUANTITY: {pair_str}")
                                skip_next = True
                                continue
                    
                    # Regular text line
                    text_line = ' '.join([w['text'] for w in line_words])
                    
                    # Merge color name and code if detected
                    if 'Color Name:' in text_line:
                        color_name = text_line.replace('Color Name:', '').strip()
                        # Check next line for color code
                        if i + 1 < len(sorted_lines):
                            next_y, next_words = sorted_lines[i + 1]
                            next_text = ' '.join([w['text'] for w in sorted(next_words, key=lambda w: w['x0'])])
                            if 'Color Code:' in next_text:
                                color_code = next_text.replace('Color Code:', '').strip()
                                structured_text.append(f"Color: {color_name} (Code: {color_code})")
                                skip_next = True
                                continue
                    
                    # Mark price lines
                    if 'US$' in text_line or '$' in text_line:
                        if not any(k in text_line for k in ['Retail', 'Wholesale', 'Discount', 'Total', 'Price', 'MSRP']):
                            text_line = f"PRICING: {text_line}"
                    
                    structured_text.append(text_line)
        
        result_text = '\n'.join(structured_text)
        
        # Check if we got meaningful text
        if len(result_text.strip()) < 100:
            logger.info("PDF appears scanned, running OCR...")
            return extract_text_from_pdf_ocr(pdf_bytes)
        
        return result_text
        
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise Exception(f"Error reading PDF: {str(e)}")

# ------------------------------------------------------------------
# OCR fallback for scanned PDFs
# ------------------------------------------------------------------
def extract_text_from_pdf_ocr(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF using OCR."""
    try:
        pdfplumber = __import__('pdfplumber')
        reader = get_ocr_reader()
        if not reader:
            raise Exception("OCR reader not available")
        
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            # Limit OCR to first 3 pages for large PDFs
            pages_to_process = min(total_pages, 3)
            
            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                logger.info(f"OCR processing page {page_num + 1}/{pages_to_process}...")
                # Convert page to image
                img = page.to_image(resolution=150)  # Lower resolution for speed
                img_np = np.array(img.original)
                results = reader.readtext(img_np, detail=0)
                text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                text += "\n".join(results) + "\n"
        
        if pages_to_process < total_pages:
            text += f"\n... (OCR limited to first {pages_to_process} of {total_pages} pages)"
        
        return text
    except Exception as e:
        raise Exception(f"OCR error: {str(e)}")

# ------------------------------------------------------------------
# Image extraction
# ------------------------------------------------------------------
def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR."""
    try:
        reader = get_ocr_reader()
        if not reader:
            raise Exception("OCR reader not available")
        
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Open image with PIL
            img = Image.open(tmp_path)
            # Resize large images
            if img.size[0] > 2000 or img.size[1] > 2000:
                img.thumbnail((2000, 2000))
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0)
            return "\n".join(results)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise Exception(f"Error reading image: {str(e)}")

# ------------------------------------------------------------------
# Split large text into chunks for processing
# ------------------------------------------------------------------
def split_text_into_chunks(text: str, max_chunk_size: int = 10000):
    """Split large text into smaller chunks (smaller for free tier)."""
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1
        
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

# ------------------------------------------------------------------
# Gemini parsing with chunking, caching, and retry logic
# ------------------------------------------------------------------
def parse_with_gemini(text: str):
    """Parse extracted text with Gemini - handles quota & retries."""
    global LAST_REQUEST_TIME, MODEL_QUOTA_REMAINING
    
    # Check cache first
    cached = get_cached_result(text)
    if cached:
        return cached
    
    # Rate limiting for free tier
    if LAST_REQUEST_TIME:
        elapsed = time.time() - LAST_REQUEST_TIME
        min_delay = 2  # Minimum 2 seconds between requests
        if elapsed < min_delay:
            wait = min_delay - elapsed
            logger.info(f"Rate limiting: waiting {wait:.1f}s...")
            time.sleep(wait)
    
    # If text is too large, split into chunks
    if len(text) > 20000:
        logger.info(f"Text too large ({len(text)} chars), splitting into chunks...")
        chunks = split_text_into_chunks(text, max_chunk_size=10000)
        logger.info(f"Split into {len(chunks)} chunks")
        
        all_items = []
        for idx, chunk in enumerate(chunks, 1):
            if MODEL_QUOTA_REMAINING <= 1:
                logger.warning("⚠️  Approaching quota limit, stopping chunked processing")
                break
            
            logger.info(f"Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
            try:
                chunk_items = parse_with_gemini_single(chunk)
                all_items.extend(chunk_items)
                MODEL_QUOTA_REMAINING -= 1
                LAST_REQUEST_TIME = time.time()
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {str(e)}")
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.error("Quota exceeded, stopping chunk processing")
                    break
                continue
        
        # Deduplicate items
        seen = set()
        unique_items = []
        for item in all_items:
            key = f"{item.get('product', '')}_{item.get('color_name', '')}_{item.get('size', '')}"
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        logger.info(f"Total unique items from chunks: {len(unique_items)}")
        cache_result(text, unique_items)
        return unique_items
    
    # Single request for smaller text
    try:
        items = parse_with_gemini_single(text)
        MODEL_QUOTA_REMAINING -= 1
        LAST_REQUEST_TIME = time.time()
        cache_result(text, items)
        return items
    except Exception as e:
        should_retry, wait_time = handle_quota_error(str(e))
        if should_retry and wait_time > 0:
            logger.error(f"Rate limited, would need to wait {wait_time}s")
            raise Exception(f"Rate limited (quota exceeded). Please retry in {wait_time} seconds.")
        raise

def parse_with_gemini_single(text: str):
    """Parse a single chunk of text with Gemini."""
    if len(text) > 15000:
        text = text[:15000] + "\n... (truncated)"

    prompt = f"""You are an expert invoice parser. Extract ALL product/item information into structured JSON.

**CRITICAL: SIZE/QUANTITY PARSING RULES**

The text contains preprocessed lines like:
"SIZEQUANTITY: M:1 | L:2 | XL:2 | XXL:1"

This means:
- Size M has quantity 1
- Size L has quantity 2
- Size XL has quantity 2
- Size XXL has quantity 1
- Any size NOT in the list has ZERO quantity and should NOT be included

**CREATE ONE ROW PER SIZE THAT HAS A QUANTITY**

**FIELDS TO EXTRACT**:
- product: Full product name
- color_name: Color name
- color_code: Color code
- size: SINGLE size value (M, L, XL)
- quantity: INTEGER units for THIS specific size
- wholesale_price: Unit wholesale price (per item)

**OUTPUT FORMAT**:
Return ONLY valid JSON with no markdown:
{{"items": [{{"product": "...", "color_name": "...", "color_code": "...", "size": "M", "quantity": "1", "wholesale_price": "40.00"}}]}}

Document text:
{text}

Return only the JSON object with the "items" array."""

    if not model:
        logger.warning("No Gemini model available, returning MOCK data")
        return [{"product": "Mock Product", "color_name": "Red", "color_code": "R01", "size": "M", "quantity": "2", "wholesale_price": "45.00"}]

    try:
        # Generate content
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")

        result = response.text.strip()
        
        # Clean markdown
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        # Find JSON object
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]

        data = json.loads(result)
        
        # Extract items
        if isinstance(data, dict):
            items = data.get("items", data.get("products", [data]))
        elif isinstance(data, list):
            items = data
        else:
            return []

        # Validate and fix items
        items = validate_and_fix_items(items)
        return items

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise Exception(f"Gemini AI error: {str(e)}")

# ------------------------------------------------------------------
# Validate and fix items
# ------------------------------------------------------------------
def validate_and_fix_items(items):
    """Validate and fix parsed items."""
    fixed_items = []
    
    for item in items:
        fixed_item = item.copy()
        
        # Fix quantity field
        if 'quantity' in fixed_item:
            try:
                qty_str = str(fixed_item['quantity']).replace(',', '').strip()
                qty_val = float(qty_str)
                
                # If looks like price (decimal or > 1000)
                if '.' in qty_str or qty_val > 1000:
                    if 'total_cost' not in fixed_item or not fixed_item['total_cost']:
                        fixed_item['total_cost'] = qty_str
                    
                    # Try to calculate from unit price
                    unit_price = None
                    for field in ['wholesale_price', 'unit_price', 'msrp']:
                        if field in fixed_item and fixed_item[field]:
                            try:
                                unit_price = float(str(fixed_item[field]).replace(',', '').replace('$', '').strip())
                                break
                            except:
                                pass
                    
                    if unit_price and unit_price > 0:
                        calc_qty = qty_val / unit_price
                        fixed_item['quantity'] = str(int(round(calc_qty)))
                    else:
                        fixed_item['quantity'] = "1"
                else:
                    fixed_item['quantity'] = str(int(round(qty_val)))
            except:
                fixed_item['quantity'] = "1"
        
        if 'quantity' not in fixed_item or not fixed_item['quantity']:
            fixed_item['quantity'] = "1"
        
        # Fix size field - ensure single value
        if 'size' in fixed_item and fixed_item['size']:
            size_str = str(fixed_item['size'])
            if ',' in size_str or '|' in size_str:
                fixed_item['size'] = size_str.split(',')[0].split('|')[0].strip()
        
        fixed_items.append(fixed_item)
    
    return fixed_items

# ------------------------------------------------------------------
# Flask App with CORS and timeout handling
# ------------------------------------------------------------------
app = Flask(__name__)

# Allow all origins (for Codespaces & Vercel)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.route('/api/parse', methods=['POST', 'OPTIONS'])
def parse_document():
    """Main parse endpoint - handles PDF, images, and OCR."""
    if request.method == 'OPTIONS':
        return '', 204

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Log file size for debugging
    file_bytes = file.read()
    file_size_kb = len(file_bytes) / 1024
    logger.info(f"File size: {file_size_kb:.1f} KB")
    
    file_ext = file.filename.split('.')[-1].lower()
    
    try:
        # Route based on file type
        if file_ext == 'pdf':
            logger.info(f"Processing PDF: {file.filename}")
            extracted_text = extract_text_from_pdf(file_bytes)
        elif file_ext in ['png', 'jpg', 'jpeg']:
            logger.info(f"Processing image: {file.filename}")
            extracted_text = extract_text_from_image(file_bytes)
        else:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        text_length = len(extracted_text.strip())
        logger.info(f"Extracted text length: {text_length} characters")
        
        if text_length < 50:
            return jsonify({'error': 'Document has no selectable text. Try a different file.'}), 400

        # Parse with Gemini
        items = parse_with_gemini(extracted_text)
        
        logger.info(f"Successfully parsed {len(items)} items")
        return jsonify({'items': items, 'raw_text': extracted_text[:5000]})  # Limit raw text
        
    except Exception as e:
        logger.exception("Parsing failed")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint for Railway/Vercel"""
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'status': 'healthy', 
        'backend': 'available', 
        'model': MODEL_NAME or "MOCK",
        'ocr_available': ocr_reader is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - welcome message"""
    return jsonify({
        'message': 'ShopKeep Parser Backend',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'parse': '/api/parse'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)