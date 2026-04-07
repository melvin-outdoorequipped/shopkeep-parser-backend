import io
import json
import os
import logging
import tempfile
from collections import defaultdict
import pdfplumber
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import easyocr
import numpy as np

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
# Model selection with automatic fallback (same as desktop app)
# ------------------------------------------------------------------
FREE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro", 
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-pro"
]

model = None
MODEL_NAME = None

for candidate in FREE_MODELS:
    try:
        tmp_model = genai.GenerativeModel(candidate)
        # Test call to see if free quota is available
        resp = tmp_model.generate_content("Hello")
        if resp and resp.text:
            model = tmp_model
            MODEL_NAME = candidate
            logger.info(f"✅ Using model: {MODEL_NAME}")
            break
    except Exception as e:
        logger.warning(f"{candidate} unavailable: {e}")

if not model:
    logger.warning("All Gemini models unavailable, using mock fallback")

# ------------------------------------------------------------------
# PDF text extraction (EXACT MATCH with desktop application)
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF with COORDINATE-BASED size/quantity matching.
    This EXACTLY matches the desktop application's parsing logic.
    """
    pdfplumber = __import__('pdfplumber')
    structured_text = []
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        
        for page_num, page in enumerate(pdf.pages, 1):
            logger.info(f"Reading page {page_num}/{total_pages}...")
            structured_text.append(f"\n--- Page {page_num} ---\n")
            
            # Extract words with coordinates (same as desktop app)
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            
            if not words:
                # Fallback to regular text extraction
                t = page.extract_text()
                if t:
                    structured_text.append(t)
                continue
            
            # Group words by Y coordinate (same line) - EXACT MATCH
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
                
                # Check if this is a size line (contains multiple size labels)
                size_tokens = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', 'XXXXL', 'XXXXXL']
                size_words = [w for w in line_words if w['text'] in size_tokens]
                
                # Also check for numeric sizes (28, 29, 30, etc.) - EXACT MATCH
                numeric_sizes = [w for w in line_words if w['text'].isdigit() and len(w['text']) == 2]
                if len(numeric_sizes) >= 3:
                    size_words = numeric_sizes
                
                # Need at least 3 sizes to be considered a size header row
                if len(size_words) >= 3:
                    # Look for quantities in the next line
                    if i + 1 < len(sorted_lines):
                        next_y, next_words = sorted_lines[i + 1]
                        
                        # Check if next line contains only numbers (quantities)
                        qty_words = [w for w in next_words if w['text'].isdigit()]
                        
                        if qty_words and len(qty_words) <= len(size_words):
                            # COORDINATE-BASED MATCHING! - EXACT MATCH
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
                                # Numeric sizes
                                pair_str = ' | '.join([f"Size{p}" for p in pairs])
                            else:
                                # Letter sizes
                                pair_str = ' | '.join(pairs)
                            
                            structured_text.append(f"SIZEQUANTITY: {pair_str}")
                            skip_next = True  # Skip the quantity line
                            continue
                
                # Regular text line
                text_line = ' '.join([w['text'] for w in line_words])
                
                # Merge color name and code if detected - EXACT MATCH
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
                
                # Mark price lines - EXACT MATCH
                if 'US$' in text_line or '$' in text_line:
                    if not any(k in text_line for k in ['Retail', 'Wholesale', 'Discount', 'Total', 'Price', 'MSRP']):
                        text_line = f"PRICING: {text_line}"
                
                structured_text.append(text_line)
    
    result_text = '\n'.join(structured_text)
    
    # Check if we got meaningful text (same as desktop app)
    if len(result_text.strip()) < 100:
        logger.info("PDF appears scanned, running OCR...")
        return extract_text_from_pdf_ocr(pdf_bytes)
    
    return result_text

# ------------------------------------------------------------------
# OCR fallback for scanned PDFs (EXACT MATCH with desktop app)
# ------------------------------------------------------------------
def extract_text_from_pdf_ocr(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF using OCR - EXACT MATCH with desktop app."""
    try:
        pdfplumber = __import__('pdfplumber')
        reader = get_ocr_reader()
        if not reader:
            raise Exception("OCR reader not available")
        
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"OCR processing page {page_num}/{total_pages}...")
                # Convert page to image
                img = page.to_image(resolution=300)
                img_np = np.array(img.original)
                results = reader.readtext(img_np, detail=0)
                text += f"\n--- Page {page_num} (OCR) ---\n"
                text += "\n".join(results) + "\n"
        return text
    except Exception as e:
        raise Exception(f"OCR error: {str(e)}")

# ------------------------------------------------------------------
# Image extraction (EXACT MATCH with desktop app)
# ------------------------------------------------------------------
def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR - EXACT MATCH with desktop app."""
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
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0)
            return "\n".join(results)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise Exception(f"Error reading image: {str(e)}")

# ------------------------------------------------------------------
# Gemini parsing with ENHANCED PROMPT (EXACT MATCH with desktop app)
# ------------------------------------------------------------------
def parse_with_gemini(text: str):
    """Parse extracted text with Gemini - EXACT MATCH with desktop app prompt."""
    if len(text) > 30000:
        text = text[:30000] + "\n... (truncated)"

    # EXACT SAME PROMPT as desktop application
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

Example:
Product: VIKEN SS SHIRT
Color: KHAKI (Code: 078)
SIZEQUANTITY: M:1 | L:2 | XL:2 | XXL:1
Wholesale: US$40.00

Output 4 separate rows:
{{"product": "VIKEN SS SHIRT", "color_name": "KHAKI", "color_code": "078", "size": "M", "quantity": "1", "wholesale_price": "40.00"}},
{{"product": "VIKEN SS SHIRT", "color_name": "KHAKI", "color_code": "078", "size": "L", "quantity": "2", "wholesale_price": "40.00"}},
{{"product": "VIKEN SS SHIRT", "color_name": "KHAKI", "color_code": "078", "size": "XL", "quantity": "2", "wholesale_price": "40.00"}},
{{"product": "VIKEN SS SHIRT", "color_name": "KHAKI", "color_code": "078", "size": "XXL", "quantity": "1", "wholesale_price": "40.00"}}

**QUANTITY RULES**:
- quantity = INTEGER ONLY (1, 2, 5, never 240.00)
- If you see "6 US$240.00", the 6 is the TOTAL quantity for ALL sizes combined, $240.00 is total_cost
- Lines with "PRICING:" or "US$" are prices, NOT quantities
- Each size row should have its own individual quantity from the SIZEQUANTITY line

**FIELDS TO EXTRACT**:
- product: Full product name
- brand: Brand if present
- style: Style number/SKU
- upc: UPC barcode if available
- product_code: Secondary product code
- color_name: Color name
- color_code: Color code
- size: SINGLE size value (M, L, XL, not "M, L, XL")
- quantity: INTEGER units for THIS specific size
- wholesale_price: Unit wholesale price (per item)
- unit_price: Unit retail price
- msrp: MSRP if shown
- discount: Discount amount or percentage
- total_cost: Line total for this size variant (quantity × price)

**OUTPUT FORMAT**:
Return ONLY valid JSON with no markdown, no backticks, no explanations:
{{
  "items": [
    {{"product": "...", "style": "...", "color_name": "...", "color_code": "...", "size": "M", "quantity": "1", "wholesale_price": "40.00", ...}},
    {{"product": "...", "style": "...", "color_name": "...", "color_code": "...", "size": "L", "quantity": "2", "wholesale_price": "40.00", ...}}
  ]
}}

Document text:
{text}

Return only the JSON object with the "items" array."""

    if not model:
        logger.warning("No Gemini model available, returning MOCK data")
        return [
            {"product": "Mock Product", "color_name": "Red", "color_code": "R01", "size": "M", "quantity": "2", "wholesale_price": "45.00"}
        ]

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")

        result = response.text.strip()
        
        # Clean markdown - EXACT MATCH with desktop app
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        data = json.loads(result)
        
        # Extract items - EXACT MATCH with desktop app
        if isinstance(data, dict):
            items = data.get("items", data.get("products", [data]))
        elif isinstance(data, list):
            items = data
        else:
            raise Exception("Unexpected response format")

        # Validate and fix items - EXACT MATCH with desktop app
        items = validate_and_fix_items(items)
        return items

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Gemini response: {str(e)}\nResponse: {result[:500]}")
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")

# ------------------------------------------------------------------
# Validate and fix items (EXACT MATCH with desktop app)
# ------------------------------------------------------------------
def validate_and_fix_items(items):
    """Validate and fix parsed items - EXACT MATCH with desktop app."""
    fixed_items = []
    
    for item in items:
        fixed_item = item.copy()
        
        # Fix quantity field - EXACT MATCH
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
        
        # Fix size field - ensure single value - EXACT MATCH
        if 'size' in fixed_item and fixed_item['size']:
            size_str = str(fixed_item['size'])
            if ',' in size_str or '|' in size_str:
                fixed_item['size'] = size_str.split(',')[0].split('|')[0].strip()
        
        fixed_items.append(fixed_item)
    
    return fixed_items

# ------------------------------------------------------------------
# Flask App with CORS
# ------------------------------------------------------------------
app = Flask(__name__)

# Allow all origins (for Codespaces)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
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

    file_bytes = file.read()
    file_ext = file.filename.split('.')[-1].lower()
    
    try:
        # Route based on file type (EXACT MATCH with desktop app)
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
        return jsonify({'items': items, 'raw_text': extracted_text})
        
    except Exception as e:
        logger.exception("Parsing failed")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'status': 'healthy', 
        'backend': 'available', 
        'model': MODEL_NAME or "MOCK",
        'ocr_available': ocr_reader is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
