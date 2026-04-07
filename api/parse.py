import io
import json
import os
import logging
from collections import defaultdict
import pdfplumber
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

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

# Try to use a stable model
MODEL_NAME = "gemini-2.5-flash"  # widely available
try:
    model = genai.GenerativeModel(MODEL_NAME)
    # Quick test
    test_response = model.generate_content("Hello")
    if not test_response.text:
        raise Exception("Model returned empty response")
    logger.info(f"✅ Using model: {MODEL_NAME}")
except Exception as e:
    logger.warning(f"{MODEL_NAME} failed: {e}, falling back to gemini-pro")
    MODEL_NAME = "gemini-2.5-flash"
    model = genai.GenerativeModel(MODEL_NAME)

# ------------------------------------------------------------------
# PDF text extraction (coordinate‑based)
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    structured_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            structured_text.append(f"\n--- Page {page_num} ---\n")
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                t = page.extract_text()
                if t:
                    structured_text.append(t)
                continue

            lines_dict = defaultdict(list)
            for w in words:
                y = round(w['top'])
                lines_dict[y].append(w)

            sorted_lines = sorted(lines_dict.items())
            skip_next = False
            for i, (y, line_words) in enumerate(sorted_lines):
                if skip_next:
                    skip_next = False
                    continue

                line_words = sorted(line_words, key=lambda w: w['x0'])
                size_tokens = {'XS','S','M','L','XL','XXL','XXXL','XXXXL','XXXXXL'}
                size_words = [w for w in line_words if w['text'] in size_tokens]
                numeric_sizes = [w for w in line_words if w['text'].isdigit() and len(w['text'])==2]
                if len(numeric_sizes) >= 3:
                    size_words = numeric_sizes

                if len(size_words) >= 3 and i+1 < len(sorted_lines):
                    next_y, next_words = sorted_lines[i+1]
                    qty_words = [w for w in next_words if w['text'].isdigit()]
                    if qty_words and len(qty_words) <= len(size_words):
                        pairs = []
                        for qw in qty_words:
                            best = min(size_words, key=lambda s: abs(s['x0'] - qw['x0']))
                            pairs.append(f"{best['text']}:{qw['text']}")
                        if size_words[0]['text'].isdigit():
                            pair_str = ' | '.join([f"Size{p}" for p in pairs])
                        else:
                            pair_str = ' | '.join(pairs)
                        structured_text.append(f"SIZEQUANTITY: {pair_str}")
                        skip_next = True
                        continue

                text_line = ' '.join([w['text'] for w in line_words])
                if 'Color Name:' in text_line and i+1 < len(sorted_lines):
                    next_y, next_words = sorted_lines[i+1]
                    next_text = ' '.join([w['text'] for w in sorted(next_words, key=lambda w: w['x0'])])
                    if 'Color Code:' in next_text:
                        color_name = text_line.replace('Color Name:', '').strip()
                        color_code = next_text.replace('Color Code:', '').strip()
                        structured_text.append(f"Color: {color_name} (Code: {color_code})")
                        skip_next = True
                        continue

                if 'US$' in text_line or '$' in text_line:
                    if not any(k in text_line for k in ['Retail','Wholesale','Discount','Total','Price','MSRP']):
                        text_line = f"PRICING: {text_line}"
                structured_text.append(text_line)

    return '\n'.join(structured_text)

# ------------------------------------------------------------------
# Gemini parsing with detailed error reporting
# ------------------------------------------------------------------
def parse_with_gemini(text: str):
    if len(text) > 30000:
        text = text[:30000] + "\n... (truncated)"

    prompt = f"""You are an expert invoice parser that returns ONLY valid JSON.

Extract all product line items from the document below. 
Each item must have:
- "product" (description or SKU, if missing use "Unknown product")
- "color_name" (if present, else empty string)
- "color_code" (if present, else empty string)
- "size" (e.g., "S", "M", "L", "XL" or numeric like "32", else empty string)
- "quantity" (as a string, e.g., "2")
- "wholesale_price" (numeric string without $, e.g., "45.99")

Rules:
- If a field is not found, use an empty string (not null).
- If there are no items, return {{"items": []}}.
- Do NOT include any markdown, explanation or extra text.

Example response:
{{"items": [
    {{"product": "Classic Tee", "color_name": "Navy", "color_code": "NVY", "size": "M", "quantity": "4", "wholesale_price": "19.99"}},
    {{"product": "Hoodie", "color_name": "Black", "color_code": "BLK", "size": "XL", "quantity": "2", "wholesale_price": "45.00"}}
]}}

Document text:
{text}
"""

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
        result = response.text.strip()
        logger.info(f"Gemini raw response (first 500 chars): {result[:500]}")

        # Clean markdown
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        data = json.loads(result)
        if isinstance(data, dict):
            items = data.get("items", data.get("products", []))
            return items if isinstance(items, list) else []
        elif isinstance(data, list):
            return data
        else:
            logger.warning(f"Unexpected JSON type: {type(data)}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}\nRaw result: {result}")
        raise Exception(f"Gemini returned invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Gemini parsing error: {str(e)}")
        raise Exception(f"Gemini AI error: {str(e)}")

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/parse', methods=['POST', 'OPTIONS'])
def parse_document():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    pdf_bytes = file.read()
    try:
        extracted_text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted text length: {len(extracted_text)} characters")

        if len(extracted_text.strip()) < 50:
            return jsonify({'error': 'PDF has no selectable text. Use a text‑based PDF.'}), 400

        items = parse_with_gemini(extracted_text)
        return jsonify({'items': items, 'raw_text': extracted_text})
    except Exception as e:
        logger.exception("Parsing failed")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'backend': 'available', 'model': MODEL_NAME})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))    
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)