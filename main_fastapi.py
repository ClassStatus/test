"""
PDF Table Extractor API (Production)
====================================

This API allows users to upload a PDF file and automatically extract tables from it. The tables are then made available for download in various formats (HTML, Excel, CSV, JSON, Tally XML). The system also handles password-protected PDFs, image-based PDFs (using OCR), and cleans up temporary files after 10 minutes.

How it works (for non-developers):
- Upload a PDF file (optionally with a password).
- The system tries to extract tables using text extraction. If that fails, it uses OCR (reads the PDF as images).
- Extracted tables are saved in multiple formats.
- You get download links for each format.
- Files are automatically deleted after 10 minutes for privacy and storage management.

"""
# =============================
# Imports and Setup
# =============================
import os
import shutil
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pdfplumber
import pandas as pd
import io
import sys
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import re
from concurrent.futures import ThreadPoolExecutor
import logging
from paddlex import create_pipeline
from bs4 import BeautifulSoup
from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store temp files
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# FastAPI app setup with timeout configuration
app = FastAPI(
    title="Production PDF Table Extractor API",
    description="Upload PDF, get unique download links for HTML, Excel, CSV, JSON. Files auto-delete after 10 min.",
    version="2.0.0-prod"
)

# Allowed frontend domains (add your production domain here later)
ALLOWED_ORIGINS = {"http://localhost:3000", "http://localhost", "http://127.0.0.1:8000", "https://mywebsite.com", "*"}  # Allow all origins for now

def check_origin(request: Request):
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        # Allow requests without origin header for now
        return
    if not any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS):
        # Allow all origins for now to prevent CORS issues
        return

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Background Cleanup Job
# =============================
# Background cleanup job: delete files older than 10 min
CLEANUP_INTERVAL = 600  # seconds (10 min)
FILE_LIFETIME = 600     # seconds (10 min)

def cleanup_temp_files():
    try:
        now = time.time()
        for folder in os.listdir(TEMP_DIR):
            folder_path = os.path.join(TEMP_DIR, folder)
            if os.path.isdir(folder_path):
                # Check folder creation/modification time
                mtime = os.path.getmtime(folder_path)
                if now - mtime > FILE_LIFETIME:
                    try:
                        shutil.rmtree(folder_path)
                        logger.info(f"Cleaned up expired folder: {folder}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup folder {folder}: {e}")
    except Exception as e:
        logger.error(f"Cleanup job failed: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize PaddleX pipeline globally
paddlex_pipeline = create_pipeline(pipeline="table_recognition")

# =============================
# Helper Functions
# =============================

def extract_balances(tables, unique_tables=None):
    """
    Try to find opening and closing balances from the extracted tables.
    If multiple tables with the same headers are found, merge them and use the largest.
    Returns (opening_balance, closing_balance) or (None, None) if not found.
    """
    # If unique_tables is provided, use the largest merged table for balances
    if unique_tables:
        # Find the largest merged table (most rows)
        merged = None
        for dfs in unique_tables.values():
            merged_df = pd.concat(dfs, ignore_index=True)
            if merged is None or len(merged_df) > len(merged):
                merged = merged_df
        if merged is not None and not merged.empty:
            # Try to find balance column
            balance_col = None
            for col in merged.columns:
                if col is not None and 'balance' in str(col).lower():
                    balance_col = col
                    break
            if balance_col:
                opening = merged[balance_col].iloc[0]
                closing = merged[balance_col].iloc[-1]
                return opening, closing
    # Fallback: use first table
    if not tables:
        return None, None
    df = tables[0]['data']
    if df.empty:
        return None, None
    balance_col = None
    for col in df.columns:
        if col is not None and 'balance' in str(col).lower():
            balance_col = col
            break
    if balance_col:
        opening = df[balance_col].iloc[0]
        closing = df[balance_col].iloc[-1]
        return opening, closing
    return None, None

def to_tally_xml(tables):
    """
    Convert the first extracted table to Tally-compatible XML format for accounting import.
    Returns XML string.
    """
    # Only use the first table for Tally export
    if not tables:
        return ""
    df = tables[0]['data']
    if df.empty:
        return ""
    # Try to find columns
    date_col = None
    desc_col = None
    debit_col = None
    credit_col = None
    balance_col = None
    for col in df.columns:
        if col is None:
            continue
        lcol = str(col).lower()
        if not date_col and 'date' in lcol:
            date_col = col
        if not desc_col and ('desc' in lcol or 'particular' in lcol or 'narration' in lcol):
            desc_col = col
        if not debit_col and 'debit' in lcol:
            debit_col = col
        if not credit_col and 'credit' in lcol:
            credit_col = col
        if not balance_col and 'balance' in lcol:
            balance_col = col
    # Fallbacks
    if not date_col:
        date_col = df.columns[0] if len(df.columns) > 0 else None
    if not desc_col:
        desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0] if len(df.columns) > 0 else None
    # Build XML
    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ENVELOPE>',
        ' <HEADER>',
        '  <TALLYREQUEST>Import Data</TALLYREQUEST>',
        ' </HEADER>',
        ' <BODY>',
        '  <IMPORTDATA>',
        '   <REQUESTDESC>',
        '    <REPORTNAME>Vouchers</REPORTNAME>',
        '   </REQUESTDESC>',
        '   <REQUESTDATA>',
    ]
    for _, row in df.iterrows():
        date_val = str(row[date_col]) if date_col and date_col in row else ''
        desc_val = str(row[desc_col]) if desc_col and desc_col in row else ''
        debit_val = str(row[debit_col]) if debit_col and debit_col in row else ''
        credit_val = str(row[credit_col]) if credit_col and credit_col in row else ''
        balance_val = str(row[balance_col]) if balance_col and balance_col in row else ''
        xml.append('    <TALLYMESSAGE>')
        xml.append('     <VOUCHER VCHTYPE="Bank Statement" ACTION="Create">')
        xml.append(f'      <DATE>{date_val}</DATE>')
        xml.append(f'      <NARRATION>{desc_val}</NARRATION>')
        if debit_val:
            xml.append(f'      <DEBIT>{debit_val}</DEBIT>')
        if credit_val:
            xml.append(f'      <CREDIT>{credit_val}</CREDIT>')
        if balance_val:
            xml.append(f'      <BALANCE>{balance_val}</BALANCE>')
        xml.append('     </VOUCHER>')
        xml.append('    </TALLYMESSAGE>')
    xml += [
        '   </REQUESTDATA>',
        '  </IMPORTDATA>',
        ' </BODY>',
        '</ENVELOPE>'
    ]
    return '\n'.join(xml)

def clean_multiline_cells(df):
    """
    Cleans up multi-line cells in a DataFrame by replacing line breaks with spaces,
    so multi-line data stays in the same cell and does not create new rows.
    """
    return df.applymap(lambda x: str(x).replace('\n', ' ') if isinstance(x, str) else x)

def extract_and_save(pdf_bytes, output_directory, password=None, file_map=None):
    """
    Main extraction function. Tries text-based extraction first, then OCR if needed.
    Saves tables in all requested formats. Returns summary info for the API response.
    """
    logger.info(f"[extract_and_save] Called with output dir: {output_directory}")
    tables = []
    unique_tables = {}  # key: tuple(headers), value: list of DataFrames
    non_blank_pages = set()
    ocr_used = False
    ocr_message = None
    try:
        logger.info("[extract_and_save] Attempting to open PDF with pdfplumber...")
        # Try to open PDF and extract tables using text extraction
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
        if pdf is None:
            logger.error("[extract_and_save] Failed to open PDF - may be password protected")
            raise Exception("Failed to open PDF - may be password protected")
        if not hasattr(pdf, 'pages') or pdf.pages is None:
            logger.error("[extract_and_save] PDF appears to be password protected or corrupted")
            raise Exception("PDF is password protected or corrupted")
        logger.info(f"[extract_and_save] PDF opened successfully. Total pages: {len(pdf.pages)}")
        for page_num, page in enumerate(pdf.pages, 1):
            if page is None:
                logger.warning(f"[extract_and_save] Page {page_num} is None, skipping.")
                continue
            found_table = False
            for table in page.find_tables():
                data = table.extract()
                if data and len(data) > 1:
                    logger.info(f"[extract_and_save] Table found on page {page_num}, rows: {len(data)-1}")
                    df = pd.DataFrame(data[1:], columns=data[0])
                    tables.append({"page": page_num, "data": df})
                    headers_key = tuple(df.columns)
                    if headers_key not in unique_tables:
                        unique_tables[headers_key] = []
                    unique_tables[headers_key].append(df)
                    found_table = True
            if found_table:
                non_blank_pages.add(page_num)
        pdf.close()
        logger.info(f"[extract_and_save] Finished text-based extraction. Tables found: {len(tables)}")
    except Exception as e:
        logger.error(f"[extract_and_save] Exception during text-based extraction: {e}")
        try:
            if 'pdf' in locals() and pdf is not None:
                pdf.close()
        except:
            pass
        err_msg = str(e) if e is not None else "Unknown error"
        err_msg_lower = err_msg.lower() if err_msg else ""
        password_keywords = [
            "password", "encrypted", "incorrect password", "protected", 
            "authentication", "security", "locked", "restricted",
            "requires password", "password required", "access denied"
        ]
        is_password_error = any(keyword in err_msg_lower for keyword in password_keywords)
        if is_password_error:
            if password:
                raise Exception("Incorrect PDF password")
            else:
                raise Exception("PDF is password protected")
        else:
            if password is None:
                try:
                    test_pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password="")
                    test_pdf.close()
                except:
                    raise Exception("PDF is password protected")
            raise e

    # If no tables found with text extraction, try PaddleX OCR
    if not tables:
        logger.info("[extract_and_save] No tables found with text extraction, attempting PaddleX OCR...")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            ocr_used = True
            ocr_message = "PaddleX OCR was used to process image-based PDF"
            for page_num, img in enumerate(images, 1):
                temp_img_path = os.path.join(output_directory, f"paddlex_page_{page_num}.png")
                img.save(temp_img_path)
                output_gen = paddlex_pipeline.predict(
                    input=temp_img_path,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
                output = next(output_gen) if hasattr(output_gen, '__iter__') and not isinstance(output_gen, dict) else output_gen
                if 'table_res_list' in output:
                    for table_res in output['table_res_list']:
                        html = table_res.get('pred_html')
                        if html:
                            soup = BeautifulSoup(html, 'html.parser')
                            table_tag = soup.find('table')
                            if table_tag:
                                rows = []
                                for tr in table_tag.find_all('tr'):
                                    row = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
                                    if row:
                                        rows.append(row)
                                if rows:
                                    max_cols = max(len(row) for row in rows)
                                    padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
                                    header = padded_rows[0]
                                    if len(header) < max_cols:
                                        header = [f"Column {i+1}" for i in range(max_cols)]
                                    data_rows = padded_rows[1:] if len(padded_rows) > 1 else []
                                    df = pd.DataFrame(data_rows, columns=header)
                                    df = clean_multiline_cells(df)
                                    tables.append({"page": page_num, "data": df})
                                    headers_key = tuple(df.columns)
                                    if headers_key not in unique_tables:
                                        unique_tables[headers_key] = []
                                    unique_tables[headers_key].append(df)
                                    non_blank_pages.add(page_num)
            logger.info(f"[extract_and_save] PaddleX OCR extraction complete. Tables found: {len(tables)}")
        except Exception as ocr_e:
            logger.error(f"[extract_and_save] PaddleX OCR failed: {ocr_e}")
            ocr_message = f"PaddleX OCR failed: {str(ocr_e)}"

    # If still no tables found, try PaddleOCR as a fallback
    if not tables:
        logger.info("[extract_and_save] No tables found with PaddleX OCR, attempting PaddleOCR main project...")
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det=True, rec=True, structure=True)
            images = convert_from_bytes(pdf_bytes, dpi=200)
            for page_num, img in enumerate(images, 1):
                img_np = np.array(img)
                # PaddleOCR expects BGR
                if img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                result = ocr.ocr(img_np, cls=True, det=True, rec=True, structure=True)
                # result is a list of tables and text blocks
                for res in result:
                    if isinstance(res, dict) and 'html' in res:
                        html = res['html']
                        soup = BeautifulSoup(html, 'html.parser')
                        table_tag = soup.find('table')
                        if table_tag:
                            rows = []
                            for tr in table_tag.find_all('tr'):
                                row = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
                                if row:
                                    rows.append(row)
                            if rows:
                                max_cols = max(len(row) for row in rows)
                                padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
                                header = padded_rows[0]
                                if len(header) < max_cols:
                                    header = [f"Column {i+1}" for i in range(max_cols)]
                                data_rows = padded_rows[1:] if len(padded_rows) > 1 else []
                                df = pd.DataFrame(data_rows, columns=header)
                                df = clean_multiline_cells(df)
                                tables.append({"page": page_num, "data": df})
                                headers_key = tuple(df.columns)
                                if headers_key not in unique_tables:
                                    unique_tables[headers_key] = []
                                unique_tables[headers_key].append(df)
                                non_blank_pages.add(page_num)
            logger.info(f"[extract_and_save] PaddleOCR extraction complete. Tables found: {len(tables)}")
            ocr_used = True
            ocr_message = (ocr_message or "") + " | PaddleOCR main project was used as a fallback."
        except Exception as paddleocr_e:
            logger.error(f"[extract_and_save] PaddleOCR main project failed: {paddleocr_e}")
            ocr_message = (ocr_message or "") + f" | PaddleOCR main project failed: {str(paddleocr_e)}"

    # Save files in different formats
    if tables and file_map:
        try:
            # Save Excel
            if "excel" in file_map:
                with pd.ExcelWriter(os.path.join(output_directory, file_map["excel"]), engine='xlsxwriter') as writer:
                    for i, table in enumerate(tables):
                        sheet_name = f"Page_{table['page']}" if len(tables) > 1 else "Tables"
                        table['data'].to_excel(writer, sheet_name=sheet_name, index=False)

            # Save CSV
            if "csv" in file_map:
                if len(tables) == 1:
                    tables[0]['data'].to_csv(os.path.join(output_directory, file_map["csv"]), index=False)
                else:
                    # Save first table as CSV
                    tables[0]['data'].to_csv(os.path.join(output_directory, file_map["csv"]), index=False)

            # Save JSON
            if "json" in file_map:
                json_data = []
                for table in tables:
                    json_data.append({
                        "page": table['page'],
                        "data": table['data'].to_dict(orient="records")
                    })
                import json
                with open(os.path.join(output_directory, file_map["json"]), "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, default=str)

            # Save Tally XML
            if "tallyxml" in file_map:
                tally_xml = to_tally_xml(tables)
                with open(os.path.join(output_directory, file_map["tallyxml"]), "w", encoding="utf-8") as f:
                    f.write(tally_xml)

        except Exception as save_e:
            logger.error(f"[extract_and_save] Error saving files: {save_e}")
            raise Exception(f"Failed to save output files: {str(save_e)}")

    # Calculate balances
    opening_balance, closing_balance = extract_balances(tables, unique_tables)
    
    return len(tables), len(non_blank_pages) if non_blank_pages else 0, opening_balance, closing_balance, ocr_used, ocr_message

# =============================
# Async Wrapper for Extraction
# =============================

async def extract_and_save_async(pdf_bytes, output_directory, password=None, file_map=None):
    """
    Async wrapper for extract_and_save to avoid blocking the server.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, extract_and_save, pdf_bytes, output_directory, password, file_map)

# =============================
# API Endpoints
# =============================

# Remove HTML from supported formats
SUPPORTED_FORMATS = ["excel", "csv", "json", "tallyxml"]

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
    request: Request = None,
    _: None = Depends(check_origin)
):
    """
    Endpoint to upload a PDF file and extract tables from it.
    - Accepts a PDF file and optional password.
    - Returns download links for extracted tables in various formats (except HTML).
    - Always returns 'pages_count' and 'process_time' in the response.
    - Handles errors and provides user-friendly messages.
    """
    import time
    logger.info("[upload_pdf] Received upload request.")
    start_time = time.time()
    
    # Use minutes for process_time
    def get_process_time_minutes(start_time):
        return round((time.time() - start_time) / 60, 2)

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning("[upload_pdf] Invalid file type.")
        return {
            "success": False, 
            "error_code": "INVALID_FILE_TYPE",
            "message": "Only PDF files are allowed. Please upload a PDF file.",
            "details": "The uploaded file must have a .pdf extension.",
            "pages_count": 0,
            "process_time": get_process_time_minutes(start_time)
        }
    
    # Read file with timeout
    try:
        pdf_bytes = await asyncio.wait_for(file.read(), timeout=30.0)  # 30 second timeout
        logger.info(f"[upload_pdf] File '{file.filename}' read into memory. Size: {len(pdf_bytes)} bytes.")
    except asyncio.TimeoutError:
        logger.error("[upload_pdf] File read timeout")
        return {
            "success": False,
            "error_code": "FILE_READ_TIMEOUT",
            "message": "File upload took too long.",
            "details": "Please try with a smaller PDF file.",
            "pages_count": 0,
            "process_time": get_process_time_minutes(start_time)
        }
    
    file_id = str(uuid.uuid4())
    output_directory = os.path.join(TEMP_DIR, file_id)
    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"[upload_pdf] Output directory created: {output_directory}")
    
    # Prepare output file names for each format (no HTML)
    base_name = os.path.splitext(file.filename)[0]
    file_map = {
        "excel": f"{base_name}.xlsx", 
        "csv": f"{base_name}.csv",
        "json": f"{base_name}.json",
        "tallyxml": f"{base_name}_tally.xml"
    }
    
    # Save the original PDF for reference
    with open(os.path.join(output_directory, "original.pdf"), "wb") as f:
        f.write(pdf_bytes)
    logger.info(f"[upload_pdf] Original PDF saved to {os.path.join(output_directory, 'original.pdf')}")
    
    try:
        logger.info("[upload_pdf] Calling extract_and_save_async...")
        result = await asyncio.wait_for(
            extract_and_save_async(pdf_bytes, output_directory, password=password, file_map=file_map),
            timeout=120.0  # 2 minute timeout for processing
        )
        tables_found, pages_count, opening_balance, closing_balance, ocr_used, ocr_message = result
        logger.info(f"[upload_pdf] Extraction complete. Tables found: {tables_found}, Pages: {pages_count}, OCR used: {ocr_used}")
        
        tables = []
        unique_tables = {}
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
        if pdf is None or not hasattr(pdf, 'pages') or pdf.pages is None:
            raise Exception("Failed to open PDF for merged tables extraction")
        for page_num, page in enumerate(pdf.pages, 1):
            if page is None:
                continue
            for table in page.find_tables():
                data = table.extract()
                if data and len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                    tables.append({"page": page_num, "data": df})
                    headers_key = tuple(df.columns)
                    if headers_key not in unique_tables:
                        unique_tables[headers_key] = []
                    unique_tables[headers_key].append(df)
        pdf.close()
        merged_tables_json = []
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_tables_json.append({
                "columns": list(merged_df.columns),
                "rows": merged_df.to_dict(orient="records")
            })
    except asyncio.TimeoutError:
        logger.error("[upload_pdf] Processing timeout")
        shutil.rmtree(output_directory)
        return {
            "success": False,
            "error_code": "PROCESSING_TIMEOUT",
            "message": "PDF processing took too long.",
            "details": "Please try with a smaller or simpler PDF file.",
            "pages_count": 0,
            "process_time": get_process_time_minutes(start_time)
        }
    except Exception as e:
        logger.error(f"[upload_pdf] Exception: {e}")
        try:
            if 'pdf' in locals() and pdf is not None:
                pdf.close()
        except:
            pass
        err_msg = str(e) if e is not None else "Unknown error"
        err_msg_lower = err_msg.lower() if err_msg else ""
        if "password" in err_msg_lower or "encrypted" in err_msg_lower or "incorrect password" in err_msg_lower or "protected" in err_msg_lower:
            shutil.rmtree(output_directory)
            if password:
                return {
                    "success": False,
                    "error_code": "INCORRECT_PASSWORD", 
                    "message": "The provided password is incorrect.",
                    "details": "Please check your password and try again.",
                    "pages_count": 0,
                    "process_time": get_process_time_minutes(start_time)
                }
            else:
                return {
                    "success": False,
                    "error_code": "PASSWORD_REQUIRED",
                    "message": "This PDF is password protected.",
                    "details": "Please provide the password to extract tables.",
                    "pages_count": 0,
                    "process_time": get_process_time_minutes(start_time)
                }
        elif "corrupted" in err_msg_lower or "damaged" in err_msg_lower:
            shutil.rmtree(output_directory)
            return {
                "success": False,
                "error_code": "CORRUPTED_FILE",
                "message": "The PDF file appears to be corrupted or damaged.",
                "details": "Please try uploading a different PDF file.",
                "pages_count": 0,
                "process_time": get_process_time_minutes(start_time)
            }
        elif "unsupported" in err_msg_lower or "format" in err_msg_lower:
            shutil.rmtree(output_directory)
            return {
                "success": False,
                "error_code": "UNSUPPORTED_FORMAT",
                "message": "This PDF format is not supported.",
                "details": "Please try with a different PDF file.",
                "pages_count": 0,
                "process_time": get_process_time_minutes(start_time)
            }
        else:
            shutil.rmtree(output_directory)
            return {
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "message": "Failed to process the PDF file.",
                "details": f"Error: {err_msg}",
                "pages_count": 0,
                "process_time": get_process_time_minutes(start_time)
            }
    if tables_found == 0:
        logger.info("[upload_pdf] No tables found. Cleaning up output directory.")
        shutil.rmtree(output_directory)
        if ocr_used:
            return {
                "success": False,
                "error_code": "NO_TABLES_FOUND",
                "message": "🔍 No Tables Found",
                "details": ocr_message if ocr_message else "No tables could be extracted from this PDF, even with OCR.",
                "pages_count": pages_count,
                "process_time": get_process_time_minutes(start_time),
                "ocr_used": ocr_used
            }
        else:
            return {
                "success": False,
                "error_code": "NO_TABLES_FOUND",
                "message": "📋 No Tables Found",
                "details": f"Processed {pages_count} pages but found no extractable tables. This PDF might be image-based or contain no tabular data.",
                "pages_count": pages_count,
                "process_time": get_process_time_minutes(start_time),
                "ocr_used": ocr_used
            }
    links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
    response_data = {
        "success": True,
        "tables_found": tables_found,
        "pages_count": pages_count,
        "file_id": file_id,
        "download_links": links,
        "output_file_names": file_map,
        "opening_balance": opening_balance,
        "closing_balance": closing_balance,
        "merged_tables_json": merged_tables_json,
        "ocr_used": ocr_used,
        "process_time": get_process_time_minutes(start_time)
    }
    if ocr_used and ocr_message:
        response_data["ocr_message"] = ocr_message
    logger.info(f"[upload_pdf] Returning response for file_id: {file_id}")
    return response_data

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    """
    Endpoint to download the extracted table in the requested format.
    - file_id: Unique ID for the uploaded PDF session.
    - fmt: Format to download (excel, csv, json, tallyxml).
    Returns the file if found, or a user-friendly error if not.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid format. Please choose one of: excel, csv, json, tallyxml.")
    safe_id = file_id.replace("..", "")
    output_directory = os.path.join(TEMP_DIR, safe_id)
    if not os.path.exists(output_directory):
        raise HTTPException(status_code=404, detail="File not found or expired. Please re-upload your PDF.")
    files = os.listdir(output_directory)
    file_name = None
    ext_map = {
        "excel": ".xlsx", 
        "csv": ".csv",
        "json": ".json",
        "tallyxml": "_tally.xml"
    }
    for f in files:
        if fmt == "tallyxml" and f.endswith(ext_map[fmt]):
            file_name = f
            break
        elif f.endswith(ext_map[fmt]):
            file_name = f
            break
    if not file_name:
        raise HTTPException(status_code=404, detail="Requested format not found. Please try another format or re-upload your PDF.")
    file_path = os.path.join(output_directory, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired. Please re-upload your PDF.")
    media_types = {
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json",
        "tallyxml": "application/xml"
    }
    return FileResponse(file_path, media_type=media_types[fmt], filename=file_name)

@app.get("/")
def root():
    """
    Root endpoint. Explains how to use the API in plain English.
    """
    return {"message": "Production PDF Table Extractor API. POST /upload with PDF, get download links."}

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring. Returns status, timestamp, version, and number of temp files.
    Useful for both developers and non-developers to check if the service is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-prod",
        "temp_files_count": len(os.listdir(TEMP_DIR)) if os.path.exists(TEMP_DIR) else 0
    } 