from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
import base64
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Comicizer API", version="1.0.0")

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configuration
TEXT_MODEL = 'gemini-3-pro-preview'
IMAGE_MODEL = 'gemini-3-pro-image-preview'
MAX_PAGES = 3

# Configure Google Generative AI
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

# Request Models
class TopicRequest(BaseModel):
    subject: str
    topic: str
    class_name: str
    chapter: str

# Helper Functions
def analyze_content(content: str, context: str) -> str:
    """Analyze content and create comic-friendly summary"""
    prompt = f"""With Nobita and Doraemon as the main characters, create a comic format that guides readers to learn and understand this topic progressively from basic to advanced.
    
Context: {context}
Content: {content}

Please summarize the core content in a way suitable for a {MAX_PAGES}-page comic. Each page should be a SEPARATE, STANDALONE comic page."""
    
    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(prompt)
    return response.text

def analyze_pdf(file_bytes: bytes, mime_type: str) -> str:
    """Analyze uploaded PDF/image content"""
    prompt = f"""With Nobita and Doraemon as the main characters, create a comic format that guides readers to learn and understand this content progressively from basic to advanced.

Please summarize the core content from this document. The comic should have a maximum of {MAX_PAGES} SEPARATE pages."""
    
    model = genai.GenerativeModel(TEXT_MODEL)
    
    # Create the content parts
    file_data = {
        'mime_type': mime_type,
        'data': base64.b64encode(file_bytes).decode('utf-8')
    }
    
    response = model.generate_content([prompt, file_data])
    return response.text

def plan_story(analysis_context: str) -> list:
    """Plan the comic book structure"""
    prompt = f"""Context: {analysis_context}

Based on the above discussion, analyze and determine how this comic learning guide should be divided into {MAX_PAGES} SEPARATE pages, and what the content of each INDIVIDUAL page should be.

IMPORTANT: Each page should be completely separate and standalone. Do not combine multiple pages into one.

Return a JSON array with this structure:
[
  {{
    "pageNumber": 1,
    "description": "The narrative content of this SINGLE page only",
    "visualCue": "Description of the visual scene for THIS SINGLE PAGE ONLY"
  }},
  {{
    "pageNumber": 2,
    "description": "The narrative content of this SINGLE page only",
    "visualCue": "Description of the visual scene for THIS SINGLE PAGE ONLY"
  }},
  {{
    "pageNumber": 3,
    "description": "The narrative content of this SINGLE page only",
    "visualCue": "Description of the visual scene for THIS SINGLE PAGE ONLY"
  }}
]

Return ONLY the JSON array, no additional text."""
    
    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(prompt)
    
    # Extract JSON from response
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    pages = json.loads(text.strip())
    
    # Ensure we have exactly MAX_PAGES
    if len(pages) > MAX_PAGES:
        pages = pages[:MAX_PAGES]
    elif len(pages) < MAX_PAGES:
        print(f"Warning: Only {len(pages)} pages generated, expected {MAX_PAGES}")
    
    return pages

def generate_comic_image(context: str, page_plan: dict, total_pages: int) -> bytes:
    """Generate a single comic page image"""
    prompt = f"""Generate ONE SINGLE comic book page (page {page_plan['pageNumber']} of {total_pages}) using Doraemon and Nobita characters.

IMPORTANT: Generate ONLY ONE PAGE, not multiple pages. This is a single comic page in portrait orientation (3:4 aspect ratio).

Context: {context}

THIS PAGE's Content: {page_plan['description']}
THIS PAGE's Visual Scene: {page_plan['visualCue']}

Create a SINGLE comic page with:
- Clear comic panels showing the story progression for THIS PAGE ONLY
- Speech bubbles with educational dialogue
- Engaging visuals featuring Doraemon and Nobita
- Educational content presented in an entertaining way
- Page number "{page_plan['pageNumber']}" visible somewhere on the page

Style: Professional comic book art style, suitable for educational content, in English.

Remember: This is ONLY page {page_plan['pageNumber']}, not a multi-page spread."""
    
    model = genai.GenerativeModel(IMAGE_MODEL)
    
    # Configure generation with aspect ratio
    generation_config = {
        "temperature": 0.7,
    }
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    # Extract image data from response
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Check for inline_data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            img_data = part.inline_data.data
                            if isinstance(img_data, str):
                                return base64.b64decode(img_data)
                            return img_data
                        
                        # Check for direct data attribute
                        if hasattr(part, 'data') and part.data:
                            if isinstance(part.data, str):
                                return base64.b64decode(part.data)
                            return part.data
    
    raise HTTPException(
        status_code=500, 
        detail=f"Failed to generate image for page {page_plan['pageNumber']}. Response structure unexpected."
    )

def create_comic_pdf(pages_data: list) -> bytes:
    """Create a PDF from comic pages - ONE PAGE PER PDF PAGE"""
    buffer = io.BytesIO()
    
    # Create PDF with letter size
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    print(f"\n=== Creating PDF with {len(pages_data)} pages ===")
    
    for idx, page in enumerate(pages_data):
        try:
            print(f"\nProcessing PDF page {idx + 1}/{len(pages_data)} (Page Number: {page['page_number']})")
            
            # Load image
            img_bytes = page['image_bytes']
            
            # Verify we have valid image data
            if not img_bytes:
                raise ValueError("Empty image data")
            
            print(f"Image data size: {len(img_bytes)} bytes")
            
            # Try to open and verify the image
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Verify it's a valid image
            
            # Reopen after verify (verify closes the file)
            img = Image.open(io.BytesIO(img_bytes))
            
            print(f"Image size: {img.size}, Mode: {img.mode}")
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save to a new BytesIO for ImageReader
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Calculate dimensions to fit page while maintaining aspect ratio
            img_width, img_height = img.size
            aspect = img_height / img_width
            
            # Use maximum available space (leaving small margins)
            margin = 40
            available_width = width - (2 * margin)
            available_height = height - (2 * margin) - 60  # Extra space for page number
            
            # Calculate size maintaining aspect ratio
            if available_width * aspect <= available_height:
                # Width is limiting factor
                new_width = available_width
                new_height = new_width * aspect
            else:
                # Height is limiting factor
                new_height = available_height
                new_width = new_height / aspect
            
            # Center the image
            x = (width - new_width) / 2
            y = (height - new_height) / 2
            
            print(f"Drawing image at ({x}, {y}) with size ({new_width}, {new_height})")
            
            # Draw image
            img_reader = ImageReader(img_buffer)
            c.drawImage(img_reader, x, y, new_width, new_height)
            
            # Add page number at bottom center
            c.setFont("Helvetica-Bold", 14)
            page_text = f"Page {page['page_number']} of {len(pages_data)}"
            text_width = c.stringWidth(page_text, "Helvetica-Bold", 14)
            c.drawString((width - text_width) / 2, 20, page_text)
            
            print(f"Completed page {idx + 1}, calling showPage()")
            
            # THIS IS CRITICAL: Create a new page in the PDF
            c.showPage()
            
        except Exception as e:
            print(f"Error processing page {page.get('page_number', '?')}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Add error page
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, height / 2, f"Error loading page {page.get('page_number', '?')}")
            c.setFont("Helvetica", 12)
            c.drawString(100, height / 2 - 30, str(e))
            c.showPage()
    
    print("\nSaving PDF...")
    c.save()
    buffer.seek(0)
    pdf_size = len(buffer.getvalue())
    print(f"PDF created successfully! Size: {pdf_size} bytes")
    
    return buffer.getvalue()

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Comicizer API - Transform educational content into comics!",
        "version": "1.0.1 - Fixed page separation",
        "endpoints": {
            "/generate": "POST - Generate comic from subject/topic details",
            "/generate-from-file": "POST - Generate comic from uploaded PDF/image",
            "/health": "GET - Check API health"
        }
    }

@app.post("/generate")
async def generate_from_topic(request: TopicRequest):
    """Generate comic PDF from topic information"""
    try:
        # Create context
        context = f"Subject: {request.subject}, Class: {request.class_name}, Chapter: {request.chapter}, Topic: {request.topic}"
        
        # Step 1: Analyze
        print(f"\n=== Starting comic generation for topic: {request.topic} ===")
        print("Step 1: Analyzing content...")
        analysis = analyze_content(request.topic, context)
        
        # Step 2: Plan story
        print("\nStep 2: Planning comic structure...")
        plan = plan_story(analysis)
        print(f"Generated plan with {len(plan)} pages")
        
        # Step 3: Generate images for each page SEPARATELY
        print("\nStep 3: Generating images...")
        pages_data = []
        total_pages = len(plan)
        
        for i, page_plan in enumerate(plan):
            print(f"\n--- Generating page {page_plan['pageNumber']} of {total_pages} ---")
            print(f"Description: {page_plan['description'][:100]}...")
            
            image_bytes = generate_comic_image(analysis, page_plan, total_pages)
            
            print(f"Generated image for page {page_plan['pageNumber']}: {len(image_bytes)} bytes")
            
            pages_data.append({
                'page_number': page_plan['pageNumber'],
                'description': page_plan['description'],
                'image_bytes': image_bytes
            })
        
        # Step 4: Create PDF with separate pages
        print("\nStep 4: Creating PDF with separate pages...")
        pdf_bytes = create_comic_pdf(pages_data)
        
        print(f"\n=== Comic generation completed! PDF size: {len(pdf_bytes)} bytes ===\n")
        
        # Return PDF
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=comic_{request.subject}_{request.topic.replace(' ', '_')}.pdf"
            }
        )
        
    except Exception as e:
        print(f"\n!!! Error in generate_from_topic: {str(e)} !!!")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-from-file")
async def generate_from_file(file: UploadFile = File(...)):
    """Generate comic PDF from uploaded PDF or image"""
    try:
        # Validate file type
        if not file.content_type:
            raise HTTPException(status_code=400, detail="Could not determine file type")
        
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read file
        file_bytes = await file.read()
        
        # Step 1: Analyze uploaded content
        print(f"\n=== Starting comic generation from file: {file.filename} ===")
        print("Step 1: Analyzing uploaded content...")
        analysis = analyze_pdf(file_bytes, file.content_type)
        
        # Step 2: Plan story
        print("\nStep 2: Planning comic structure...")
        plan = plan_story(analysis)
        print(f"Generated plan with {len(plan)} pages")
        
        # Step 3: Generate images for each page SEPARATELY
        print("\nStep 3: Generating images...")
        pages_data = []
        total_pages = len(plan)
        
        for i, page_plan in enumerate(plan):
            print(f"\n--- Generating page {page_plan['pageNumber']} of {total_pages} ---")
            print(f"Description: {page_plan['description'][:100]}...")
            
            image_bytes = generate_comic_image(analysis, page_plan, total_pages)
            
            print(f"Generated image for page {page_plan['pageNumber']}: {len(image_bytes)} bytes")
            
            pages_data.append({
                'page_number': page_plan['pageNumber'],
                'description': page_plan['description'],
                'image_bytes': image_bytes
            })
        
        # Step 4: Create PDF with separate pages
        print("\nStep 4: Creating PDF with separate pages...")
        pdf_bytes = create_comic_pdf(pages_data)
        
        print(f"\n=== Comic generation completed! PDF size: {len(pdf_bytes)} bytes ===\n")
        
        # Return PDF
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=comic_{file.filename.rsplit('.', 1)[0]}.pdf"
            }
        )
        
    except json.JSONDecodeError as e:
        print(f"\n!!! JSON parsing error: {str(e)} !!!")
        raise HTTPException(status_code=500, detail=f"Failed to parse story plan: {str(e)}")
    except Exception as e:
        print(f"\n!!! Error in generate_from_file: {str(e)} !!!")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "api_configured": bool(API_KEY),
        "max_pages": MAX_PAGES,
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL
    }