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

Please summarize the core content in a way suitable for a {MAX_PAGES}-page comic."""
    
    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(prompt)
    return response.text

def analyze_pdf(file_bytes: bytes, mime_type: str) -> str:
    """Analyze uploaded PDF/image content"""
    prompt = f"""With Nobita and Doraemon as the main characters, create a comic format that guides readers to learn and understand this content progressively from basic to advanced.

Please summarize the core content from this document. The comic should have a maximum of {MAX_PAGES} pages."""
    
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

Based on the above discussion, analyze and determine how this comic learning guide should be divided into pages, and what the content of each page should be. You must generate maximum of {MAX_PAGES} pages.

Return a JSON array with this structure:
[
  {{
    "pageNumber": 1,
    "description": "The narrative content of this page",
    "visualCue": "Description of the visual scene for the image generator"
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
    
    return json.loads(text.strip())

def generate_comic_image(context: str, page_plan: dict) -> bytes:
    """Generate a single comic page image"""
    prompt = f"""Based on the above discussion, generate a learning comic using Doraemon and Nobita characters for page {page_plan['pageNumber']} (portrait orientation with 3:4 aspect ratio, in English).

Context: {context}

Page Description: {page_plan['description']}
Visual Scene: {page_plan['visualCue']}

Style: Comic book art style with clear panels, speech bubbles, and engaging visuals suitable for educational content."""
    
    model = genai.GenerativeModel(IMAGE_MODEL)
    response = model.generate_content(prompt)
    
    # Extract image data
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data'):
                return base64.b64decode(part.inline_data.data)
    
    raise HTTPException(status_code=500, detail=f"Failed to generate image for page {page_plan['pageNumber']}")

def create_comic_pdf(pages_data: list) -> bytes:
    """Create a PDF from comic pages"""
    buffer = io.BytesIO()
    
    # Create PDF with letter size
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    for page in pages_data:
        # Load image
        img_bytes = page['image_bytes']
        img = Image.open(io.BytesIO(img_bytes))
        
        # Calculate dimensions to fit page while maintaining aspect ratio
        img_width, img_height = img.size
        aspect = img_height / img_width
        
        # Use 90% of page width, calculate height accordingly
        new_width = width * 0.9
        new_height = new_width * aspect
        
        # Center the image
        x = (width - new_width) / 2
        y = (height - new_height) / 2
        
        # Draw image
        img_reader = ImageReader(io.BytesIO(img_bytes))
        c.drawImage(img_reader, x, y, new_width, new_height)
        
        # Add page description at bottom
        c.setFont("Helvetica", 10)
        text_y = y - 30
        
        # Word wrap the description
        description = page['description']
        max_width = width * 0.9
        words = description.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            if c.stringWidth(test_line, "Helvetica", 10) > max_width:
                current_line.pop()
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text lines
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            c.drawString(x, text_y - (i * 12), line)
        
        # Add page number
        c.drawString(width / 2 - 20, 30, f"Page {page['page_number']}")
        
        c.showPage()
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Comicizer API - Transform educational content into comics!",
        "endpoints": {
            "/generate-from-topic": "POST - Generate comic from subject/topic details",
            "/generate-from-file": "POST - Generate comic from uploaded PDF/image"
        }
    }

@app.post("/generate-from-topic")
async def generate_from_topic(request: TopicRequest):
    """Generate comic PDF from topic information"""
    try:
        # Create context
        context = f"Subject: {request.subject}, Class: {request.class_name}, Chapter: {request.chapter}, Topic: {request.topic}"
        
        # Step 1: Analyze
        print(f"Analyzing topic: {request.topic}")
        analysis = analyze_content(request.topic, context)
        
        # Step 2: Plan story
        print("Planning comic structure...")
        plan = plan_story(analysis)
        
        # Step 3: Generate images
        pages_data = []
        for page_plan in plan:
            print(f"Generating page {page_plan['pageNumber']}...")
            image_bytes = generate_comic_image(analysis, page_plan)
            pages_data.append({
                'page_number': page_plan['pageNumber'],
                'description': page_plan['description'],
                'image_bytes': image_bytes
            })
        
        # Step 4: Create PDF
        print("Creating PDF...")
        pdf_bytes = create_comic_pdf(pages_data)
        
        # Return PDF
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=comic_{request.subject}_{request.topic}.pdf"
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
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
        print(f"Analyzing uploaded file: {file.filename}")
        analysis = analyze_pdf(file_bytes, file.content_type)
        
        # Step 2: Plan story
        print("Planning comic structure...")
        plan = plan_story(analysis)
        
        # Step 3: Generate images
        pages_data = []
        for page_plan in plan:
            print(f"Generating page {page_plan['pageNumber']}...")
            image_bytes = generate_comic_image(analysis, page_plan)
            pages_data.append({
                'page_number': page_plan['pageNumber'],
                'description': page_plan['description'],
                'image_bytes': image_bytes
            })
        
        # Step 4: Create PDF
        print("Creating PDF...")
        pdf_bytes = create_comic_pdf(pages_data)
        
        # Return PDF
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=comic_{file.filename.rsplit('.', 1)[0]}.pdf"
            }
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse story plan: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_configured": bool(API_KEY)}
