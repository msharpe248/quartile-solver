from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import io
import os
import numpy as np
import cv2
from typing import List, Optional

app = FastAPI()

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

def load_dictionary():
    global word_list
    word_list = set()
    
    # Download word list if not exists
    dict_path = "static/words.txt"
    if not os.path.exists(dict_path):
        print("Downloading word list...")
        import urllib.request
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        urllib.request.urlretrieve(url, dict_path)
    
    # Load words
    with open(dict_path, "r") as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 2:  # Only add words of 2 or more letters
                word_list.add(word)
    print(f"Loaded {len(word_list)} words")

def find_words(tiles):
    # Convert all tiles to lowercase for consistency and remove empty tiles
    tiles = [t.lower() for t in tiles if t]
    
    # Get all possible combinations of tiles
    words = set()
    
    # First check single tiles
    for tile in tiles:
        if tile in word_list:
            words.add(tile)
    
    # Try combinations of 2-4 tiles
    from itertools import combinations, permutations
    for length in range(2, min(5, len(tiles) + 1)):
        for combo in combinations(tiles, length):
            # Try all possible orderings of these tiles
            for perm in permutations(combo):
                word = ''.join(perm)
                if word in word_list:
                    words.add(word)
    
    # Sort words by length (longest first) then alphabetically
    return sorted(words, key=lambda w: (-len(w), w))

def process_box(image, box):
    try:
        x, y, w, h = box
        
        # Extract the box with a small margin
        margin = 5
        roi = image[max(0, y+margin):min(image.shape[0], y+h-margin), 
                   max(0, x+margin):min(image.shape[1], x+w-margin)]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Save the processed tile for debugging
        #debug_path = os.path.join(os.getcwd(), "static", f"tile_{x}_{y}.png")
        #cv2.imwrite(debug_path, gray)
        
        # Configure Tesseract for this specific case
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz'
        
        # Get the text
        text = pytesseract.image_to_string(gray, config=custom_config).strip().lower()
        
        # Clean up the text (remove any newlines or extra spaces)
        text = ''.join(text.split())
        
        return text
        
    except Exception as e:
        print(f"Error processing box: {str(e)}")
        return ""

@app.on_event("startup")
async def startup_event():
    load_dictionary()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        contents = await file.read()

        # convert contents to an image
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Failed to read image"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Save edges for debugging
        #cv2.imwrite(os.path.join(os.getcwd(), "static", "edges.png"), edges)
        
        # Dilate edges to connect them
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours in the edge image
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort boxes
        boxes = []
        expected_box_count = 20  # We expect 5 rows × 4 columns
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take the top 20 largest contours that match our criteria
        for contour in contours:
            if len(boxes) >= expected_box_count:
                break
                
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            
            # Filter based on aspect ratio (boxes are about 1.47:1)
            if 1.3 < aspect_ratio < 1.6:
                boxes.append((x, y, w, h))
        
        # Sort boxes by position
        if boxes:
            # Calculate the average height to determine row grouping
            avg_height = sum(h for _, _, _, h in boxes) / len(boxes)
            row_threshold = avg_height / 2
            
            # Sort by row first (using y coordinate) then by x within each row
            boxes.sort(key=lambda b: (b[1] // int(row_threshold), b[0]))
        
        # Draw boxes for debugging
        #debug_image = image.copy()
        #for i, (x, y, w, h) in enumerate(boxes):
        #    cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #    cv2.putText(debug_image, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #cv2.imwrite(os.path.join(os.getcwd(), "static", "detected_boxes.png"), debug_image)
        
        # Process each box
        tiles = []
        
        for i, box in enumerate(boxes):
            try:
                text = process_box(image, box)
                if text:
                    tiles.append(text)
            except Exception as e:
                print(f"Error processing box {i+1}: {str(e)}")
        
        # Find possible words
        words = find_words(tiles)
        
        return {
            "tiles": ", ".join(tiles),
            "words": words
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/manual")
async def manual_input(request: Request):
    try:
        # Get the JSON data from the request
        data = await request.json()
        tiles = data.get("tiles", "").strip()
        
        if not tiles:
            return {"error": "No tiles provided"}
            
        # Split the tiles by comma and clean them up
        tiles_list = [t.strip().lower() for t in tiles.split(",") if t.strip()]
        
        if not tiles_list:
            return {"error": "No valid tiles found"}
            
        # Find possible words
        words = find_words(tiles_list)
        
        return {
            "tiles": ", ".join(tiles_list),
            "words": words
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
