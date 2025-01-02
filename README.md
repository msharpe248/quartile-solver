# Quartile Solver

A web application that helps solve Quartile puzzles by identifying tiles from uploaded images or manual input and finding all possible words that can be formed.

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for image processing)

### Installing Tesseract OCR

On macOS:
```bash
brew install tesseract
```

## Installation

1. Clone or download this repository
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory
2. Run the application:
```bash
python main.py
```
3. Open your web browser and go to `http://localhost:8000`

## Features

- Upload images of Quartile games to automatically detect tiles
- Manually input tiles
- View all possible words that can be formed from the tiles
- Edit detected tiles if necessary
- Responsive web interface

## Usage

1. **Upload Image Method**:
   - Click "Choose File" in the Upload Image section
   - Select an image of your Quartile game
   - Click "Upload and Solve"

2. **Manual Input Method**:
   - Enter the letters in the "Enter tiles" input field
   - Click "Solve"

The application will display:
- The detected/entered tiles
- A list of all possible words that can be formed using these tiles
