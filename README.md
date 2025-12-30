# MorseVision

Millions of people with mobility challenges like Parkinson's disease and ALS struggle with traditional communication methods. MorseVision provides an accessible alternative: a hands-free, voice-free way to communicate using only eye blinks.

With MorseVision, you can translate eye blinks into text and communicate with AI. MorseVision detects intentional blinks (measuring their duration and sequence), converts them to Morse code, and processes the resulting text through an AI assistant for intelligent responses.

## Tech Stack

- **Python (Flask)** - Web framework for serving the app
- **OpenCV (cv2)** - Video processing and face detection
- **cvzone** - Face mesh detection and utilities
- **Mistral AI** - AI chat assistant

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   - Open `morse.py`, and replace `"Your API KEY"` with your Mistral AI API key:
     ```python
     MISTRAL_API_KEY = "your_actual_api_key_here"
     ```

3. **Run the application:**
   ```bash
   python morse.py
   ```

4. **Access the app:**
   - Open your browser and navigate to `http://localhost:5001`

