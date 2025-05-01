from huggingface_hub import notebook_login

notebook_login()


import re
import json
import torch
import logging
from PIL import Image
from symspellpy import SymSpell, Verbosity
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
IMAGE_PATH = "/content/Maggie.jpg" # Replace with your image path


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_OCR_CHARS = 3000
MAX_INGREDIENTS = 100
SYM_SPELL_EDIT_DISTANCE = 1

# --- Setup ---
try:
   sym_spell = SymSpell(max_dictionary_edit_distance=SYM_SPELL_EDIT_DISTANCE, prefix_length=7)
   sym_spell.load_dictionary('frequency_dict.txt', term_index=0, count_index=1)
   logging.info("SymSpell dictionary loaded.")
except Exception as e:
   logging.error(f"Failed to load SymSpell dictionary: {e}")
   sym_spell = None
   
  # --- OCR Function---
def ocr_to_text(image_path):
    """Performs OCR on the image and returns the extracted text."""
    try:
      img = Image.open(image_path)
      img_gray = img.convert('L')
      text = pytesseract.image_to_string(img_gray, config='--psm 6').strip()
      logging.info(f"OCR successful for {image_path}. Text length: {len(text)}")
      return text[:MAX_OCR_CHARS]
    except FileNotFoundError:
       logging.error(f"Error: Image file not found at {image_path}")
       return None
    except Exception as e:
       logging.error(f"Error during OCR processing: {e}")
       return None
# --- Ingredient Cleaning Function  ---
def clean_ingredients(text):
   """Cleans the OCR text to extract a list of ingredients."""
   if not text:
      return []
   text = ' '.join(text.split()).lower()
   ingredient_markers = ["ingredients:", "contains:", "ingredients :", "contains :"]
   start_index = -1
   ingredient_text = text
   for marker in ingredient_markers:
      try:
         idx = text.index(marker)
          # Check if this marker is preceded by nutrition info
         preceding_text = text[:idx]
         if "nutrition facts" not in preceding_text and "serving size" not in preceding_text:
            start_index = idx + len(marker)
         logging.info(f"Found potential ingredient marker '{marker}'")
         ingredient_text = text[start_index:]
         break # Use the first valid marker found
      except ValueError:
         continue
   if start_index == -1:
      logging.warning("Could not find a clear ingredient start marker. Attempting cleanup on full text.")
   end_markers = ["nutrition facts", "serving size", "% daily value", "manufactured by", "distributed by", "produced by"]
   for marker in end_markers:
      marker_index = ingredient_text.find(marker)
      if marker_index != -1:
         logging.info(f"Removing text after '{marker}'")
         ingredient_text = ingredient_text[:marker_index].strip()   
         ingredient_text = re.sub(r'less than \d+% of\s*[:]*\s*', '', ingredient_text, flags=re.IGNORECASE)
         ingredient_text = re.sub(r'contains \d+% or less of\s*[:]*\s*', '', ingredient_text, flags=re.IGNORECASE)
         potential_ingredients = re.split(r'[;,]\s*(?![^()]*\))|\.\s+(?![^()]*\))|\s+and\s+(?![^()]*\))', ingredient_text)
         cleaned = []
         
         
         
         
         
         