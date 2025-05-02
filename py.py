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
         for ing in potential_ingredients:
            cleaned_ing = re.sub(r"^[^\w(]+|[^\w)]+$", "", ing.strip()).strip() # Allow starting '(' and ending ')'
            cleaned_ing = cleaned_ing.replace(' :', '') # Remove stray colons
            if cleaned_ing and len(cleaned_ing) > 1 and not cleaned_ing.isdigit():
               corrected_ing = cleaned_ing
               if sym_spell:
                  suggestions = sym_spell.lookup(cleaned_ing, Verbosity.CLOSEST, max_edit_distance=SYM_SPELL_EDIT_DISTANCE, include_unknown=True)
                  if suggestions and suggestions[0].distance < SYM_SPELL_EDIT_DISTANCE + 1: # Only correct if close enough
                     best_suggestion = suggestions[0].term
                     if not (any(char.isdigit() for char in cleaned_ing) and not any(char.isdigit() for char in best_suggestion)):
                        if best_suggestion != cleaned_ing:
                           logging.info(f"SymSpell corrected '{cleaned_ing}' to '{best_suggestion}' (distance {suggestions[0].distance})")
                        corrected_ing = best_suggestion
                     else:
                            logging.info(f"SymSpell skipped correction for '{cleaned_ing}' due to number mismatch.")
               if corrected_ing and corrected_ing not in cleaned: # Check non-empty after potential correction
                  cleaned.append(corrected_ing)
            
         logging.info(f"Found {len(cleaned)} potential ingredients.")
# --- LLM Analysis Function ---
model = None
tokenizer = None
pipe = None

def load_model():
    global model, tokenizer, pipe
    if pipe is not None and hasattr(pipe, 'model_name') and pipe.model_name == MODEL_NAME: # Check if correct model loaded 
       logging.info(f"Model {MODEL_NAME} already loaded.")
       return True 
    logging.info(f"Loading model: {MODEL_NAME}")
    try: 
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            
             logging.info("Tokenizer missing pad token, setting to EOS token.")
             tokenizer.pad_token = tokenizer.eos_token
             
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
           
        )
        pipe = pipeline(
           "text-generation",
           model=model,
           tokenizer=tokenizer,
           torch_dtype=torch.bfloat16,
           device_map="auto"
            )
         # Store model name for checking later
        pipe.model_name = MODEL_NAME
        logging.info("Model loaded successfully.")
        return True
    except Exception as e:
       logging.error(f"Error loading model {MODEL_NAME}: {e}", exc_info=True) # Log traceback
       model = None
       tokenizer = None
       pipe = None
       return False
def analyze_ingredients_llm(ingredients):
   global pipe
   if not pipe:
      logging.error("LLM Pipeline not available.")
      return {"error": "Model not loaded"}
   if not ingredients:
      logging.warning("No ingredients provided for analysis.")
      return {"analysis": []}
   ingredient_list_str = ', '.join(ingredients)
   logging.info(f"Analyzing ingredients: {ingredient_list_str}")
   messages = [
      {
            "role": "system",
            "content": """You are a highly specialized and meticulous food safety evaluation engine. Your SOLE function is to identify ingredients from a given list that pose **SUBSTANTIAL and WIDELY RECOGNIZED health risks** based on strong scientific consensus (e.g., reports from WHO, FDA, EFSA) or significant regulatory actions (e.g., bans, strict limits) in major regions.

**Your PRIMARY directive is accuracy and avoiding false positives.** Do NOT flag ingredients based on minor controversies, niche dietary theories, common allergies (unless the ingredient itself is inherently risky beyond being an allergen), or general 'unhealthiness' like sugar or salt in typical contexts.

**Apply a VERY HIGH THRESHOLD for flagging an ingredient.**
**Output Requirements:**
1.  Your response MUST be **ONLY** a valid JSON object. No introductory text, no explanations, no apologies, no closing remarks. Start with `{` and end with `}`.
2.  The JSON object MUST contain a single key: `"analysis"`.
3.  The value of `"analysis"` MUST be a list.
4.  **CRITICAL RULE:** If **NO** ingredients in the provided list meet the **HIGH THRESHOLD** for substantial and widely recognized risk, this list MUST be **EMPTY**. Example: `{"analysis": []}`. This is the expected output for safe or common ingredient lists.
5.  If and ONLY if one or more ingredients meet the high threshold, add an object for EACH such ingredient to the `"analysis"` list.
6.  Each ingredient object MUST contain EXACTLY these keys:
     *   `"ingredient"`: String. The name of the problematic ingredient. Include E-number if common (e.g., "Aspartame (E951)").
     *   `"safety_index"`: Integer (1-3). Represents the confidence and severity of the risk: 1 = High Confidence/Severe Risk (e.g., banned substance, strong warnings), 2 = Moderate Confidence/Risk (e.g., significant controversy with strong evidence, strict limits), 3 = Lower Confidence/Specific Risk (e.g., risky only for specific vulnerable groups based on strong evidence). **DO NOT use 4 or 5.**
      *   `"unsafe_for"`: Object with keys:
            *   `"age_groups"`: List of strings (e.g., ["children", "infants"]). Empty `[]` if not specific.
             *   `"diseases"`: List of strings (e.g., ["phenylketonuria"]). Empty `[]` if not specific.
              *   `"allergies"`: List of strings (e.g., ["sulfite sensitivity"]). **ONLY list if the risk goes BEYOND a typical allergic reaction.** Empty `[]` otherwise.
               *   `"severity"`: String: "critical" (for index 1), "moderate" (for index 2), or "low" (for index 3, representing risk mainly to specific groups).
**Examples of what NOT to flag (return `{"analysis": []}`):**
*   Lists containing only: Sugar, Salt, Flour, Water, Vegetable Oil, Citric Acid, Ascorbic Acid, Natural Flavors, Spices, Vinegar, Baking Soda, Yeast, Lecithin (soy/sunflower), Milk, Eggs, Wheat, etc.
*   Common allergens like Soy, Nuts, Dairy, Gluten when listed normally.
*   Standard vitamins and minerals.
*   Commonly accepted preservatives like Sodium Benzoate, Potassium Sorbate unless there's a very specific, high-risk context.
*   Certain Artificial Colors with strong links to health issues (e.g., Red 3, Yellow 5 in some contexts)
*   Aspartame (mainly for PKU risk, flag as lower severity unless specifically asked otherwise)
*   Olestra
*   Specific additives banned or heavily restricted in major regions.
**FINAL INSTRUCTION:** Before outputting, double-check: Did any ingredient CLEARLY cross the high threshold for substantial, widely recognized risk? If not, your entire output MUST BE EXACTLY `{"analysis": []}`."""
   },
        {
           
            "role": "user",
            "content": f"Analyze the following ingredient list according to the strict rules and high threshold defined in the system prompt. Return ONLY the required JSON object.\n\nIngredients: {ingredient_list_str}"
        }
        ]
   try:
        # Used the tokenizer's chat template (Mistral Instruct uses [INST]...[/INST])
        # Important: add_generation_prompt=True tells the template to add the prompt for the assistant's turn
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   except Exception as e:
        logging.error(f"Could not apply chat template: {e}. Model might not generate correctly.")
        # Fallback might be needed but is model-specific and less reliable
        return {"error": f"Failed to create prompt using chat template: {e}"}
     
   logging.info("Sending prompt to LLM.")
   try:
        outputs = pipe(
            prompt,
            max_new_tokens=1024,
            do_sample=True, # Keep sampling for potentially better/more natural analysis
            temperature=0.5, # Slightly lower temp might help structure adherence
            top_p=0.9,
            eos_token_id=pipe.tokenizer.eos_token_id,
             pad_token_id=pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id is not None else pipe.tokenizer.eos_token_id
        )
        response_text = outputs[0]['generated_text']
        prompt_end_marker = "[/INST]"
        prompt_end_index = response_text.rfind(prompt_end_marker)
        if prompt_end_index != -1:
            generated_part = response_text[prompt_end_index + len(prompt_end_marker):].strip()
        else:
           if response_text.startswith(prompt):
               generated_part = response_text[len(prompt):].strip()












































