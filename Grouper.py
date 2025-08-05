import pandas as pd
import re
import os
import warnings
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")

# --- Preprocessing ---
def preprocess(text):
    if pd.isnull(text):
        return ""

    text = text.lower()

    # Replace all Unicode space-like characters with a normal space
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

    # Remove pound/hash symbols
    text = text.replace('#', '')
    
    # Words and phrases to remove
    REMOVE_TERMS = [
        r'\bu\.?\s*s\.?\s*a\.?\b',  # USA, U.S.A., etc.
        r'\bverbatim\b',
        r'\[?\s*no additional locality data on sheet\s*\]?',
        r'\[?\s*no additional data\s*\]?',
        r'\[?\s*locality not indicated\s*\]?',
        r'\[?\s*not readable\s*\]?',
        r'\[?\s*illegible\s*\]?',
        r'\[?\s*none\s*\]?',
        r'\[?\s*unspecified\s*\]?',
        r'\[?\s*no location data on label\s*\]?',
        r'\bno locality\b',
        r'\bnone listed\b',
        r'\bno further locality\b',
        r'\bno location\b',
        r'\b(?:about|ca\.?)\s+',  # Approximate qualifiers
        r'\(air\)',  # ← this line removes (air)
        r'(^\s*(coll\.?|collected|found)\b[\s,:-]*|\b(collected|found)\s+(from|in|at|on|along|near)\b)'
    ]

    for pattern in REMOVE_TERMS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)



    # --- Remove repeated locality prefix before semicolon if repeated later "Oklahoma City; near county line on W 10th street, Oklahoma City"---
    if ";" in text:
        prefix, rest = text.split(";", 1)
        prefix = prefix.strip()
        if prefix and prefix in rest:
            text = rest.strip()

    # --- Normalize possessives ---
    text = re.sub(r"\b(\w+)'s\b", r"\1s", text)

    # Normalize all variants of "mi", "mi.", " mi " to " miles "
    text = re.sub(r'\bmis\.?\b', ' miles ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmi\.?\b', ' miles ', text, flags=re.IGNORECASE)

    # --- Normalize "km" to " kilometers " and "'" to " feet "
    text = re.sub(r'\bkm\.?\b', ' kilometers ', text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*['’]", r"\1 feet", text)

     # change " m " to meters if preceding number is above 20, miles if below 20
    def convert_m_unit(match):
        num = float(match.group(1))
        unit = "meters" if num > 20 else "miles"
        return f"{int(num) if num.is_integer() else num} {unit}"
    
    # Handles glued and spaced versions like "100m" and "100 m"
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*m\b', convert_m_unit, text, flags=re.IGNORECASE)

    # Insert a space between numbers and units if stuck together (e.g., "5miles" → "5 miles")
    text = re.sub(r'(\d+(?:\.\d+)?)(?=\s*?(miles|mile|km|kilometers|kilometer|mi|ft|feet))', r'\1 ', text, flags=re.IGNORECASE)

    # --- Force singular "mile" to plural "miles" ---
    text = re.sub(r'\bmile\b', 'miles', text, flags=re.IGNORECASE)

    # --- Normalize all forms of 'state highway' ---
    text = re.sub(r'\bstate\s+highway\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstate\s+hiway\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstate\s+hwy\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstate\s+hwy\.?\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsh\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bst\.?\s*hwy\.?\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bus\s+hwy\.?\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstate\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhy\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    # --- Fallback catch-all for unnumbered state highways ---
    text = re.sub(r'\b(state\s+hwy|state\s+highway|sh|st\.?\s*hwy)\b', 'highway', text, flags=re.IGNORECASE)

    # Normalize "TX 10", "Tex 10", "Texas 10" → "highway 10"
    text = re.sub(r'\btex(?:as)?\.?\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    # Normalize "OK 10", "Okla 10", "Oklahoma 10" → "highway 10"
    text = re.sub(r'\bok(?:la)?(?:homa)?\.?\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)

    # --- Normalize specific U.S. Highway variants to "highway <number>" ---
    text = re.sub(r'\bu\.?\s*s\.?\s+(highway|hwy)\s+(\d+)\b', r'highway \2', text, flags=re.IGNORECASE)  # handles "U. S. Hwy"
    text = re.sub(r'\bus\s+highway\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bus\s+hwy\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bus\.?\s*hwy\.?\s*(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bush\s*(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    # Normalize bare US highway numbers like "US 10", "U.S. 10", "U. S. 10"
    text = re.sub(r'\bu\.?\s*s\.?\s*(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bus\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    # Catch generic references to U.S. highways without numbers ---
    text = re.sub(r'\b(u\.?\s*s\.?|us|ush)\s+(highway|hwy)\b', 'highway', text, flags=re.IGNORECASE)
    # Normalize Interstate variants like "I-40", "I 40", "I. 40", "Interstate 40" → "highway 40"
    text = re.sub(r'\binterstate\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi[\.\-\s]?(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    # Normalize FM to farm-to-market
    text = re.sub(r'\bf[\.\s]*m[\.\s]*(road)?[\s\.]*#?(\d+)\b',r'farm-to-market \2', text, flags=re.IGNORECASE)

    # --- Split glued compass direction + "of" (e.g., " nof," → " n of,") ---
    text = re.sub(r'(?<=\s)([nswe]{1,3})of(?=[\s\.,:;!?])', r'\1 of', text, flags=re.IGNORECASE)

    # --- Normalize compound compass directions ---
    text = re.sub(r'(?<!\w)[nN][\.\s]?[eE](?!\w)', 'northeast', text)
    text = re.sub(r'(?<!\w)[nN][\.\s]?[wW](?!\w)', 'northwest', text)
    text = re.sub(r'(?<!\w)[sS][\.\s]?[eE](?!\w)', 'southeast', text)
    text = re.sub(r'(?<!\w)[sS][\.\s]?[wW](?!\w)', 'southwest', text)

    # --- Normalize single-letter compass directions ---
    text = re.sub(r'(?<![\w\'])\bn[\.\s]*(?=\W|$)', 'north ', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![\w\'])\bs[\.\s]*(?=\W|$)', 'south ', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![\w\'])\be[\.\s]*(?=\W|$)', 'east ', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![\w\'])\bw[\.\s]*(?=\W|$)', 'west ', text, flags=re.IGNORECASE)

    # Join separated compass directions with optional periods
    text = re.sub(r'\bnorth[\.\s]+east\b', 'northeast', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnorth[\.\s]+west\b', 'northwest', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsouth[\.\s]+east\b', 'southeast', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsouth[\.\s]+west\b', 'southwest', text, flags=re.IGNORECASE)

    # Normalize space-separated compound directions like "west southwest" → "west-southwest"
    compound_directions = {
        r'\bnorth\s+northeast\b': 'north-northeast',
        r'\bnorth\s+northwest\b': 'north-northwest',
        r'\bsouth\s+southeast\b': 'south-southeast',
        r'\bsouth\s+southwest\b': 'south-southwest',
        r'\beast\s+northeast\b': 'east-northeast',
        r'\beast\s+southeast\b': 'east-southeast',
        r'\bwest\s+northwest\b': 'west-northwest',
        r'\bwest\s+southwest\b': 'west-southwest',
    }
    
    for pattern, replacement in compound_directions.items():
        text = re.sub(pattern, replacement, text)

    # Normalize compass abbreviations (e.g., NNE → north-northeast)
    abbr_map = {
        'nne': 'north-northeast',
        'nnw': 'north-northwest',
        'ene': 'east-northeast',
        'ese': 'east-southeast',
        'sse': 'south-southeast',
        'ssw': 'south-southwest',
        'wsw': 'west-southwest',
        'wnw': 'west-northwest'
    }

    for abbr, full in abbr_map.items():
        text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)

    # Fuzzy-correct direction typos after normalization (Commented out cause it is more trouble than its worth.)
    #DIRECTION_TERMS = [
    #    'north', 'south', 'east', 'west',
    #    'northeast', 'northwest', 'southeast', 'southwest',
    #    'north-northeast', 'north-northwest',
    #    'south-southeast', 'south-southwest',
    #    'east-northeast', 'east-southeast',
    #    'west-northwest', 'west-southwest'
    #]
    #FUZZY_DIR_THRESHOLD = 90
#
    #
    #def correct_direction_typos(text):
    #    tokens = text.split()
    #    corrected = []
    #    for tok in tokens:
    #        match, score, _ = process.extractOne(tok, DIRECTION_TERMS)
    #        if score >= FUZZY_DIR_THRESHOLD:
    #            corrected.append(match)
    #        else:
    #            corrected.append(tok)
    #    return ' '.join(corrected)
    #
    #text = correct_direction_typos(text)

    # --- Common abbreviation replacements ---
    
    ABBREVIATIONS = {
        r'\bjct\b': 'junction',
        r'\bint\b': 'intersection',
        r'\b(\d{1,4}(?:st|nd|rd|th)?)\s+st\.?\b': r'\1 street',  # "3rd st" → "3rd street"
        r'\bst\.?\b': 'street',
        r'\bcir\.?\b': 'circle',
        r'\bave\.?\b': 'avenue',
        r'\brt\.?\b': 'route',
        r'\bdr\.?\b': 'drive',
        r'\bblvd\.?\b': 'boulevard',
        r'\bcr\s*(\d+)\b': r'county road \1', # "cr 123" → "county road 123"
        r'\brd\.?\b': 'road',
        r'\bhwy\.?\b': 'highway',
        r'\bmt\.?\b': 'mountain',
        r'\bmtn\.?\b': 'mountain',
        r'\bmts\.?\b': 'mountains',
        r'\bmtns\.?\b': 'mountains',
        r'\br[\.\-\s]?r[\.\-]?(?=\W|$)': 'railroad',  # Matches "rr", "r.r", "r-r", "r r", etc. at word end → "railroad"
        r'\br\.(?=\W|$)': 'river',    # Matches "r." or "riv." → "river"
        r'\briv\.(?=\W|$)': 'river',
        r'\bmi\b': 'miles',
        r'\bft\.?\b': 'fort',
        r'\bcp\.?\b': 'camp',
        r'\bbldg\.?\b': 'building',
        r'\s+x\s+': ' ',   # Clean "x" as a separator like "5 x 10" → "5 10"
        r'&': 'and',  # Symbol replacements
        r'\+': 'and',
        r'\bok\b': 'oklahoma',
        r'\bokla\b': 'oklahoma',
        r'\bOKC\b': 'oklahoma city',
        r'\btx\b': 'texas',
        r'\bTex\b': 'texas',
        r'\bar\b': 'arkansas',
        r'\bark\b': 'arkansas',
        r'\bwma\.?\b': 'wildlife management area',
        r'\bnra\.?\b': 'national recreation area',
        r'\bco\.\b': 'county',
        r'\bco\b': 'county'
    }
    
    # Apply all abbreviation replacements
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # --- Convert spelled-out ordinals like "tenth" to "10th" ---
    ordinal_words = {
        'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
        'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
        'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
        'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th',
        'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
        'nineteenth': '19th', 'twentieth': '20th'
    }

    for word, ordinal in ordinal_words.items():
        text = re.sub(rf'\b{word}\b', ordinal, text, flags=re.IGNORECASE)

    # --- Convert spelled-out numbers before miles to digits ---
    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteen': '19', 'twenty': '20'
    }


    # --- Replace spelled-out numbers with digits only when followed by distance units or directional words using a lookahead pattern ---
    directions = [
            'north', 'south', 'east', 'west',
            'northeast', 'northwest', 'southeast', 'southwest',
            'north-northeast', 'north-northwest',
            'south-southeast', 'south-southwest',
            'east-northeast', 'east-southeast',
            'west-northwest', 'west-southwest'
    ]
    dir_pattern = '|'.join(directions)
    
    units_pattern = r'miles?|kilometers?|km|mi'
    directions_pattern = r'north|south|east|west|northeast|northwest|southeast|southwest'
    
    for word, digit in number_words.items():
        text = re.sub(
            rf'\b{word}\b(?=\s*({units_pattern}|{directions_pattern})\b)',
            digit,
            text,
            flags=re.IGNORECASE
        )


    # --- Normalize spelled-out fractions like "one-half" ---
    fraction_words = {
        r'\bone[\s-]+half\b': '0.5',
        r'\bone[\s-]+third\b': '0.33',
        r'\btwo[\s-]+thirds\b': '0.66',
        r'\bone[\s-]+fourth\b': '0.25',
        r'\bthree[\s-]+fourths\b': '0.75',
        r'\bthree[\s-]+quarters?\b': '0.75',  # ← handles both quarter and quarters
        r'\bone[\s-]+quarter\b': '0.25',
    }

    for pattern, replacement in fraction_words.items():
        text = re.sub(pattern, replacement, text)
    
    # --- Mixed ASCII fractions ---
    text = re.sub(r'(\d+)\s+1/2\b', lambda m: str(float(m.group(1)) + 0.5), text)
    text = re.sub(r'(\d+)\s+1/4\b', lambda m: str(float(m.group(1)) + 0.25), text)
    text = re.sub(r'(\d+)\s+3/4\b', lambda m: str(float(m.group(1)) + 0.75), text)
    text = re.sub(r'(\d+)\s+1/3\b', lambda m: str(float(m.group(1)) + 0.33), text)
    text = re.sub(r'(\d+)\s+2/3\b', lambda m: str(float(m.group(1)) + 0.66), text)
    text = re.sub(r'(\d+)\s+1/8\b', lambda m: str(float(m.group(1)) + 0.125), text)


    # --- Mixed Unicode fractions ---
    text = re.sub(r'(\d+)\s*½', lambda m: str(float(m.group(1)) + 0.5), text)
    text = re.sub(r'(\d+)\s*¼', lambda m: str(float(m.group(1)) + 0.25), text)
    text = re.sub(r'(\d+)\s*¾', lambda m: str(float(m.group(1)) + 0.75), text)
    text = re.sub(r'(\d+)\s*⅓', lambda m: str(float(m.group(1)) + 0.33), text)
    text = re.sub(r'(\d+)\s*⅔', lambda m: str(float(m.group(1)) + 0.66), text)
    text = re.sub(r'(\d+)\s*⅛', lambda m: str(float(m.group(1)) + 0.125), text)

    # --- Standalone fractions ---
    text = re.sub(r'\b1/2\b', '0.5', text)
    text = re.sub(r'\b1/4\b', '0.25', text)
    text = re.sub(r'\b3/4\b', '0.75', text)
    text = re.sub(r'\b1/3\b', '0.33', text)
    text = re.sub(r'\b2/3\b', '0.66', text)
    text = re.sub(r'\b1/8\b', '0.125', text)
    text = re.sub(r'\b½\b', '0.5', text)
    text = re.sub(r'\b¼\b', '0.25', text)
    text = re.sub(r'\b¾\b', '0.75', text)
    text = re.sub(r'\b⅓\b', '0.33', text)
    text = re.sub(r'\b⅔\b', '0.66', text)
    text = re.sub(r'\b⅛\b', '0.125', text)

    # Normalize leading decimals with zeros (".5" to "0.5") if preceded by whitespace or line start
    text = re.sub(r'(^|\s)\.(\d+)', r'\g<1>0.\2', text)

    # Normalize numbers like "1." to "1" (when not part of a decimal)
    text = re.sub(r'\b(\d+)\.(?!\d)', r'\1', text)

    # Remove "of a" between number and miles/kilometers (e.g., "0.75 of a miles" → "0.75 miles")
    text = re.sub(r'(\d+(?:\.\d+)?)(\s+)of\s+a\s+(miles?|mile|kilometers?|km)\b', r'\1 \3', text, flags=re.IGNORECASE)

    # Strip .0 from numbers like 5.0 miles to 5 miles
    text = re.sub(r'(\d+)\.0\b', r'\1', text)

    # Normalize numbers directly before compass directions with no unit to miles
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(north|south|east|west|northeast|northwest|southeast|southwest)\b',
        r'\1 miles \2',
        text,
        flags=re.IGNORECASE
    )

    # --- Normalize patterns like "6mi.E." or "5kmW" → "6 miles east" ---
    text = re.sub(
        r'(\d+(\.\d+)?)(?:\s*)mi\.?\s*([nsew])\b',
        lambda m: f"{m.group(1)} miles {'north' if m.group(3).lower() == 'n' else 'south' if m.group(3).lower() == 's' else 'east' if m.group(3).lower() == 'e' else 'west'}",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(\d+(\.\d+)?)(?:\s*)km\.?\s*([nsew])\b',
        lambda m: f"{m.group(1)} kilometers {'north' if m.group(3).lower() == 'n' else 'south' if m.group(3).lower() == 's' else 'east' if m.group(3).lower() == 'e' else 'west'}",
        text,
        flags=re.IGNORECASE
    )

    # Remove all punctuation except for periods used in decimal numbers
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)  # remove periods not part of decimal numbers

    # Preserve hyphens within known compound directions before stripping punctuation
    DIRECTION_COMPOUNDS = [
        'north-northeast', 'north-northwest', 'south-southeast', 'south-southwest',
        'east-northeast', 'east-southeast', 'west-northwest', 'west-southwest'
    ]
    
    for compound in DIRECTION_COMPOUNDS:
        text = text.replace(compound, compound.replace('-', '___'))  # temp protect hyphens
    
    # Now remove unwanted punctuation
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    
    # Restore hyphens
    text = text.replace('___', '-')

    # --- Normalize whitespace ---
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# --- Extract distances and directions ---
def extract_distance_direction(text):
    """
    Extracts (distance, direction, unit) tuples from text.
    Primary extraction expects normalized input like:
        '125 meters east', '0.35 miles south', '40 feet west'

    Also handles fallback patterns where distance and direction appear out of order,
    e.g., 'northwest of town 9 miles' or '9 miles northwest of town'
    """

    if pd.isnull(text):
        return []

    # --- Main extraction pattern ---
    # Match number + optional unit + direction in order
    # Example matches: '5 miles north', '3.5 kilometers southwest'
    pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*'                           # Number (with optional decimal)
        r'(miles|kilometers|meters|feet)?[\s,]*'        # Optional unit
        r'(north|south|east|west|'                      # Direction (simple and compound)
        r'northeast|northwest|southeast|southwest)\b',
        flags=re.IGNORECASE
    )

    matches = pattern.findall(text)
    results = []

    # --- Format and normalize the matched results ---
    for number, unit, direction in matches:
        num = float(number)
        if num.is_integer():
            num = int(num)  # Convert to int if it's a whole number (e.g., 5.0 → 5)
        unit = unit.lower() if unit else ''  # Normalize unit (e.g., 'Miles' → 'miles')
        results.append((str(num), direction.lower(), unit))  # Lowercase for consistency

    # --- If matches found, return them sorted by direction and then distance ---
    if results:
        results.sort(key=lambda x: (x[1], float(x[0])))
        return results

    # --- Fallback logic (if no standard pattern was matched) ---
    # Try to detect *exactly one* number+unit and *exactly one* direction in any order

    # Find first occurrence of a number + unit (e.g., "9 miles")
    fallback_number = re.search(
        r'\b(\d+(?:\.\d+)?)\s*(miles|kilometers|meters|feet)\b',
        text,
        flags=re.IGNORECASE
    )

    # Find first occurrence of a direction (e.g., "northwest")
    fallback_direction = re.search(
        r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b',
        text,
        flags=re.IGNORECASE
    )

    # If both are found, assume this is a valid out-of-order distance-direction pair
    if fallback_number and fallback_direction:
        number = float(fallback_number.group(1))
        if number.is_integer():
            number = int(number)
        unit = fallback_number.group(2).lower()
        direction = fallback_direction.group(1).lower()
        return [(str(number), direction, unit)]

    # If nothing found, return empty list
    return []


# --- Prompt for file ---
csv_path = input("Enter path to CSV or TSV file: ").strip()
if not os.path.isfile(csv_path):
    print("File not found.")
    exit()

ext = os.path.splitext(csv_path)[1].lower()
if ext == '.tsv':
    sep = '\t'
elif ext == '.csv':
    sep = ','
else:
    print("Unsupported file type. Please provide a .csv or .tsv file.")
    exit()

df = pd.read_csv(csv_path, sep=sep)

grouping_field = "bels_location_id"

if 'locality' not in df.columns or grouping_field not in df.columns:
    print(f"CSV must contain 'locality' and '{grouping_field}' columns.")
    exit()

# --- Unique localities by grouping field ---
grouped = df.drop_duplicates(subset=grouping_field).copy()
grouped = grouped.reset_index(drop=True)
grouped['normalized_locality'] = grouped['locality'].apply(preprocess)
grouped['distance_direction'] = grouped['normalized_locality'].str.replace('*', '', regex=False).apply(extract_distance_direction)

# --- TF-IDF setup ---
important_phrases = [
    'north', 'south', 'east', 'west',
    'northeast', 'northwest', 'southeast', 'southwest'
]

def custom_tokenizer(text):
    # Remove all punctuation except periods in numbers (e.g., 3.5)
    text = re.sub(r'[^\w\s.]', '', text)
    # Tokenize on words and decimal numbers
    return re.findall(r'\b\d+(?:\.\d+)?\b|\b\w+\b', text)

# Custom stop words tuned for geographic data
custom_stop_words = [
    'the', 'a', 'an', 'in', 'at', 'on', 'for', 'by', 'with', 'and', 'of', 'or', 'but', 'from', 'between', 'along',
    'texas', 'oklahoma', 
    'junction', 'intersection',
    'sandy', 'clay', 'soil', 'loam', 'sandy', 'rocky', 'silt', 'bed', 'bank'
    'x'
]

vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    lowercase=False,
    stop_words=custom_stop_words
)

# --- Initial TF-IDF matrix on pre-alias normalized locality ---
X = vectorizer.fit_transform(grouped['normalized_locality'])
vocab = vectorizer.vocabulary_

token_freq = {token: X[:, idx].nnz for token, idx in vocab.items()}  # document frequency
vocab_keys = list(vocab.keys())

protected_tokens = set([
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "northern", "southern", "eastern", "western", "central",
    "1st", "2nd", "3rd", "4th", "5th", "6th",
    "7th", "8th", "9th", "10th", "11th", "12th",
    "13th", "14th", "15th", "16th", "17th", "18th",
    "19th", "20th"
])
merged = {}

def dynamic_threshold(token1, token2, base_threshold=75, max_threshold=90):
    avg_len = (len(token1) + len(token2)) / 2
    if avg_len <= 5:
        return base_threshold
    elif avg_len >= 15:
        return max_threshold
    else:
        return base_threshold + ((avg_len - 5) / 10) * (max_threshold - base_threshold)

township_pattern = r'^[trs]\d{1,3}[nsew]?$'

# --- Fuzzy token aliasing ---
for i in range(len(vocab_keys)):
    token_i = vocab_keys[i]

    # Skip token_i if it's a digit, protected, or already merged
    if (
        token_i.replace(".", "").isdigit()
        or token_i in protected_tokens
        or token_i in merged
    ):
        continue

    if re.fullmatch(r'\d{1,4}(st|nd|rd|th)', token_i):
        continue
    if re.fullmatch(township_pattern, token_i):
        continue

    for j in range(i + 1, len(vocab_keys)):
        token_j = vocab_keys[j]

        # Skip token_j if it's a digit or already merged — but NOT if it's protected
        if (
            token_j.replace(".", "").isdigit()
            or token_j in merged
        ):
            continue

        if re.fullmatch(r'\d{1,4}(st|nd|rd|th)', token_j):
            continue
        if re.fullmatch(township_pattern, token_j):
            continue

        # Length similarity check
        len_i = len(token_i)
        len_j = len(token_j)
        if min(len_i, len_j) / max(len_i, len_j) < 0.8:
            continue

        score = fuzz.ratio(token_i, token_j)
        threshold = dynamic_threshold(token_i, token_j)

        if score >= threshold:
            freq_i = token_freq.get(token_i, 0)
            freq_j = token_freq.get(token_j, 0)

            if freq_i < 5 and freq_j >= 5:
                canonical, other = token_j, token_i
            elif freq_j < 5 and freq_i >= 5:
                canonical, other = token_i, token_j
            else:
                continue

            # Prevent overriding protected tokens as aliases
            if other in protected_tokens or other in merged:
                continue

            print(f"Aliasing '{other}' ({token_freq.get(other, 0)}) to '{canonical}' ({token_freq.get(canonical, 0)}) (score {score:.2f} ≥ {threshold:.2f})")
            merged[other] = canonical


# --- Apply alias substitutions into normalized locality ---
def apply_aliases(text, alias_map):
    tokens = text.split()
    result = []

    for tok in tokens:
        if tok in alias_map:
            result.append('*' + alias_map[tok])
        else:
            result.append(tok)

    return ' '.join(result)


grouped['normalized_locality'] = grouped['normalized_locality'].apply(lambda t: apply_aliases(t, merged))

# --- Rebuild TF-IDF matrix on alias-applied text ---
X = vectorizer.fit_transform(grouped['normalized_locality'])
vocab = vectorizer.vocabulary_

# --- Re-weight directional and numeric tokens ---
for token, idx in vocab.items():
    if token in important_phrases or re.fullmatch(r'\d+(\.\d+)?', token):
        X[:, idx] *= 1.10

# --- Cosine similarity ---
similarity = cosine_similarity(X)

threshold = 0.85

suggested_ids = [-1] * len(grouped)
group_counter = 1

for i in range(len(grouped)):
    if suggested_ids[i] != -1:
        continue
    suggested_ids[i] = group_counter
    for j in range(i + 1, len(grouped)):
        if suggested_ids[j] == -1 and similarity[i, j] >= threshold:
            suggested_ids[j] = group_counter
    group_counter += 1

grouped['Suggested_ID'] = suggested_ids
grouped['Grouper_ID'] = grouped['Suggested_ID'].astype(str)

# Now safe to group by Grouper_ID
group_members = (
    grouped
    .drop(columns=['Grouper_ID'])
    .groupby(grouped['Grouper_ID'])
    .apply(lambda df: df.index.tolist())
    .to_dict()
)

grouped['Suggested_ID'] = suggested_ids

# --- Compute confidence ---
confidence_scores = []

for idx in range(len(grouped)):
    group_id = grouped.iloc[idx]['Grouper_ID']
    members = group_members[group_id]

    # If the group has only one member, confidence is 1
    if len(members) == 1:
        confidence_scores.append(1.0)
        continue

    # Calculate average similarity to other members of the group
    sims = [
        similarity[idx, other_idx]
        for other_idx in members
        if other_idx != idx
    ]
    confidence = sum(sims) / len(sims)
    confidence_scores.append(confidence)

# Assign after loop
grouped['Confidence'] = [round(c * 100, 1) for c in confidence_scores]




# --- Validate suggested groups by distance/direction ---
for group_id in grouped['Suggested_ID'].unique():
    members = grouped[grouped['Suggested_ID'] == group_id]

    # get the list of extracted distance/direction for each member
    distance_lists = members['distance_direction']

    # if all members either have no distance/direction or all have the exact same, do not split
    if len(distance_lists.apply(tuple).unique()) <= 1:
        continue

    # if no members have any distance/direction data, do not split
    signatures = distance_lists.apply(tuple).unique()
    if all(len(lst) == 0 for lst in distance_lists):
        continue
    for idx, sig in enumerate(signatures, start=1):  # start=1 to begin with .1
        suffix = f".{idx}"
        to_update = members[distance_lists.apply(tuple) == sig].index
        grouped.loc[to_update, 'Grouper_ID'] = f"{group_id}{suffix}"


# If the original locality is blank, null, or matches known placeholders, set Grouper_ID to 0
null_strings = [
    'unknown',
    'no locality',
    '[no locality]',
    '[no additional data]',
    '[no additional locality data on sheet]',
    '[locality not indicated]',
    '[unspecified]',
    '[No location data on label.]',
    '[ Not readable ]',
    '[none]',
    'none listed',
    'no further locality',
    'no location'
]
grouped.loc[
    grouped['locality'].isnull()
    | (grouped['locality'].str.strip() == '')
    | (grouped['locality'].str.strip().str.lower().isin([s.lower() for s in null_strings]))
    ,
    'Grouper_ID'
] = '0'


# --- Reorder singletons based on similarity to closest larger group ---
from collections import defaultdict
import numpy as np
import time

start_time = time.time()
print("Identifying singleton placements...")

group_id_to_indices = defaultdict(list)
for idx, gid in enumerate(grouped['Grouper_ID']):
    group_id_to_indices[gid].append(idx)

# Count how many rows belong to each base group (before .01, .02 suffixes)
base_group_counts = grouped['Grouper_ID'].apply(lambda x: str(x).split('.')[0]).value_counts().to_dict()

# Filter singleton_ids (that are not directionally split)
singleton_ids = []
for gid, idxs in group_id_to_indices.items():
    if len(idxs) != 1:
        continue  # Not a singleton

    match = re.match(r'^(\d+)\.\d+$', str(gid))
    if match:
        base_id = match.group(1)
        if base_group_counts.get(base_id, 0) > 1:
            continue  # It's a directional split — skip it

    singleton_ids.append(gid)

# Now define non-singleton_ids AFTER filtering valid singleton_ids
non_singleton_ids = [gid for gid in group_id_to_indices if gid not in singleton_ids]

MIN_SINGLETON_SIMILARITY = 0.80
singleton_inserts = {}
for singleton_id in singleton_ids:
    singleton_idx = group_id_to_indices[singleton_id][0]
    best_score = -1
    best_match_id = None

    for gid in non_singleton_ids:
        group_idxs = group_id_to_indices[gid]
        max_sim = max(similarity[singleton_idx, other_idx] for other_idx in group_idxs)

        if max_sim > best_score:
            best_score = max_sim
            best_match_id = gid

    if best_score >= MIN_SINGLETON_SIMILARITY:
        singleton_inserts[singleton_id] = best_match_id

print(f"Placed {len(singleton_inserts)} of {len(singleton_ids)} singleton groups based on similarity ≥ {MIN_SINGLETON_SIMILARITY}.")
print(f"Completed in {time.time() - start_time:.2f} seconds.")

# --- Convert extracted distance_direction tuples to readable string ---
grouped['Distance_Direction'] = grouped['distance_direction'].apply(
    lambda lst: '; '.join([f"{d} {u} {dir}" if u else f"{d} {dir}" for d, dir, u in lst]) if lst else ''
)

# --- Merge back ---
output_df = df.merge(
    grouped[[grouping_field, 'Grouper_ID', 'normalized_locality']],
    on=grouping_field,
    how='left'
)

# --- Export ---
columns_to_export = [
    'catalogNumber', 'institutionCode', 'collectionCode', 'county',
    'locality', 'bels_location_id', 'Grouper_ID', 'normalized_locality', 'Confidence',
    'Distance_Direction'
]

# For consistent columns, protect against missing
columns_to_export = [col for col in columns_to_export if col in grouped.columns]

import re

def sort_key(val):
    match = re.match(r'^(\d+)(?:\.(\d+))?$', str(val))
    if match:
        num = int(match.group(1))
        suffix = int(match.group(2)) if match.group(2) else -1  # -1 puts base group first
        return (num, suffix)
    else:
        return (float('inf'), float('inf'))

export_df = grouped[columns_to_export].drop_duplicates()
def grouper_sort_key(gid):
    if gid in singleton_inserts:
        anchor_gid = singleton_inserts[gid]
        anchor_tuple = sort_key(anchor_gid)
        singleton_weight = 0.5  # Ensure it's placed *after* the anchor
        return (anchor_tuple[0], anchor_tuple[1] + singleton_weight)
    else:
        return sort_key(gid)

export_df = export_df.sort_values(by='Grouper_ID', key=lambda col: col.map(grouper_sort_key))


output_file = os.path.splitext(csv_path)[0] + '-key.csv'
export_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Exported with suggested groups to: {output_file}")