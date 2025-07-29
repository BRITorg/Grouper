import pandas as pd
import re
import os
import warnings
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")

# --- Preprocessing ---
def preprocess(text):
    if pd.isnull(text):
        return ""

    text = text.lower()

    # Replace all Unicode space-like characters with a normal space
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

    # --- Remove repeated locality prefix before semicolon if repeated later ---
    if ";" in text:
        prefix, rest = text.split(";", 1)
        prefix = prefix.strip()
        if prefix and prefix in rest:
            text = rest.strip()

    # --- Normalize possessives ---
    text = re.sub(r"\b(\w+)'s\b", r"\1s", text)


    # Normalize all variants of "mi", "mi.", " mi " to " miles "
    text = re.sub(r'\bmi\.?\b', ' miles ', text, flags=re.IGNORECASE)
    # --- Normalize "km" to " kilometers "
    text = re.sub(r'\bkm\.?\b', ' kilometers ', text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*['’]", r"\1 feet", text)
    # Normalize feet and meters
    text = re.sub(r'\s+', ' ', text)
    # change " m " to meters if preceding number is above 20, miles if below 20
    def convert_m_unit(match):
        num = float(match.group(1))
        unit = "meters" if num > 20 else "miles"
        return f"{int(num) if num.is_integer() else num} {unit}"
    
    # Handles glued and spaced versions like "100m" and "100 m"
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*m\b', convert_m_unit, text, flags=re.IGNORECASE)


    # Insert a space between numbers and units if stuck together (e.g., "5miles" → "5 miles")
    text = re.sub(r'(\d+(?:\.\d+)?)(?=\s*?(miles|mile|km|kilometers|kilometer|mi|ft|feet))', r'\1 ', text, flags=re.IGNORECASE)


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
    # --- Catch generic references to U.S. highways without numbers ---
    text = re.sub(r'\b(u\.?\s*s\.?|us|ush)\s+(highway|hwy)\b', 'highway', text, flags=re.IGNORECASE)
    # Normalize Interstate variants like "I-40", "I 40", "I. 40", "Interstate 40" → "highway 40"
    text = re.sub(r'\binterstate\s+(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi[\.\-\s]?(\d+)\b', r'highway \1', text, flags=re.IGNORECASE)

    # Remove "U.S.A.", "USA", "U. S. A.", etc.
    text = re.sub(r'\bu\.?\s*s\.?\s*a\.?\b', '', text, flags=re.IGNORECASE)

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

    # --- Common abbreviation replacements ---
    text = re.sub(r'\bjct\b', 'junction', text, flags=re.IGNORECASE)
    text = re.sub(r'\bst\.?\b', 'street', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcir\.?\b', 'circle', text, flags=re.IGNORECASE)
    text = re.sub(r'\bave\.?\b', 'avenue', text, flags=re.IGNORECASE)
    text = re.sub(r'\brt\.?\b', 'route', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdr\.?\b', 'drive', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d{1,4}(?:st|nd|rd|th)?)\s+st\.?\b', r'\1 street', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcr\s*(\d+)\b', r'county road \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\brd\.?\b', 'road', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhwy\.?\b', 'highway', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmt\.?\b', 'mountain', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmtn\.?\b', 'mountain', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmts\.?\b', 'mountains', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmtns\.?\b', 'mountains', text, flags=re.IGNORECASE)
    text = re.sub(r'\br[\.\-\s]?r[\.\-]?(?=\W|$)', 'railroad', text, flags=re.IGNORECASE)
    text = re.sub(r'\br\.(?=\W|$)', 'river', text, flags=re.IGNORECASE)
    text = re.sub(r'\briv\.(?=\W|$)', 'river', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmi\b', 'miles', text, flags=re.IGNORECASE)
    text = re.sub(r'\bft\.?\b', 'fort', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcp\.?\b', 'camp', text, flags=re.IGNORECASE)
    text = re.sub(r'&', 'and', text)
    text = re.sub(r'\+', 'and', text)
    text = re.sub(r'\bok\b', 'oklahoma', text, flags=re.IGNORECASE)
    text = re.sub(r'\bokla\b', 'oklahoma', text, flags=re.IGNORECASE)
    text = re.sub(r'\btx\b', 'texas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTex\b', 'texas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bar\b', 'arkansas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bark\b', 'arkansas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcoll\.?\b', 'collected', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWMA\.?\b', 'wildlife management area', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNRA\.?\b', 'national recreation area', text, flags=re.IGNORECASE)
    text = re.sub(r'\bco\.\b', 'county', text, flags=re.IGNORECASE)
    text = re.sub(r'\bco\b', 'county', text, flags=re.IGNORECASE)
    #remove word before "county" but not if it's "county road"
    text = re.sub(r'\b(?!road\b)\w+\s+county\b', '', text, flags=re.IGNORECASE)

    # --- Convert spelled-out numbers before miles to digits ---
    number_words = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'eleven': '11',
        'twelve': '12',
        'thirteen': '13',
        'fourteen': '14',
        'fifteen': '15',
        'sixteen': '16',
        'seventeen': '17',
        'eighteen': '18',
        'nineteen': '19',
        'twenty': '20'
    }

    directions = [
        'north', 'south', 'east', 'west',
        'northeast', 'northwest', 'southeast', 'southwest'
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

    # --- Force singular "mile" to plural "miles" ---
    text = re.sub(r'\bmile\b', 'miles', text, flags=re.IGNORECASE)

    # --- Remove approximate qualifiers like "ca", "ca.", or "about" with punctuation ---
    text = re.sub(r'\b(?:about|ca\.?)\s+', '', text, flags=re.IGNORECASE)
    
    # --- Remove trailing periods from known words ---
    text = re.sub(r'\b(miles|north|south|east|west|northeast|northwest|southeast|southwest)\.', r'\1', text)

    # Remove "collected from, in, at, on, etc."
    text = re.sub(r'\bcollected\s+(from|in|at|on|along)\b', '', text, flags=re.IGNORECASE)
    # Remove "collected" solo at the beginning
    text = re.sub(r'^\s*collected\b[\s,:-]*', '', text, flags=re.IGNORECASE)

    # Remove trailing periods from the entire string
    text = re.sub(r'\.\s*$', '', text)

    # Remove pound/hash symbols
    text = text.replace('#', '')

    # Remove "Verbatim", "no additional locality" variations
    text = re.sub(r'[\(\[]\s*verbatim\s*[\)\]]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*no additional locality data on sheet\s*\]', '', text, flags=re.IGNORECASE)


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
    text = re.sub(r'[^\w\s.]', ' ', text)        # remove other punctuation

    # --- Normalize whitespace ---
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# --- Extract distances and directions ---
def extract_distance_direction(text):
    """
    Example: '3 miles south and 2 miles east' returns a sorted list of (distance, direction)
    """
    if pd.isnull(text):
        return []
    pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*(?:miles?|mi|kilometers|km)?[\s,]*(north|south|east|west|northeast|northwest|southeast|southwest)\b',
        flags=re.IGNORECASE
    )

    matches = pattern.findall(text)
    results = []
    for m in matches:
        num = float(m[0])
        if num.is_integer():
            num = int(num)
        results.append( (str(num), m[1].lower()) )
    # sort results by direction then distance
    results.sort(key=lambda x: (x[1], float(x[0])))
    return results

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
grouped['normalized_locality'] = grouped['locality'].apply(preprocess)
grouped['distance_direction'] = grouped['normalized_locality'].apply(extract_distance_direction)

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
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth"
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

        if (
            token_j.replace(".", "").isdigit()
            or token_j in protected_tokens
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

            if other in merged:
                continue

            print(f"Aliasing '{other}' ({token_freq[other]}) to '{canonical}' ({token_freq[canonical]}) (score {score:.2f} >= {threshold:.2f})")
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
group_members = {}

for i in range(len(grouped)):
    if suggested_ids[i] != -1:
        continue
    suggested_ids[i] = group_counter
    group_members[group_counter] = [i]
    for j in range(i + 1, len(grouped)):
        if suggested_ids[j] == -1 and similarity[i, j] >= threshold:
            suggested_ids[j] = group_counter
            group_members[group_counter].append(j)
    group_counter += 1

grouped['Suggested_ID'] = suggested_ids

# --- Compute confidence ---
confidence_scores = []
for idx in range(len(grouped)):
    group_id = grouped.iloc[idx]['Suggested_ID']
    members = group_members[group_id]
    if len(members) == 1:
        confidence = 1.0
    else:
        sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sims.append(similarity[members[i], members[j]])
        confidence = sum(sims) / len(sims)
    confidence_scores.append(confidence)
grouped['Confidence'] = [round(c * 100, 1) for c in confidence_scores]

# --- Validate suggested groups by distance/direction ---
grouped['Grouper_ID'] = grouped['Suggested_ID'].astype(str)

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


# --- Merge back ---
output_df = df.merge(
    grouped[[grouping_field, 'Grouper_ID', 'normalized_locality']],
    on=grouping_field,
    how='left'
)

# --- Export ---
columns_to_export = [
    'catalogNumber', 'institutionCode', 'collectionCode', 'county',
    'locality', 'bels_location_id', 'Grouper_ID', 'normalized_locality', 'Confidence'
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
export_df = export_df.sort_values(by='Grouper_ID', key=lambda col: col.map(sort_key))

output_file = os.path.splitext(csv_path)[0] + '-key.csv'
export_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Exported with suggested groups to: {output_file}")