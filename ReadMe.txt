><(((º>  Grouper ><(((º>

Grouper ingests a CSV/TSV of herbarium (or similar) records and groups similar localities. It then exports a compact "key" file with suggested group IDs, confidence scores, and a readable distance/direction summary. It's intended for use with pyBELS grouped data.


https://scribehow.com/viewer/BELSFish_Workflow__FVs4RRiqS7SS10UjxJp7UQ


><(((º> What it does ><(((º>

-Loads your file (.csv or .tsv) and validates required columns
-Preprocesses locality text (lowercasing, unit normalization, compass directions, removing boilerplate like “no additional data,” etc.)
-Extracts distance/direction tuples (e.g., 5 miles east → (5, east, miles))
-Builds TF-IDF with a custom tokenizer that keeps numbers and decimals
-Finds fuzzy aliases for similar tokens (e.g., hiway → highway) using RapidFuzz and dynamically chosen thresholds
-Rebuilds TF-IDF after aliasing and slightly up-weights numeric and directional tokens
-Groups records by cosine similarity (assigns Suggested_ID and Grouper_ID)
-Splits groups into subgroups when members differ by distance/direction signatures (12.1, 12.2, …)
-Null/placeholder localities get Grouper_ID = '0'
-Computes confidence as the average intra-group cosine similarity (0–100)
-Orders singletons after their most similar non-singleton group (for human-friendly review order)
-Exports a -key.csv that summarizes the grouping
><(((º> Requirements ><(((º>

pandas==2.2.3
scikit-learn==1.6.1
rapidfuzz==3.9.6

><(((º> Expected input columns ><(((º>

Your input file must include:
locality — free-text locality description (string)
bels_location_id — stable key used for grouping/merging (string/integer)

If present, these are used in the export for convenience:
catalogNumber, institutionCode, collectionCode, county

If any of the convenience columns are missing, the exporter will simply omit them.

><(((º> Usage ><(((º>

# Option A: provide the path on the command line
python grouper.py path/to/occurrences.csv

# Option B: run and paste the path when prompted
python grouper.py
Enter path to CSV/TSV file: /full/path/occurrences.tsv

The script infers the delimiter from the file extension: .csv → comma, .tsv → tab.
Unsupported extensions will exit with a clear message.

><(((º> Output ><(((º>

A single key file is written next to your input file:
<original-filename>-key.csv

Columns in the key file:
catalogNumber (optional passthrough)
institutionCode (optional passthrough)
collectionCode (optional passthrough)
county (optional passthrough)
locality (original text)
bels_location_id (original key)
Grouper_ID (string group ID; e.g., 12, 12.1, 0 for nulls)
normalized_locality (preprocessed text used for similarity)
Confidence (0–100; average intra-group cosine similarity, 1.0 for singletons → 100.0)
Distance_Direction (human-readable join of extracted tuples; e.g., 5 miles east; 0.5 miles north)

Note: Internally the script also merges the group IDs back to the full dataset (output_df), but the only file written is the compact *-key.csv intended for review/workflow joins.

><(((º> How it works ><(((º>

Text normalization:
-make everything lowercase
-replace bizarro spaces with normal spaces
-normalize units (mi→miles, km→kilometers, '→feet)
-highways (US 77, I-35, SH 6 → highway 77/35/6)
-street abbreviations
-compass abbreviations/compounds
-spelled-out numbers/fractions (incl. Unicode fractions)
-curated list of un-georeference-able placeholder phrases

Distance/direction extraction:
-Recognizes patterns like 3.5 kilometers southwest, 6mi E
-Fallback to find out-of-order phrases when no distance, unit, direction found

Vectorization:
-Custom tokenizer keeps words and decimal numbers
-removes most punctuation except decimal points
-uses tailored stopwords to down-weight generic habitat/soil noise.

Fuzzy token aliasing:
-RapidFuzz finds near-duplicate tokens with a dynamic threshold that scales with token length
-protects compass/directional tokens, ordinals, township codes, and numerics

Grouping:
-Cosine similarity over TF-IDF vectors with a default threshold of 0.85 assigns Suggested_ID. 
-Distance/direction signatures can split a group into .1/.2/... subgroups when necessary

Confidence:
-Per-record score = average similarity to other members of its group (×100; 1 member → 100.0 by definition)

Human-friendly ordering:
-Singletons are placed after their closest non-singleton group when the max similarity ≥ 0.80 (configurable in code)

><(((º> Configuration ><(((º>

Similarity threshold: 
-group_by_similarity(..., threshold=0.85) (currently hard-coded inside).

Singleton placement minimum: 
-reorder_similar_singletons(..., min_similarity=0.80)

Dynamic fuzzy thresholding: 
-see dynamic_threshold() (base/max thresholds) and fuzzy_alias_tokens()

Protected tokens / stop words / abbreviations:
-get_custom_stop_words()
-fuzzy_alias_tokens().protected_tokens
-preprocess() abbreviation maps and removal lists


><(((º> Troubleshooting ><(((º>

"I got a 'CSV must contain 'locality' and 'bels_location_id' columns.' error!"
Ensure your header names match exactly.

"Why is this so slow?""
Cosine similarity uses an all-pairs matrix - it's comparing every record to every other record. I've run it on batches of 20,000+ records and it's pretty fast. If performance is poor, try pre-clustering by county/region to reduce pair counts.

"Why is this converting miles to meters?"
convert_m_unit() converts "m" to "meters" when it's preceeded by a number above 20, and "miles" when preceded by a number below 20. If this isn't working for your dataset, you can adjust this.

"It's too agressive/not agressive enought in merging stuff.
Tweak the cosine threshold (default 0.85) and/or aliasing aggressiveness.

"The aliasing is changing stuff that's obviously fine!"
Yeah, it probably is! Don't worry too much about good words being lost. Since the aliasing only works on low-count words and it's changing them consistently, it likely doesn't have much negative effect on grouping.

If you have ideas for improvement, please reach out to cmeyer@fwbg.org
