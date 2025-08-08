import csv
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

COLUMN_ORDER = [
    "catalogNumber", "scientificName", "country", "stateProvince", "institutionCode", "collectionCode", "county", "locality",
    "Completed", "decimalLatitude", "decimalLongitude", "geodeticDatum",
    "coordinateUncertaintyInMeters", "verbatimCoordinates", "georeferencedBy",
    "georeferenceProtocol", "georeferenceSources", "georeferenceVerificationStatus",
    "georeferenceRemarks", "id", "recordedBy", "recordNumber", "eventDate", "year", "month", "day", "habitat", "references",
    "bels_location_id",
]

CREATED_COLUMNS = [
    "Grouper_ID", "isInstitution",
    "decimalLatitudeCount", "decimalLongitudeCount", "MOOSH", "InstitutionCount", "REVIEW", "wheresWalter"
]

def get_folder():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select Folder Containing TSV Files")

def colnum_to_excel_col(n):
    col = ""
    while n >= 0:
        col = chr(n % 26 + 65) + col
        n = n // 26 - 1
    return col

def process_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read '{file_path}': {e}")
        return

    # Ensure required columns exist
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA

    # Add created columns if missing
    for col in CREATED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[[col for col in COLUMN_ORDER + CREATED_COLUMNS if col in df.columns]]

    # Fill formulas
    df["Grouper_ID"] = pd.NA
    try:
        institution_col_letter = colnum_to_excel_col(df.columns.get_loc("institutionCode"))
        df["isInstitution"] = [
            f'=REGEXMATCH({institution_col_letter}{i+2}, "BRIT|TEX|OKL|OKLA|LL|LLC|HSU|ILL|WILLI|TCSW|NTSC|VDB|ILLS|BAYLU|CAMU|CSU|DUR|ECSC|NOSU|NWOSU|OCU|PAUH|SAT|SEU|SHST|SRSC|TAES|TTC|TULS|UTEP|WTS")'
            for i in range(len(df))
        ]
    except ValueError:
        print(f"⚠️ Column 'institutionCode' not found in '{file_path}'. Skipping isInstitution formula.")
        df["isInstitution"] = pd.NA
    
    try:
        all_columns = list(df.columns)
        lat_col = colnum_to_excel_col(all_columns.index("decimalLatitude"))
        lon_col = colnum_to_excel_col(all_columns.index("decimalLongitude"))
        finalname_col = colnum_to_excel_col(all_columns.index("Grouper_ID"))
        uncertainty_col = colnum_to_excel_col(all_columns.index("coordinateUncertaintyInMeters"))

        df["decimalLatitudeCount"] = [
            f'=COUNTUNIQUEIFS({lat_col}:{lat_col}, {finalname_col}:{finalname_col}, {finalname_col}{i+2})'
            for i in range(len(df))
        ]
        df["decimalLongitudeCount"] = [
            f'=COUNTUNIQUEIFS({lon_col}:{lon_col}, {finalname_col}:{finalname_col}, {finalname_col}{i+2})'
            for i in range(len(df))
        ]
        df["MOOSH"] = [
            f'=CONCATENATE({finalname_col}{i+2},{lat_col}{i+2},{lon_col}{i+2},{uncertainty_col}{i+2})'
            for i in range(len(df))
        ]

        latcount_col = colnum_to_excel_col(all_columns.index("decimalLatitudeCount"))
        loncount_col = colnum_to_excel_col(all_columns.index("decimalLongitudeCount"))
        moosh_col = colnum_to_excel_col(all_columns.index("MOOSH"))
        isinstitution_col = colnum_to_excel_col(all_columns.index("isInstitution"))

        df["REVIEW"] = [
            f'=IF(SUM({loncount_col}{row},{moosh_col}{row})=0,'
            f'IF({finalname_col}{row-1}<>${finalname_col}{row},"NONE","Skip-none"),'
            f'IF(SUM({latcount_col}{row},{loncount_col}{row})=2,'
            f'IF(NOT(ISBLANK({loncount_col}{row})),"ONE","-"),'
            f'IF(NOT(ISBLANK({loncount_col}{row})),'
            f'IF(COUNTIFS(${moosh_col}$2:{moosh_col}{row},{moosh_col}{row})=1,"TON","Skip-dupCoord"),"-")))'
            for row in range(2, len(df)+2)
        ]

        df["InstitutionCount"] = [
            f'=IF({finalname_col}{i+2}="", "", COUNTIFS({finalname_col}:{finalname_col}, {finalname_col}{i+2}, {isinstitution_col}:{isinstitution_col}, TRUE))'
            for i in range(len(df))
        ]

        lat_col_letter = colnum_to_excel_col(all_columns.index("decimalLatitude"))
        lon_col_letter = colnum_to_excel_col(all_columns.index("decimalLongitude"))
        uncertainty_col_letter = colnum_to_excel_col(all_columns.index("coordinateUncertaintyInMeters"))

        locality_col_letter = colnum_to_excel_col(all_columns.index("locality"))
        eventdate_col_letter = colnum_to_excel_col(all_columns.index("eventDate"))

        df["wheresWalter"] = [
            f'=IF({lat_col_letter}{i+2}="", "", HYPERLINK(CONCATENATE("https://cmeyer56555.github.io/Grouper/?lat=", {lat_col_letter}{i+2}, "&lon=", {lon_col_letter}{i+2}, "&radius=", {uncertainty_col_letter}{i+2}, "&locality=", ENCODEURL({locality_col_letter}{i+2}), "&date=", ENCODEURL({eventdate_col_letter}{i+2})), "MAP"))'
            for i in range(len(df))
        ]


    except ValueError as e:
        print(f"⚠️ Skipping formula columns in '{file_path}' due to missing columns: {e}")

    # Write to output
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}-trimmed.tsv"
    df.to_csv(output_path, sep='\t', index=False)
    print(f"✅ Saved: {output_path}")

def main():
    folder = get_folder()
    if not folder:
        print("No folder selected. Exiting...")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".tsv")]
    if not files:
        print("No TSV files found in the folder.")
        return

    for file in files:
        path = os.path.join(folder, file)
        process_file(path)

if __name__ == "__main__":
    main()
