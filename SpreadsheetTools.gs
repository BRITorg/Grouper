//8-21-25

function onOpen() {
  const ui = SpreadsheetApp.getUi();

  // Create submenus
  const KeyMenu = ui.createMenu('Key Sheet Tools')
    .addItem('Remove Common Words', 'removeCommonWords')
    .addItem('Split IDs by keywords', 'addDirectionalDecimalsToGroupID')
    .addItem('Recalculate Confidence', 'calculateGroupConfidence')
    .addItem('Update Cell Shading', 'updateSerialOnKeySheet');

  const OriginalMenu = ui.createMenu('Original Sheet Tools')
    .addItem('Populate Blank Catalog Numbers', 'fillBlankCatalogNumbers')
    .addItem('Populate Grouper_id from Key', 'fillGrouperIDFormulas')
    .addItem('Finalize Review Column', 'FinalizeReview');


  const ToCogeMenu = ui.createMenu('To CoGe Export Tools')
    .addItem('Export CSV(s) for CoGe', 'exportCSVsByCounty')

  const FromCogeMenu = ui.createMenu('From CoGe Import Tools')
    .addItem('Highlight/Count Duplicates', 'highlightCorrectedAndSkippedDuplicates')
    .addItem('Populate Data from CoGe', 'fillCoGeFormulas')

// Main menu with nested submenus
ui.createMenu('BELSFish')
  .addSubMenu(KeyMenu)
  .addSubMenu(OriginalMenu)
  .addSubMenu(ToCogeMenu)
  .addSubMenu(FromCogeMenu)
  .addToUi();
}

//Key Sheet Tools
function removeCommonWords() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Key");
  const dataRange = sheet.getDataRange();
  const data = dataRange.getValues();

  // Identify column indexes
  const header = data[0];
  const idCol = header.indexOf('Grouper_ID');
  const locCol = header.indexOf('normalized_locality');

  if (idCol === -1 || locCol === -1) {
    SpreadsheetApp.getUi().alert('Columns "Grouper_ID" and/or "normalized_locality" not found.');
    return;
  }

  // Determine if "unique_locality" column already exists
  let outputCol = header.indexOf('unique_locality');
  if (outputCol === -1) {
    outputCol = header.length;
    sheet.getRange(1, outputCol + 1).setValue('unique_locality');
  }

  const wordsToExclude = ["of", "on", "and", "in", "at", "the", "around", "to"];  // <- Words to always remove (case-insensitive)
  const groups = {};

  for (let i = 1; i < data.length; i++) {
    const id = data[i][idCol];
    const locality = data[i][locCol];
    if (!groups[id]) groups[id] = [];
    groups[id].push({ row: i, text: locality });
  }

  const result = new Array(data.length).fill('');
  for (const group of Object.values(groups)) {
    const texts = group.map(entry => entry.text);
    const wordSets = texts.map(text =>
      new Set(text.toLowerCase().match(/\b[\w’'-]+\b/g) || [])
    );

    const commonWords = [...wordSets[0]].filter(word =>
      wordSets.every(set => set.has(word))
    );

    for (const entry of group) {
      const filtered = (entry.text.match(/\b[\w’'-]+\b/g) || [])
        .filter(word =>
          !commonWords.includes(word.toLowerCase()) &&
          !wordsToExclude.includes(word.toLowerCase())
        )
        .join(' ');
      result[entry.row] = filtered;
    }
  }

  // Write results
  for (let i = 1; i < result.length; i++) {
    sheet.getRange(i + 1, outputCol + 1).setValue(result[i]);
  }
}


function addDirectionalDecimalsToGroupID() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Key");
  const data = sheet.getDataRange().getValues();

  const header = data[0];
  const idCol = header.indexOf('Grouper_ID');
  const localityCol = header.indexOf('unique_locality');

  if (idCol === -1 || localityCol === -1) {
    SpreadsheetApp.getUi().alert('Missing "Grouper_ID" or "unique_locality" column.');
    return;
  }

  // Longer direction terms come first so we don’t match "south" inside "southwest"
  const keywordDecimalMap = {
    'northwest': 0.08,
    'southwest': 0.07,
    'southeast': 0.06,
    'northeast': 0.05,
    'west': 0.04,
    'east': 0.03,
    'south': 0.02,
    'north': 0.01,
    'near': 0.09
  };

  for (let i = 1; i < data.length; i++) {
    let locality = data[i][localityCol];
    const id = data[i][idCol];

    if (typeof locality === 'string' && id !== '') {
      locality = locality.toLowerCase().trim();
      const words = locality.split(/\s+/); // Split by whitespace
      const matched = new Set();

      // Check full string for compound direction keywords first
      for (const keyword in keywordDecimalMap) {
        if (locality.includes(keyword)) {
          matched.add(keyword);
        }
      }

      // Fallback: check token-by-token
      for (const word of words) {
        if (keywordDecimalMap[word]) {
          matched.add(word);
        }
      }

      const baseID = parseFloat(id);
      if (!isNaN(baseID)) {
        let newID = null;

        if (matched.size === 1) {
          const keyword = [...matched][0];
          newID = baseID + keywordDecimalMap[keyword];
        } else if (matched.size > 1) {
          newID = baseID + 0.099;
        }

        if (newID !== null) {
          if (Math.floor(newID) === baseID) {
            sheet.getRange(i + 1, idCol + 1).setValue(newID.toFixed(3));
          }
        }
      }
    }
  }

  SpreadsheetApp.getUi().alert('Grouper_IDs updated with directional or proximity decimal values.');
}

function calculateGroupConfidence() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Key");
  const data = sheet.getDataRange().getValues();
  const header = data[0];

  const idCol = header.indexOf('Grouper_ID');
  const normCol = header.indexOf('normalized_locality');

  if (idCol === -1 || normCol === -1) {
    SpreadsheetApp.getUi().alert('Columns "Grouper_ID" and/or "normalized_locality" not found.');
    return;
  }

  let confCol = header.indexOf('Confidence');
  if (confCol === -1) {
    confCol = header.length;
    sheet.getRange(1, confCol + 1).setValue('Confidence');
  }

  // Group normalized text by Grouper_ID
  const groups = {};
  for (let i = 1; i < data.length; i++) {
    const id = data[i][idCol];
    const text = data[i][normCol] || '';
    if (!groups[id]) groups[id] = [];
    groups[id].push({ row: i, text });
  }

  // Simple TF-IDF and cosine sim for small groups
  function tokenize(text) {
    return text.toLowerCase().match(/\b[\w\.]+\b/g) || [];
  }

  function tf(tokens) {
    const freq = {};
    tokens.forEach(t => freq[t] = (freq[t] || 0) + 1);
    const tfVec = {};
    for (let t in freq) tfVec[t] = freq[t] / tokens.length;
    return tfVec;
  }

  function cosine(vec1, vec2) {
    let dot = 0, mag1 = 0, mag2 = 0;
    const allKeys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    allKeys.forEach(k => {
      const a = vec1[k] || 0, b = vec2[k] || 0;
      dot += a * b;
      mag1 += a * a;
      mag2 += b * b;
    });
    return mag1 && mag2 ? dot / (Math.sqrt(mag1) * Math.sqrt(mag2)) : 0;
  }

  for (const id in groups) {
    const entries = groups[id];
    const vectors = entries.map(e => tf(tokenize(e.text)));

    let simSum = 0;
    let count = 0;
    for (let i = 0; i < vectors.length; i++) {
      for (let j = i + 1; j < vectors.length; j++) {
        simSum += cosine(vectors[i], vectors[j]);
        count++;
      }
    }

    const confidence = count > 0 ? (simSum / count * 100).toFixed(1) : '100.0';

    // Set confidence for all rows in the group
    entries.forEach(e => {
      sheet.getRange(e.row + 1, confCol + 1).setValue(confidence);
    });
  }
}

function updateSerialOnKeySheet() {
  const sheetName = "Key";
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) {
    SpreadsheetApp.getUi().alert(`Sheet named "${sheetName}" not found.`);
    return;
  }

  // Step 1: Check if cell A1 is labeled "Serial"
  const header = sheet.getRange("A1").getValue();
  if (header !== "Serial") {
    SpreadsheetApp.getUi().alert(`Column A1 must be labeled "Serial". Found: "${header}"`);
    return;
  }

  // Step 2: Remove completely blank rows at the bottom
  const lastRow = sheet.getLastRow();
  const lastCol = sheet.getLastColumn();
  const data = sheet.getRange(1, 1, lastRow, lastCol).getValues();

  // Loop from the bottom up to find and delete blank rows
  for (let i = data.length - 1; i >= 1; i--) {
    const row = data[i];
    const isBlank = row.every(cell => cell === "" || cell === null);
    if (isBlank) {
      sheet.deleteRow(i + 1); // account for zero-based index
    } else {
      break; // stop once non-blank row is found
    }
  }

  // Step 3: Find the column index of "Grouper_ID"
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  const targetColIndex = headers.indexOf("Grouper_ID") + 1; // +1 because getRange is 1-based
  if (targetColIndex === 0) {
    SpreadsheetApp.getUi().alert(`Column "Grouper_ID" not found.`);
    return;
  }

  // Step 4: Fill A2 with 1, and A3 downward with formula
  const numRows = sheet.getLastRow();
  if (numRows < 2) return; // nothing to do

  sheet.getRange("A2").setValue(1);

  if (numRows >= 3) {
    const formulaRange = sheet.getRange(3, 1, numRows - 2); // from A3 to last row
    const formulas = [];

    for (let i = 3; i <= numRows; i++) {
      const colLetter = String.fromCharCode(64 + targetColIndex);
      const rowFormula = `=IF(${colLetter}${i}<>${colLetter}${i - 1}, A${i - 1}+1, A${i - 1})`;
      formulas.push([rowFormula]);
    }

    formulaRange.setFormulas(formulas);
  }
}

//Original Sheet Tools
function fillBlankCatalogNumbers() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Original");
  const data = sheet.getDataRange().getValues();

  const headers = data[0];
  const catalogCol = headers.indexOf("catalogNumber");
  const idCol = headers.indexOf("id");

  if (catalogCol === -1) {
    SpreadsheetApp.getUi().alert('The column "catalogNumber" was not found.');
    return;
  }

  if (idCol === -1) {
    SpreadsheetApp.getUi().alert('The column "id" was not found.');
    return;
  }

  const updates = [];

  for (let i = 1; i < data.length; i++) {
    const catalogVal = data[i][catalogCol];
    const idVal = data[i][idCol];

    if (!catalogVal || catalogVal.toString().trim() === "") {
      if (idVal !== "" && idVal !== null) {
        updates.push([i + 1, catalogCol + 1, "ID_" + idVal.toString().trim()]);
      }
    }
  }

  // Apply the updates
  updates.forEach(([row, col, value]) => {
    sheet.getRange(row, col).setValue(value);
  });
}


function FinalizeReview() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName('Original');
  if (!sheet) {
    SpreadsheetApp.getUi().alert('Sheet "Original" not found.');
    return;
  }

  // Find the REVIEW column by header name (row 1)
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  const reviewColIndex = headers.indexOf('REVIEW') + 1; // 1-based
  if (reviewColIndex === 0) {
    SpreadsheetApp.getUi().alert('Column "REVIEW" not found on the "Original" sheet.');
    return;
  }

  const lastRow = sheet.getLastRow();
  if (lastRow <= 1) {
    // No data rows
    return;
  }

  // Range beneath the header in the REVIEW column
  const reviewRange = sheet.getRange(2, reviewColIndex, lastRow - 1, 1);

  // Do an in-place "paste values only"
  reviewRange.copyTo(reviewRange, { contentsOnly: true });
}



function fillCoGeFormulas() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const original = ss.getSheetByName("Original");
  const coge = ss.getSheetByName("From CoGe");
  if (!original) {
    SpreadsheetApp.getUi().alert('Sheet "Original" not found.');
    return;
  }
  if (!coge) {
    SpreadsheetApp.getUi().alert('Sheet "From CoGe" not found.');
    return;
  }

  // ====== CONFIG: set these to your header text exactly as it appears in row 1 ======
  // Original sheet headers
  const ORIGINAL_REVIEW_HEADER = "REVIEW";            // (AG previously)
  const ORIGINAL_KEY_HEADER    = "Grouper_ID";     // (AA previously) — used to match to From CoGe

  // From CoGe headers (the column to MATCH on and the source columns to pull)
  const COGE_MATCH_HEADER      = "Grouper_ID";     // (AD previously) — used in the FILTER condition
  const MAP = [
    // { originalHeader: header text in Original where the value/formula goes,
    //   cogeHeader: header text in From CoGe to pull from,
    //   asDate: true to wrap with TEXT(...,"MM/DD/YY") }
    { originalHeader: "Completed",                        cogeHeader: "Date verified", asDate: true  }, // F ← From CoGe F
    { originalHeader: "georeferencedBy",                  cogeHeader: "Verified by",    asDate: false }, // G ← From CoGe I
    { originalHeader: "decimalLatitude",                  cogeHeader: "Corrected latitude",       asDate: false }, // H ← From CoGe H
    { originalHeader: "decimalLongitude",                 cogeHeader: "Corrected longitude",       asDate: false }, // J ← From CoGe J
    { originalHeader: "coordinateUncertaintyInMeters",    cogeHeader: "Corrected uncertainty radius",       asDate: false }, // L ← From CoGe E
    { originalHeader: "georeferenceRemarks",              cogeHeader: "Verification remarks",       asDate: false }, // P ← From CoGe G
  ];

  // Constant fills for Original
  const CONSTANTS = [
    { originalHeader: "georeferenceProtocol", text: "CCH2 Georef. Protocol 09/30/2020" }, // M
    { originalHeader: "georeferenceSources",     text: "GEOLocate Batch Processing Tool"  }, // N
  ];
  // ====== /CONFIG ======

  // Helpers
  const toColLetter = (n) => {
    let s = "";
    while (n > 0) { let m = (n - 1) % 26; s = String.fromCharCode(65 + m) + s; n = (n - m - 1) / 26; }
    return s;
  };
  const headerIndexOrDie = (headers, name, where) => {
    const i = headers.indexOf(name);
    if (i === -1) {
      throw new Error(`Header "${name}" not found in ${where}. Available: ${headers.join(" | ")}`);
    }
    return i + 1; // 1-based
  };

  // Read headers
  const origHeaders = original.getRange(1, 1, 1, original.getLastColumn()).getValues()[0].map(String);
  const cogeHeaders = coge.getRange(1, 1, 1, coge.getLastColumn()).getValues()[0].map(String);

  // Resolve dynamic columns on Original
  const reviewCol = headerIndexOrDie(origHeaders, ORIGINAL_REVIEW_HEADER, "Original");
  const reviewLetter = toColLetter(reviewCol);

  const keyCol = headerIndexOrDie(origHeaders, ORIGINAL_KEY_HEADER, "Original");
  const keyLetter = toColLetter(keyCol);

  // Resolve dynamic column on From CoGe used for matching
  const cogeMatchCol = headerIndexOrDie(cogeHeaders, COGE_MATCH_HEADER, "From CoGe");
  const cogeMatchLetter = toColLetter(cogeMatchCol);

  // Resolve targets (Original dest + From CoGe source)
  const resolvedTargets = MAP.map(t => {
    const origCol = headerIndexOrDie(origHeaders, t.originalHeader, "Original");
    const cogeCol = headerIndexOrDie(cogeHeaders, t.cogeHeader, "From CoGe");
    return {
      origCol,
      origLetter: toColLetter(origCol),
      cogeLetter: toColLetter(cogeCol),
      asDate: !!t.asDate
    };
  });

  // Resolve constants (Original dest headers)
  const resolvedConstants = CONSTANTS.map(t => {
    const origCol = headerIndexOrDie(origHeaders, t.originalHeader, "Original");
    return { col: origCol, text: t.text };
  });

  const lastRow = original.getLastRow();
  if (lastRow < 2) return;

  const startRow = 2;
  const numRows = lastRow - 1;

  // Read REVIEW (dynamic) and all destination columns once
  const reviewVals = original.getRange(startRow, reviewCol, numRows, 1).getValues();

  const destValues = {};
  for (const tgt of resolvedTargets) {
    destValues[tgt.origCol] = original.getRange(startRow, tgt.origCol, numRows, 1).getValues();
  }
  const constValues = {};
  for (const c of resolvedConstants) {
    constValues[c.col] = original.getRange(startRow, c.col, numRows, 1).getValues();
  }

  // Fill loop
  for (let i = 0; i < numRows; i++) {
    const row = startRow + i;
    const review = (reviewVals[i][0] ?? "").toString().trim().toLowerCase();

    // Case-insensitive check for 'none' or 'skip-none'
    if (review === "none" || review === "skip-none") {
      // Build filter condition referencing dynamic key columns
      const keyRef = `${keyLetter}${row}`;
      const matchRange = `'From CoGe'!${cogeMatchLetter}:${cogeMatchLetter}`;

      // Formulas
      for (const tgt of resolvedTargets) {
        const current = (destValues[tgt.origCol][i][0] ?? "").toString().trim();
        if (current === "") {
          const sourceRange = `'From CoGe'!${tgt.cogeLetter}:${tgt.cogeLetter}`;
          const base = `FILTER(${sourceRange}, ${matchRange}=${keyRef})`;
          const formula = tgt.asDate ? `=TEXT(${base}, "MM/DD/YY")` : `=${base}`;
          original.getRange(row, tgt.origCol).setFormula(formula);
        }
      }

      // Constants
      for (const c of resolvedConstants) {
        const cur = (constValues[c.col][i][0] ?? "").toString().trim();
        if (cur === "") {
          original.getRange(row, c.col).setValue(c.text);
        }
      }
    }
  }
}

//To CoGe Tools

function exportCSVsByCounty() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("To CoGe");
  const data = sheet.getDataRange().getValues();

  const headers = data[0];
  const headerLength = headers.length;

  const filteredRows = data.slice(1).filter(row => {
    const status = row[0];             // Column A: "NONE"
    const institutionCount = row[1];   // Column B: InstitutionCount
    return status === 'NONE' && Number(institutionCount) > 0;
  });

  if (filteredRows.length === 0) {
    SpreadsheetApp.getUi().alert('No rows matched the filter (NONE + InstitutionCount > 0).');
    return;
  }

  const grouped = {};

  filteredRows.forEach(row => {
    const county = row[10] || 'Blank'; // Column K is index 9, contains county
    if (!grouped[county]) grouped[county] = [];
    grouped[county].push(row.slice(0, headerLength)); // enforce same column count
  });

  const timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd_HH:mm:ss');
  const folder = DriveApp.createFolder('County_Exports_' + timestamp);

  for (const county in grouped) {
    let csvContent = headers.join(",") + "\n";

    grouped[county].forEach(row => {
      const cleaned = row.map(cell => `"${String(cell).replace(/"/g, '""')}"`);
      csvContent += cleaned.join(",") + "\n";
    });

    folder.createFile(`${county}_Export.csv`, csvContent, MimeType.CSV);
  }

  SpreadsheetApp.getUi().alert('CSV exports completed. Check your Google Drive.');
}

//From CoGe Tools

function highlightCorrectedAndSkippedDuplicates() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName("From CoGe");
  if (!sheet) {
    SpreadsheetApp.getUi().alert('Sheet "From CoGe" not found!');
    return;
  }

  // ---------- 0) CLEANUP: Replace "N\A" (any case, whole-cell) with blank in one shot ----------
  // NOTE: matchEntireCell(true) prevents touching "X N\A Y"
  sheet.createTextFinder("N\\A")
    .matchCase(false)
    .matchEntireCell(true)
    .useRegularExpression(false)
    .replaceAllWith("");

  // ---------- Helpers ----------
  const colToLetter = (col) => {
    let s = "";
    while (col > 0) {
      const m = (col - 1) % 26;
      s = String.fromCharCode(65 + m) + s;
      col = Math.floor((col - 1) / 26);
    }
    return s;
  };

  const lastRow = sheet.getLastRow();
  const lastCol = sheet.getLastColumn();
  if (lastRow < 2) {
    SpreadsheetApp.getUi().alert("No data rows found.");
    return;
  }

  const dataRange = sheet.getRange(1, 1, lastRow, lastCol);
  const data = dataRange.getValues();
  const headers = data[0];

  const catalogIndex = headers.indexOf("CatalogNumber");
  const verificationIndex = headers.indexOf("Verification type");
  const dateIndex = headers.indexOf("Date verified");

  if (catalogIndex === -1 || verificationIndex === -1 || dateIndex === -1) {
    SpreadsheetApp.getUi().alert('Missing one or more required columns: "CatalogNumber", "Verification type", or "Date verified"');
    return;
  }

  // ---------- 1) Pre-parse values for speed ----------
  // Row index in arrays is sheet row (1-based) for readability; we ignore index 0.
  const typeLower = new Array(lastRow + 1);
  const dateMs    = new Array(lastRow + 1);
  for (let r = 2; r <= lastRow; r++) {
    const row = data[r - 1];
    typeLower[r] = String(row[verificationIndex] || "").toLowerCase().trim();
    const v = row[dateIndex];
    const d = (v instanceof Date) ? v : new Date(v);
    dateMs[r] = isNaN(d) ? null : d.getTime();
  }

  // ---------- 2) Group rows by CatalogNumber ----------
  const groups = new Map(); // cat -> int[] (rows)
  for (let r = 2; r <= lastRow; r++) {
    const cat = data[r - 1][catalogIndex];
    if (!cat) continue;
    let arr = groups.get(cat);
    if (!arr) groups.set(cat, (arr = []));
    arr.push(r);
  }

  // ---------- 3) Decide highlights (collect A1 row ranges to batch-apply) ----------
  const lastColLetter = colToLetter(lastCol);
  const greenRowsA1 = [];
  const redRowsA1 = [];
  const orangeRowsA1 = [];

  const rowBand = (r) => `A${r}:${lastColLetter}${r}`;
  const newestByDate = (rows) => {
    if (rows.length === 0) return null;
    let best = null, bestMs = -Infinity;
    for (const r of rows) {
      const ms = dateMs[r];
      const v = (ms == null) ? -Infinity : ms;
      if (v > bestMs) { bestMs = v; best = r; }
    }
    return best;
  };

  for (const [, rows] of groups) {
    if (rows.length < 2) continue; // skip singletons

    // Partition by type
    const corrected = [];
    const skipped = [];
    const other = [];
    for (const r of rows) {
      const t = typeLower[r];
      if (t === "corrected") corrected.push(r);
      else if (t === "skipped") skipped.push(r);
      else other.push(r);
    }

    if (corrected.length > 0 && skipped.length > 0) {
      const nc = newestByDate(corrected);
      const ns = newestByDate(skipped);
      const dc = (nc && dateMs[nc] != null) ? dateMs[nc] : null;
      const ds = (ns && dateMs[ns] != null) ? dateMs[ns] : null;

      if (dc != null && ds != null && dc > ds) {
        greenRowsA1.push(rowBand(nc));
        redRowsA1.push(rowBand(ns));
        for (const r of rows) {
          if (r !== nc && r !== ns) orangeRowsA1.push(rowBand(r));
        }
      } else {
        // dates missing or corrected not newer -> all orange
        for (const r of rows) orangeRowsA1.push(rowBand(r));
      }
    } else if (corrected.length > 1 && skipped.length === 0) {
      const nc = newestByDate(corrected);
      greenRowsA1.push(rowBand(nc));
      for (const r of corrected) if (r !== nc) redRowsA1.push(rowBand(r));
      for (const r of rows) if (corrected.indexOf(r) === -1) orangeRowsA1.push(rowBand(r));
    } else {
      // only skippeds, or mixed with no clear rule -> all orange
      for (const r of rows) orangeRowsA1.push(rowBand(r));
    }
  }

  // ---------- 4) Clear previous backgrounds in one call ----------
  if (lastRow > 1 && lastCol > 0) {
    sheet.getRange(2, 1, lastRow - 1, lastCol).setBackground(null);
  }

  // ---------- 5) Apply highlights with 3 batched RangeList calls ----------
  if (greenRowsA1.length) sheet.getRangeList(greenRowsA1).setBackground("#b6d7a8");
  if (redRowsA1.length)   sheet.getRangeList(redRowsA1).setBackground("#f4cccc");
  if (orangeRowsA1.length)sheet.getRangeList(orangeRowsA1).setBackground("#fce5cd");

  // ---------- 6) Add/overwrite "Grouper_ID" and "count" using single ARRAYFORMULAs ----------
  // Put them at the next two free columns.
  const startCol = lastCol + 1;
  const grouperCol = startCol;
  const countCol = startCol + 1;

  sheet.getRange(1, grouperCol).setValue("Grouper_ID");
  sheet.getRange(1, countCol).setValue("count");

  const catColLetter = colToLetter(catalogIndex + 1);

  // Clear old contents in those columns (if any) then set ArrayFormulas
  if (lastRow >= 2) {
    sheet.getRange(2, grouperCol, lastRow - 1, 1).clearContent();
    sheet.getRange(2, countCol, lastRow - 1, 1).clearContent();

    // VLOOKUP version (fast). If you must keep FILTER, swap the formula below.
    // AD is column 30 on Original (A=1 ... AD=30). Adjust if different.
    sheet.getRange(2, grouperCol).setFormula(
      `=ARRAYFORMULA(IF(ROW(${catColLetter}2:${catColLetter})=1,"",` +
      `IF(${catColLetter}2:${catColLetter}="", "", IFERROR(VLOOKUP(${catColLetter}2:${catColLetter}, Original!A:AD, 30, FALSE), ""))))`
    );

    sheet.getRange(2, countCol).setFormula(
      `=ARRAYFORMULA(IF(${catColLetter}2:${catColLetter}="","",COUNTIF(${catColLetter}2:${catColLetter}, ${catColLetter}2:${catColLetter})))`
    );
  }
}





function fillGrouperIDFormulas() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const originalSheet = ss.getSheetByName("Original");
  const keySheet = ss.getSheetByName("Key");

  const originalHeaders = originalSheet.getRange(1, 1, 1, originalSheet.getLastColumn()).getValues()[0];
  const keyHeaders = keySheet.getRange(1, 1, 1, keySheet.getLastColumn()).getValues()[0];

  // Find target column in Original ("Grouper_ID" or fallback "FinalName")
  let targetCol = originalHeaders.indexOf("Grouper_ID");
  if (targetCol === -1) targetCol = originalHeaders.indexOf("FinalName");

  const originalBelsCol = originalHeaders.indexOf("bels_location_id");
  const keyGrouperCol = keyHeaders.indexOf("Grouper_ID");
  const keyBelsCol = keyHeaders.indexOf("bels_location_id");

  if (targetCol === -1 || originalBelsCol === -1 || keyGrouperCol === -1 || keyBelsCol === -1) {
    SpreadsheetApp.getUi().alert(
      'Missing required columns. Need ("Grouper_ID" or "FinalName") and "bels_location_id" in Original, plus "Grouper_ID" and "bels_location_id" in Key.'
    );
    return;
  }

  const lastRow = originalSheet.getLastRow();
  if (lastRow < 2) return; // nothing to do

  const numRows = lastRow - 1;
  const originalBels = originalSheet.getRange(2, originalBelsCol + 1, numRows, 1).getValues();

  const keyData = keySheet.getRange(2, 1, Math.max(keySheet.getLastRow() - 1, 0), keySheet.getLastColumn()).getValues();

  // Build bels_location_id → Grouper_ID map
  const belsToGrouper = {};
  keyData.forEach(row => {
    const belsId = row[keyBelsCol];
    const grouperId = row[keyGrouperCol];
    if (belsId !== "" && belsId !== null && grouperId !== "" && grouperId !== null) {
      belsToGrouper[belsId] = grouperId;
    }
  });

  // Build output column (blank when no match)
  const outCol = originalBels.map(belsRow => {
    const belsId = belsRow[0];
    if (belsId in belsToGrouper) {
      return [belsToGrouper[belsId]];
    }
    return [null]; // blank cell when no match
  });

  // Write ONLY the target column
  originalSheet.getRange(2, targetCol + 1, numRows, 1).setValues(outCol);
}