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
    .addItem('Populate Grouper_id from Key', 'fillFinalNameFormulas');

  const ToCogeMenu = ui.createMenu('To CoGe Export Tools')
    .addItem('Export CSV(s) for CoGe', 'exportCSVsByCounty')

  const FromCogeMenu = ui.createMenu('From CoGe Import Tools')
    .addItem('Highlight duplicates', 'highlightCorrectedAndSkippedDuplicates')

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


function fillFinalNameFormulas() {
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
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("From CoGe");
  if (!sheet) {
    SpreadsheetApp.getUi().alert('Sheet "From CoGe" not found!');
    return;
  }

  const data = sheet.getDataRange().getValues();
  const headers = data[0];
  const catalogIndex = headers.indexOf("catalognumber");
  const verificationIndex = headers.indexOf("Verification type");
  const dateIndex = headers.indexOf("Date verified");

  if (catalogIndex === -1 || verificationIndex === -1 || dateIndex === -1) {
    SpreadsheetApp.getUi().alert('Missing one or more required columns: "catalognumber", "Verification type", or "Date verified"');
    return;
  }

  const seen = new Map(); // catalogNumber -> row indices
  const rowsToColor = {
    green: [],
    red: [],
    orange: []
  };

  // Build map of catalog numbers to row indices
  for (let i = 1; i < data.length; i++) {
    const catNum = data[i][catalogIndex];
    if (!catNum) continue;

    if (!seen.has(catNum)) {
      seen.set(catNum, []);
    }
    seen.get(catNum).push(i); // store row index (0-based)
  }

  for (const [catNum, rowIndices] of seen.entries()) {
    if (rowIndices.length < 2) continue;

    const correctedRows = [];
    const skippedRows = [];

    rowIndices.forEach(rowIdx => {
      const type = String(data[rowIdx][verificationIndex]).toLowerCase();
      if (type === "corrected") correctedRows.push(rowIdx);
      else if (type === "skipped") skippedRows.push(rowIdx);
    });

    if (correctedRows.length > 0 && skippedRows.length > 0) {
      // Case 1: corrected and skipped
      const correctedRow = correctedRows[0];
      const skippedRow = skippedRows[0];

      const correctedDate = new Date(data[correctedRow][dateIndex]);
      const skippedDate = new Date(data[skippedRow][dateIndex]);

      if (correctedDate > skippedDate) {
        rowsToColor.green.push(correctedRow + 1);
        rowsToColor.red.push(skippedRow + 1);
      } else {
        rowsToColor.orange.push(correctedRow + 1);
        rowsToColor.orange.push(skippedRow + 1);
      }
    } else if (correctedRows.length > 1 && skippedRows.length === 0) {
      // Case 2: only multiple corrected entries
      // Sort by date descending
      const sorted = correctedRows.sort((a, b) => {
        const dateA = new Date(data[a][dateIndex]);
        const dateB = new Date(data[b][dateIndex]);
        return dateB - dateA;
      });

      rowsToColor.green.push(sorted[0] + 1); // most recent
      sorted.slice(1).forEach(idx => rowsToColor.red.push(idx + 1));
    } else {
      // Case 3: not a usable corrected/skipped pair
      rowIndices.forEach(idx => rowsToColor.orange.push(idx + 1));
    }
  }

  // --- Clear previous highlights (excluding header) ---
  sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getLastColumn()).setBackground(null);

  // --- Apply highlights ---
  rowsToColor.green.forEach(row => {
    sheet.getRange(row, 1, 1, sheet.getLastColumn()).setBackground("#b6d7a8");
  });
  rowsToColor.red.forEach(row => {
    sheet.getRange(row, 1, 1, sheet.getLastColumn()).setBackground("#f4cccc");
  });
  rowsToColor.orange.forEach(row => {
    sheet.getRange(row, 1, 1, sheet.getLastColumn()).setBackground("#fce5cd");
  });
}
