# KeenBench — Test Plan: Workbench Creation and File Management

## Test Data

### Bundled fixtures (in repo)

Base path: `/home/fkeegan/src/keenbench/`

Add the base path to the file location below to get the full path.

| File | Location | Description |
|------|----------|-------------|
| `notes.txt` | `app/integration_test/support/` | Staffing meeting notes (projects Atlas/Beacon, 5 employees, decisions, open items) |
| `data.csv` | `app/integration_test/support/` | Employee roster CSV (name, role, location, availability) with 5 rows |
| `simple.docx` | `engine/testdata/office/` | Simple Word document with paragraphs and headings |
| `multi-sheet.xlsx` | `engine/testdata/office/` | Excel workbook with multiple sheets |
| `slides.pptx` | `engine/testdata/office/` | PowerPoint with 2-3 slides |
| `report.pdf` | `engine/testdata/office/` | PDF report (read-only) |
| `notes.odt` | `engine/testdata/office/` | OpenDocument text document (read/write via office tool worker) |
| `chart.png` | `engine/testdata/office/` | PNG image (read-only) |
| `logo.svg` | `engine/testdata/office/` | SVG image (read-only) |
| `unknown.bin` | `engine/testdata/office/` | Unknown binary format (opaque) |

### Synthetic data (generated for tests)

| File | Purpose | Specification |
|------|---------|--------------|
| `big.csv` | Oversize rejection test | >25 MB, repeated rows |

## File Type Semantics

- **Read+Write:** txt, csv, md, json, xml, yaml, html, code files, docx, odt, xlsx, ods, pptx, odp
- **Read-only:** pdf, images (png, jpg, gif, webp, svg)
- **Opaque:** any other type (metadata only)

---

## Test Cases

### 3. Workbench Creation and File Management

#### TC-007: Create a new Workbench
- Priority: P0
- Preconditions: App on home screen.
- Steps:
  1. Click "New Workbench" (`AppKeys.homeNewWorkbenchButton`).
     Expected: A dialog appears (`AppKeys.newWorkbenchDialog`) with a text field (`AppKeys.newWorkbenchNameField`) and "Create" / "Cancel" buttons.
  2. Type "Financial Analysis" into the name field.
     Expected: The text "Financial Analysis" appears in the field.
  3. Click "Create" (`AppKeys.newWorkbenchCreateButton`).
     Expected: The dialog closes. The workbench screen (`AppKeys.workbenchScreen`) opens with title "Financial Analysis". The file list (`AppKeys.workbenchFileList`) is empty. The scope description text is visible (`AppKeys.workbenchScopeLimits`).
  4. Verify the composer field (`AppKeys.workbenchComposerField`) is visible.
     Expected: The composer is present and enabled. Placeholder is mode-dependent: "Describe a task..." in Agent mode, "Ask a question..." in Ask mode.

#### TC-008: Add text and CSV files
- Priority: P0
- Preconditions: Workbench "Financial Analysis" open, no Draft, no files.
- Steps:
  1. Click "Add files" and select `notes.txt` and `data.csv` (see base path + file location).
     Expected: Both files appear in the file list. `notes.txt` shows a file row (`AppKeys.workbenchFileRow('notes.txt')`) with a "TXT" badge. `data.csv` shows a row with a "CSV" badge.
  2. Verify both file rows have an extract button (`AppKeys.workbenchFileExtractButton('notes.txt')`) and a remove button (`AppKeys.workbenchFileRemoveButton('notes.txt')`).
     Expected: Both action buttons are visible and enabled (no draft exists).
  3. Verify the scope badge (`AppKeys.workbenchScopeBadge`) shows "Scoped" or the scope limits text updates to reflect the file count.
     Expected: The file count in the scope area reflects 2 files.

#### TC-009: Add office files (DOCX, ODT, XLSX, PPTX, PDF, images)
- Priority: P0
- Preconditions: Workbench open, no Draft.
- Steps:
  1. Add `simple.docx`, `notes.odt`, `multi-sheet.xlsx`, `slides.pptx`, `report.pdf`, `chart.png`, `logo.svg` via "Add files" (see base path + file location).
     Expected: All 7 files appear in the file list with appropriate badges: "DOCX", "ODT", "XLSX", "PPTX", "PDF", "PNG", "SVG".
  2. Verify `report.pdf`, `chart.png`, `logo.svg` show a "Read-only" badge on their file rows.
     Expected: Read-only files are visually distinguished.
  3. Verify the file count in scope limits reflects the correct total.
     Expected: File count shows the correct number (2 previous + 7 = 9, or per the test setup).

#### TC-011: File count limit (batch reject at 10)
- Priority: P0
- Preconditions: Workbench has 9 files already.
- Steps:
  1. Create two new temp files: `extra1.txt` and `extra2.txt`.
     Expected: Files exist on disk.
  2. Attempt to add both files via `WorkbenchFilesAdd` in a single call.
     Expected: The add operation fails with an error. The error message references the file count limit (10). No new files are added to the workbench. The file list still shows 9 files.

#### TC-012: Oversize file skip (>25 MB)
- Priority: P0
- Preconditions: Workbench open, no Draft, fewer than 10 files.
- Steps:
  1. Create `big.csv` (>25 MB, e.g., a repeated row to exceed the limit) and a small `small.txt` (a few bytes).
     Expected: Both files exist on disk.
  2. Attempt to add both `big.csv` and `small.txt` in a single `WorkbenchFilesAdd` call.
     Expected: `small.txt` is added successfully (appears in file list). `big.csv` is skipped. The add result includes a skip reason mentioning the size limit (25 MB). The file list does NOT contain `big.csv`.

#### TC-013: Duplicate filename rejection
- Priority: P0
- Preconditions: Workbench has `notes.txt` already added.
- Steps:
  1. Create a different file at a different path but also named `notes.txt` (e.g., `/tmp/test/notes.txt`).
     Expected: File exists on disk with different content than the workbench copy.
  2. Attempt to add this `notes.txt` via `WorkbenchFilesAdd`.
     Expected: The entire add batch is rejected with a duplicate-name error. No new files are added.

#### TC-014: Symlink rejection
- Priority: P1
- Preconditions: Workbench open, `notes.txt` exists on disk.
- Steps:
  1. Create a symlink `link_to_notes` pointing to `notes.txt`.
     Expected: Symlink exists.
  2. Attempt to add `link_to_notes` via `WorkbenchFilesAdd`.
     Expected: The add fails with an error mentioning "links not supported" or "symlink". No file is added.

#### TC-015: Remove file from workbench
- Priority: P0
- Preconditions: Workbench has `notes.txt` and `data.csv`, no Draft.
- Steps:
  1. Click the remove button (`AppKeys.workbenchFileRemoveButton('notes.txt')`).
     Expected: A confirmation dialog appears (`AppKeys.workbenchRemoveFileDialog`) with text 'Remove "notes.txt" from this Workbench? Originals remain untouched.'
  2. Click "Cancel" (`AppKeys.workbenchRemoveFileCancel`).
     Expected: The dialog closes. `notes.txt` is still in the file list.
  3. Click the remove button again.
     Expected: The confirmation dialog reappears.
  4. Click the red confirm button (`AppKeys.workbenchRemoveFileConfirm`).
     Expected: The dialog closes. `notes.txt` is removed from the file list. `data.csv` remains.

#### TC-016: Add/remove blocked while Draft exists
- Priority: P0
- Preconditions: Workbench has files, Draft exists.
- Steps:
  1. Verify the "Add files" button (`AppKeys.workbenchAddFilesButton`) is disabled.
     Expected: The button is grayed out or visually disabled. A tooltip says "Publish or discard the Draft" or similar.
  2. Verify the remove buttons on each file row are disabled.
     Expected: Each file's remove button (`AppKeys.workbenchFileRemoveButton(path)`) is disabled with a tooltip.
  3. Verify the extract buttons on each file row are disabled.
     Expected: Each file's extract button (`AppKeys.workbenchFileExtractButton(path)`) is disabled with a tooltip.

#### TC-017: Delete workbench from home screen
- Priority: P1
- Preconditions: Home screen with at least one workbench tile.
- Steps:
  1. Click the three-dot menu (`AppKeys.workbenchTileMenu(workbenchId)`) on a workbench tile.
     Expected: A popup menu appears with a "Delete Workbench" option.
  2. Click "Delete Workbench" (`AppKeys.workbenchTileDelete(workbenchId)`).
     Expected: A confirmation dialog appears (`AppKeys.homeDeleteWorkbenchDialog`) asking to confirm deletion.
  3. Click "Cancel" (`AppKeys.homeDeleteWorkbenchCancel`).
     Expected: The dialog closes. The workbench tile is still visible.
  4. Repeat steps 1-2, then click the red confirm button (`AppKeys.homeDeleteWorkbenchConfirm`).
     Expected: The dialog closes. The workbench tile is removed from the grid.

#### TC-018: Extract destination collision auto-renames output
- Priority: P1
- Preconditions: Workbench has `notes.txt`, no Draft, and a writable destination folder already containing `notes.txt`.
- Steps:
  1. Click extract on `notes.txt` (`AppKeys.workbenchFileExtractButton('notes.txt')`) and choose the destination folder that already has a file with the same name.
     Expected: Extraction succeeds instead of failing.
  2. Verify destination folder contents.
     Expected: A new file is created with collision suffix naming (`notes(1).txt`, then `notes(2).txt` on repeated extracts).
  3. Verify extraction feedback message in the app.
     Expected: The message indicates the file was extracted with the renamed path (for example, `Extracted "notes.txt" as "notes(1).txt".`).
