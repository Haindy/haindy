# Grid Zoom Implementation Plan

## Context
We discovered that the AI vision models have difficulty accurately identifying grid coordinates in a 60x60 grid overlay. After extensive testing, we found that:

1. **Problem**: The original yellow grid with semi-transparent cells was hard to see
2. **First improvement**: We implemented edge-only grid with pixel inversion for contrast
3. **Issue**: External labels (outside the image) caused confusion - AI was consistently off by 2-3 columns
4. **Second improvement**: Moved labels back inside cells in checkerboard pattern
5. **Persistent issue**: AI still struggles with precise coordinate identification in dense grids
6. **Discovery**: AI can see elements but struggles with spatial mapping to correct grid cells
7. **Solution**: Label ALL cells (not just checkerboard) with labels in top-right corner for consistent reference

## Progress Tracking

### Phase 1: Validation & Fine-tuning
| Task | Status | Notes |
|------|--------|-------|
| Create test script `test_zoom_grid.py` | 🟢 Completed | |
| Test Wikipedia AI "History" link | 🟢 Completed | Success with 8x8 grid |
| Fix grid rendering issues | 🟢 Completed | Double inversion, missing lines |
| Add cell labels to all cells | 🟢 Completed | Top-left with outline text |
| Improve prompts | 🟢 Completed | "most of the element" approach |
| Implement two-image fine grid | 🟢 Completed | No-grid + grid with context |
| Test Wikipedia "debate" link | ✅ Completed | 100% success with 10 runs! |
| Fine-tune coarse reliability | ✅ Completed | Centroid prompt achieved 100% |
| Fix page rendering inconsistency | ✅ Completed | Cached screenshots solution |
| Debug test script | ✅ Completed | Medium reasoning effort optimal |
| Add repeat mode to test script | 🟢 Completed | Batch testing capability |
| Add expected cell parameters | 🟢 Completed | Auto pass/fail detection |
| AI self-debugging | 🟢 Completed | Identified root cause |
| Update prompt for centroid | 🟢 Completed | 80% success rate achieved |
| Implement image compression | 🟢 Completed | JPEG compression for API calls |
| Integrate Gemini 2.5 Flash | ✅ Completed | Alternative model for better accuracy |
| Test Gemini on debate link | ✅ Completed | 100% accuracy on both grids! |
| Optimize Gemini prompts | ✅ Completed | Simple prompts with centering reminder |
| Run full test suite with Gemini | 🔴 Not Started | Test remaining UI elements |
| Test radio buttons | 🔴 Not Started | |
| Test toggle switches | 🔴 Not Started | |
| Test dropdown menus | 🔴 Not Started | |
| Test small toolbar buttons | 🔴 Not Started | |
| Document results & metrics | 🔴 Not Started | |

### Phase 2: System Impact Analysis
| Task | Status | Notes |
|------|--------|-------|
| Map component dependencies | 🔴 Not Started | |
| Analyze reporting system | 🔴 Not Started | Screenshot storage issue |
| Review coordinate flow | 🔴 Not Started | |
| Document required changes | 🔴 Not Started | |

### Phase 3: Implementation
| Component | Status | Notes |
|-----------|--------|-------|
| Action Agent updates | 🔴 Not Started | Two-stage analysis |
| Browser Controller updates | 🔴 Not Started | Direct x,y coordinates |
| Type definitions | 🔴 Not Started | PixelCoordinate type |
| Grid system updates | 🔴 Not Started | |
| Reporting system fixes | 🔴 Not Started | Screenshot links broken |
| Image storage | 🔴 Not Started | Permanent storage needed |
| Integration tests | 🔴 Not Started | |
| Documentation updates | 🔴 Not Started | |

**Legend**: 🔴 Not Started | 🟡 In Progress | 🟢 Completed | ⚠️ Blocked

## Solution: Two-Stage Zoom Approach

### How it works:
1. **Stage 1 - Coarse Grid (16x16)**:
   - Apply a coarse 16x16 grid to the full screenshot
   - AI identifies which cell contains the target element
   - Track exact pixel bounds of each cell (e.g., C7 = pixels 240,405 to 360,472)

2. **Stage 2 - Fine Grid (8x8 on 4x scaled)**:
   - Crop the identified cell region (with small padding)
   - Scale the crop 4x larger using LANCZOS resampling
   - Apply an 8x8 grid to the scaled image
   - AI identifies precise location within the zoomed region
   - Calculate exact x,y coordinates using scaling math

3. **Final Output**:
   - Direct x,y pixel coordinates (e.g., x=275, y=424)
   - No need for grid cell identifiers
   - Can be used directly with Playwright: `await page.click(x=275, y=424)`

### Implementation Approach: Test-First Development

#### Phase 1: Validation & Fine-tuning

1. **Create Ad-hoc Test Script**:
   - Build standalone test script implementing the two-stage grid system
   - Use existing OpenAI client for validation
   - Test against real-world scenarios before integration
   
2. **Primary Test Case**:
   - URL: https://en.wikipedia.org/wiki/Artificial_intelligence
   - Target: "History" link in the table of contents (left sidebar)
   - Current system failure: Cannot accurately identify this small link
   
3. **Extended Test Suite**:
   - Radio buttons
   - Toggle switches
   - Dropdown menus
   - Small buttons in toolbars
   - Links within dense text
   - Stacked navigation items
   
4. **Success Criteria**:
   - Consistent accurate identification across all test cases
   - Performance metrics acceptable (response time, API costs)
   - Only proceed to implementation after validation success

#### Phase 2: System Impact Analysis

1. **Architecture Review**:
   - Identify all components that interact with grid system
   - Map dependencies and data flow
   - Document required changes
   
2. **Component Updates Required**:
   - **Action Agent** (`src/agents/action_agent.py`)
   - **Browser Controller** (`src/browser/controller.py`)
   - **Grid System** (`src/grid/`)
   - **Types** (`src/core/types.py`)
   - **Reporting System** (`src/monitoring/reporter.py`)
   - **Execution Journaling** (`src/monitoring/`)
   - **Test Runner Agent** (coordinate handling)
   
3. **Critical Issues to Address**:
   - **Broken Screenshot Links**: Action screenshots not displaying in reports
   - **Image Storage**: Implement permanent storage for all action images
   - **Report Integration**: Ensure screenshots appear correctly within action steps

#### Phase 3: Implementation

1. **Update Action Agent** (`src/agents/action_agent.py`):
   - Implement two-stage screenshot analysis
   - First pass: coarse grid for general location
   - Second pass: zoomed fine grid for precise coordinates
   - Return x,y coordinates instead of grid cells

2. **Update Browser Controller** (`src/browser/controller.py`):
   - Modify click methods to accept x,y coordinates directly
   - Remove dependency on GridCoordinate conversion

3. **Update Types** (`src/core/types.py`):
   - Add new types for zoom approach
   - Create `PixelCoordinate(x: int, y: int)` type

4. **Update Reporting System**:
   - Fix screenshot storage and linking
   - Update HTML report templates for new coordinate system
   - Ensure all action images are preserved and accessible

5. **Testing**:
   - Integration tests with updated components
   - Regression testing on existing test scenarios
   - Performance benchmarking

### Key Code Snippets:

```python
# Coarse grid identification
COARSE_GRID_SIZE = 16
coarse_grid = GridOverlay(grid_size=COARSE_GRID_SIZE)

# Calculate cell bounds
col_idx, row_idx = coarse_grid._parse_cell_identifier(coarse_cell)
x1 = int(col_idx * cell_width)
y1 = int(row_idx * cell_height)

# Scale and apply fine grid
SCALE_FACTOR = 4
FINE_GRID_SIZE = 8
img_scaled = img_cropped.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

# Convert back to original coordinates
original_x = crop_x1 + (scaled_x / SCALE_FACTOR)
original_y = crop_y1 + (scaled_y / SCALE_FACTOR)
```

### Performance Considerations:
- Two AI API calls instead of one
- Additional image processing (crop, scale)
- But much higher accuracy - worth the tradeoff

### Configuration:
- Coarse grid: 7x7 (current best - balances cell size vs reliability)
- Fine grid: 8x8 (applied to single zoomed cell for precision)
- Scale factor: 4x (makes details clear without excessive size)
- Padding: 0 (zoom only the exact identified cell)
- Labels: All cells labeled in top-left with outline text

This approach has been tested and shows significant improvement in coordinate accuracy!

## Current Status & Findings (Updated 2025-07-18)

### Test Results Summary
- **Tests Completed**: 5 out of 7 UI element types with OpenAI o4-mini
- **Average Success Rate**: 86% with OpenAI o4-mini
- **Gemini 2.5 Flash Integration**: ✅ 100% accuracy on Wikipedia debate link test
- **Optimal Configuration**: 7x7 coarse grid, 8x8 fine grid, medium reasoning effort
- **Key Achievement**: Gemini shows superior accuracy with optimized prompts
- **Major Breakthrough**: Achieved perfect 100% accuracy (10/10 runs) on both coarse and fine grids with Gemini

### Gemini 2.5 Flash Breakthrough
- **Model Selection**: Added `--model gemini` option to test script
- **Prompt Optimization**: Simple, direct prompts with centering reminder
- **Perfect Accuracy**: 100% on both coarse (B5) and fine (F3) grids (10/10 runs each)
- **Key Insight**: "Keep in mind that I want to hit the center of the element" is crucial
- **Implementation Details**:
  - Created `src/models/gemini_client.py` wrapper for API compatibility
  - Added fast iteration options: `--use-existing-screenshot` and `--test-stage`
  - Optimized prompts through iterative testing
  - Fixed expected cells for debate test (E3/F3, not E2/F2)

### Detailed Test Progress
For comprehensive test results and findings, see: [`GRID_TEST_PROGRESS.md`](./GRID_TEST_PROGRESS.md)

This includes:
- Individual test case results (Wikipedia, Google, GitHub, eBay, YouTube)
- Success rates and cell distribution analysis
- Key discoveries and improvements
- Pending tests (radio buttons, toggles, dropdowns, etc.)


### Next Steps
1. **Fine-tune Coarse Grid Reliability**:
   - Iterate on prompt engineering
   - Test different grid sizes (7x7 vs alternatives)
   - Manual evaluation cycle: modify → test → evaluate
   
2. **Add Repeat Mode**: ✅ COMPLETED
   - Batch testing capability (e.g., run 5 tests at once)
   - Aggregate results showing coarse/fine cells
   - Faster iteration cycles

3. **Add Expected Cell Parameters**: ✅ COMPLETED
   - Define expected coarse/fine cells in test cases
   - Automatic pass/fail detection
   - Skip fine grid if coarse is wrong (guaranteed failure)
   - Remove manual evaluation burden

4. **AI Self-Debugging for Failed Tests**: ✅ COMPLETED
   - When coarse grid selection is incorrect, ask follow-up questions
   - Tell AI it selected wrong cell and provide correct answer
   - Ask: "Why did you fail to select the right cell?"
   - Ask: "What is the most important change necessary to find the correct cell?"
   - Collect and analyze AI's self-reported failure reasons
   - Use insights to improve prompts iteratively

5. **Image Compression for API Optimization**: ✅ COMPLETED
   - Convert PNG screenshots to JPEG before API calls
   - Target: Reduce 256KB PNGs to 80-90KB JPEGs (65%+ reduction)
   - Benefits:
     - Faster API response times
     - Lower bandwidth costs
     - Better scalability for high-resolution screens
   - Implementation:
     - Apply compression after grid overlay
     - Use high quality (85-95%) to preserve grid lines and text
     - Compress all images (disk and API) for consistency
   - Quick win: Can be implemented during current testing phase
   - **Results with 80% JPEG quality**:
     - Coarse grid: 419KB → 247KB (41% reduction)
     - Base screenshot: 402KB → 201KB (50% reduction)
     - Fine grid: 97KB → 64KB (34% reduction)
     - AI recognition still 100% successful

### Test Script Structure

The ad-hoc test script (`test_zoom_grid.py`) should:

1. **Setup**:
   - Initialize Playwright browser
   - Load OpenAI client
   - Define test cases with expected elements

2. **Test Flow**:
   ```python
   # For each test case:
   # 1. Navigate to test URL
   # 2. Take screenshot
   # 3. Apply coarse grid (16x16)
   # 4. Call AI to identify general area
   # 5. Crop and zoom identified cell
   # 6. Apply fine grid (8x8)
   # 7. Call AI for precise location
   # 8. Calculate final coordinates
   # 9. Attempt click and verify success
   # 10. Log results and accuracy metrics
   ```

3. **Metrics to Track**:
   - Success rate per element type
   - Average time per identification
   - API token usage
   - Coordinate accuracy (distance from actual element)

4. **Output**:
   - Detailed log of each attempt
   - Screenshots with grid overlays at each stage
   - Summary statistics
   - Recommendations for parameter tuning
