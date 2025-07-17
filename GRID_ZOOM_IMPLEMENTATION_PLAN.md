# Grid Zoom Implementation Plan

## Context
We discovered that the AI vision models have difficulty accurately identifying grid coordinates in a 60x60 grid overlay. After extensive testing, we found that:

1. **Problem**: The original yellow grid with semi-transparent cells was hard to see
2. **First improvement**: We implemented edge-only grid with pixel inversion for contrast
3. **Issue**: External labels (outside the image) caused confusion - AI was consistently off by 2-3 columns
4. **Second improvement**: Moved labels back inside cells in checkerboard pattern
5. **Persistent issue**: AI still struggles with precise coordinate identification in dense grids

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

### Implementation Files Created:
- `/home/fkeegan/src/haindy/test_zoom_grid_v2.py` - Working prototype of the zoom approach
- Grid overlay improvements in `src/grid/overlay.py`:
  - Pixel inversion for grid lines
  - Edge-only grid (no filled cells)
  - Configurable grid size
  - Internal checkerboard labels

### Next Steps:

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
   - Maybe create `PixelCoordinate(x: int, y: int)` type

4. **Testing**:
   - Test with various UI densities
   - Verify accuracy improvement over single-grid approach
   - Ensure performance is acceptable (2 AI calls vs 1)

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
- Coarse grid: 16x16 (good balance of coverage and cell size)
- Fine grid: 8x8 (readable on scaled image)
- Scale factor: 4x (makes details clear without excessive size)

This approach has been tested and shows significant improvement in coordinate accuracy!