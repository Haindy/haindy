# Phase 2: Browser & Grid System - Visual Interaction Framework

## Phase Overview

**Tasks**: Playwright wrapper, adaptive grid overlay, coordinate mapping with refinement.

**ETA**: 2-3 days

**Status**: Completed

## Objectives

This phase implements the browser automation layer and the innovative adaptive grid system that enables DOM-free visual interaction with web applications. This is a core differentiator of HAINDY's approach.

## Key Deliverables

1. **Playwright Browser Wrapper (`src/browser/driver.py`)**
   - Chromium browser initialization and management
   - WebSocket/CDP communication setup
   - Screenshot capture with consistent quality
   - Absolute coordinate clicking (not DOM-based)
   - Page navigation and state management
   - Wait strategies and timing controls
   - Browser lifecycle management

2. **Adaptive Grid Overlay System (`src/grid/overlay.py`)**
   - 60×60 base grid implementation (configurable)
   - Grid cell labeling system (A1-Z60, AA1-AZ60, etc.)
   - Visual overlay generation for debugging
   - Semi-transparent grid lines for development mode
   - Production mode with invisible grid calculations
   - Dynamic grid scaling based on viewport

3. **Coordinate Mapping System (`src/grid/coordinator.py`)**
   - Grid cell to pixel coordinate conversion
   - Support for fractional coordinates within cells
   - Viewport-aware coordinate calculation
   - Resolution-independent mapping
   - Coordinate validation and bounds checking

4. **Adaptive Refinement Engine (`src/grid/refinement.py`)**
   - Confidence-based refinement triggers
   - 3×3 cell cropping for zoom-in analysis
   - Sub-grid precision targeting
   - Iterative refinement until confidence threshold
   - Refinement history tracking

5. **Browser Controller (`src/browser/controller.py`)**
   - High-level browser operations
   - Action execution with grid coordinates
   - Screenshot management
   - State synchronization
   - Error recovery mechanisms

## Technical Details

### Grid System Architecture

The grid system implements a novel DOM-free interaction approach:

```python
class AdaptiveGrid:
    def __init__(self, width: int = 60, height: int = 60):
        self.grid_width = width
        self.grid_height = height
        self.refinement_threshold = 0.80
    
    def map_to_coordinates(self, cell: str) -> PixelCoordinates:
        """Convert grid cell (e.g., 'M23') to pixel coordinates"""
        pass
    
    def refine_selection(self, 
                        initial_cell: str, 
                        screenshot: bytes, 
                        confidence: float) -> RefinedCoordinates:
        """Apply adaptive refinement for higher precision"""
        pass
```

### Refinement Workflow Example

```
Initial Analysis (60×60 grid):
- Target: "Add to Cart" button
- Initial selection: M23 (confidence: 70%)

Refinement Process:
1. Crop 3×3 region around M23
2. Re-analyze at higher resolution
3. Determine precise offset within M23
4. Final coordinates: M23 + (0.7, 0.4)
5. Confidence increased to 95%
```

### Browser Integration

```python
class PlaywrightDriver:
    async def click_at_grid(self, 
                           grid_coords: GridCoordinates,
                           wait_after: int = 1000):
        """Execute click at grid coordinates"""
        pixel_coords = self.grid.map_to_coordinates(grid_coords)
        await self.page.mouse.click(pixel_coords.x, pixel_coords.y)
        await self.page.wait_for_timeout(wait_after)
```

## Key Features

1. **DOM-Free Interaction**
   - No dependency on HTML structure
   - Works across any web framework
   - Resilient to UI changes
   - Platform-agnostic approach

2. **Adaptive Precision**
   - Start with coarse grid for speed
   - Refine only when needed
   - Balance between performance and accuracy
   - Confidence-driven refinement

3. **Visual Debugging**
   - Development mode with visible grid
   - Screenshot annotations
   - Coordinate tracking
   - Action replay capability

## Integration Points

- **Action Agent**: Uses grid system for coordinate determination
- **Test Runner**: Executes browser actions through the driver
- **Evaluator Agent**: Captures screenshots for validation
- **Monitoring**: Logs all browser interactions and grid calculations

## Success Criteria

- Browser automation working reliably across different sites
- Grid system accurately maps UI elements
- Refinement improves accuracy to >90% confidence
- No DOM selectors used anywhere in the system
- Visual debugging aids development

## Lessons Learned

- 60×60 grid provides good balance of precision and performance
- Adaptive refinement crucial for small UI elements
- WebSocket connection more reliable than HTTP for browser control
- Screenshot quality critical for AI analysis
- Coordinate systems must account for scrolling and viewport changes