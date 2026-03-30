# GIMP Color Adjustment Demo -- Action Log

Image: `/home/fkeegan/Pictures/Samples/eduard-pretsi-mu9NgVgJeTM-unsplash.jpg` (Big Ben, Unsplash)

Goal: Convert to B&W, then selectively bring back the gold of the tower using layers + mask.

CU Provider: Google (OpenAI gpt-5.4 can't accurately target GIMP menu items at 1080p)

## Successful actions

| # | Action instruction | Notes |
|---|---|---|
| 1 | `click on the GIMP icon in the left taskbar to bring GIMP to the foreground` | |
| 2 | `click the word File in the top menu bar that reads File Edit Select View Image Layer Colors Tools Filters Windows Help` | |
| 3 | `click Open... in the File menu` | |
| 4 | `click on Pictures in the left sidebar` | |
| 5 | `click on the Samples folder` | Double-click doesn't work (uinput bug). Single click + Enter instead. |
| 5b | `press Enter to open the selected Samples folder` | |
| 6 | `click on the image file eduard-pretsi in the file list` | |
| 7 | `click the Open button in the bottom right of the dialog` | |
| 8 | `click the Keep button in the color profile dialog` | GIMP asks about embedded profile on open. |
| 9 | `click on Colors in the GIMP menu bar, which is the second row from the top reading File Edit Select View Image Layer Colors Tools Filters Windows Help` | |
| 10 | `click on the menu item Desaturate which is between Components and Map in the currently open dropdown menu` | Key: give spatial context ("between X and Y") for submenu items. |
| 11 | `click on Desaturate... in the submenu that is currently showing` | |
| 12 | `click the OK button in the Desaturate dialog` | Luminance mode, default. |
| 13 | `press Ctrl+Z to undo the desaturation` | |
| 14 | `press Ctrl+Shift+D to duplicate the current layer` | |
| 15 | Repeat steps 9-12 to desaturate the duplicate layer | |
| 16 | `click on Layer in the GIMP menu bar, which is the second row from the top reading File Edit Select View Image Layer Colors Tools Filters Windows Help` | |
| 17 | `click on the menu item Mask which is between Stack and Transparency in the currently open dropdown menu` | |
| 18 | `click on Add Layer Masks... in the submenu that is currently showing` | |
| 19 | `click the Add button in the Add Layer Mask dialog` | White (full opacity) is the default. |
| 20 | `press D to reset the foreground color to black and background to white` | |
| 21 | `press P to select the paintbrush tool` | |
| 22 | `in the tool options on the left, change the brush size to 100 by clicking on the Size field and typing 100` | |
| 23 | `paint with black over the Big Ben tower in the center of the image by clicking and dragging from the top of the tower down to the bottom` | Multiple passes needed for full coverage. |

## Known issues

- **Double-click not recognized**: uinput double-click (click_count=2) doesn't register with GTK file chooser. Workaround: single click + Enter.
- **Brush coverage**: Single drag pass is narrow. Need multiple passes or larger brush for full tower coverage.
- **Menu targeting**: Give spatial context for dropdown items ("between X and Y"). Without it, CU models target wrong coordinates.

## Bugs fixed during rehearsal

- **Interpolated mouse movement**: uinput was teleporting the cursor via ABS events. GTK submenu hover requires continuous motion. Fixed in `virtual_input.py` by interpolating from last known position to target.
- **Debug logging**: Added click/move dispatch logging to `action_mixin.py` and `virtual_input.py` for debugging coordinate targeting issues.
