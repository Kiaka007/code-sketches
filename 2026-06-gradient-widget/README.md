# Gradient Widget

An interactive Jupyter widget (built with `anywidget`/`ipywidgets`) that fills a grid with a smooth color gradient interpolated between four corner anchor points — top-left, top-right, bottom-left, bottom-right.

Each anchor point is an HSL tuple `(hue, saturation, lightness)`. Cell colors are computed by bilinear interpolation of the weighted corner values, converted through RGB for correct blending, then back to HSL for display. Click any cell to copy its HSL value to the clipboard.

## Why HSL?

Personal preference — it's easier to reason about and build color themes from HSL. The RGB round-trip can be swapped out for whatever color model you prefer.

## Dependencies

- `anywidget`, `traitlets`, `ipywidgets` — bridges JS and Python in Jupyter
- Python's built-in `colorsys` — HSL(HLS) ↔ RGB conversion

## Usage

Open `gradient-widget.ipynb` in Jupyter Lab and run the cells. Two example gradients are included ("Day" and "Night" palettes) — call `gradient_widget(TL, TR, BL, BR, dimensions)` with your own anchor points to generate others.
