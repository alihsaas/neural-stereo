# NeuralStereo

I want to measure how JPEG compression affects the depth estimates from a pair of robot cameras.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## Dataset

Three scenes from the Middlebury 2014 stereo dataset (Motorcycle, Piano, Pipes)
at the official quarter resolution (about 740 x 490), from the MiddEval3
training release, with ground-truth disparity (`disp0GT.pfm`) and per-scene
calibration (`calib.txt`). The download script fetches only these scenes and
`data/` stays out of the repository. Quarter resolution keeps every run fast on
a laptop and is close to what a small robot camera would actually transmit.

Fetch the three scenes with `uv run python scripts/download_data.py`.

## Classical stereo

I use OpenCV StereoSGBM to match rectified grayscale images. I convert disparity to depth using the scene calibration and measure coverage, bad-pixel rate, and median depth error.

## Tests

Run `uv run pytest -q --basetemp=.pytest-tmp`.

## Compression study

Run `uv run python scripts/run_compression_study.py` to compare the PNG baseline with JPEG qualities 95, 80, 60, 40, and 20 on the same scenes.

## Stage 1: classical matching (SGBM)

OpenCV StereoSGBM on grayscale, one fixed configuration per scene
(`numDisparities` from the scene's `ndisp`, blockSize 5, MODE_SGBM_3WAY).
All 18 runs (`results/results.csv`):

| scene | cond | KB/pair | Mbps est. | coverage % | bad2px % | med err cm | common cm |
|---|---|---|---|---|---|---|---|
| Motorcycle | png | 1303 | 104.3 | 85.0 | 22.6 | 0.83 | 0.83 |
| Motorcycle | q95 | 348 | 27.8 | 85.0 | 22.8 | 0.86 | 0.86 |
| Motorcycle | q80 | 167 | 13.3 | 84.6 | 23.3 | 0.99 | 0.97 |
| Motorcycle | q60 | 112 | 9.0 | 84.1 | 24.5 | 1.14 | 1.13 |
| Motorcycle | q40 | 86 | 6.9 | 83.5 | 26.0 | 1.36 | 1.34 |
| Motorcycle | q20 | 57 | 4.6 | 81.7 | 30.1 | 1.96 | 1.92 |
| Piano | png | 1016 | 81.3 | 82.0 | 34.0 | 1.32 | 1.32 |
| Piano | q95 | 240 | 19.2 | 81.4 | 34.5 | 1.40 | 1.37 |
| Piano | q80 | 108 | 8.7 | 81.5 | 35.8 | 1.63 | 1.55 |
| Piano | q60 | 72 | 5.7 | 81.2 | 37.8 | 1.96 | 1.83 |
| Piano | q40 | 54 | 4.3 | 79.2 | 41.1 | 2.50 | 2.30 |
| Piano | q20 | 35 | 2.8 | 78.6 | 47.1 | 3.65 | 3.28 |
| Pipes | png | 1226 | 98.0 | 78.3 | 33.1 | 2.24 | 2.24 |
| Pipes | q95 | 304 | 24.3 | 78.1 | 33.5 | 2.29 | 2.25 |
| Pipes | q80 | 143 | 11.4 | 77.3 | 34.9 | 2.61 | 2.52 |
| Pipes | q60 | 96 | 7.7 | 77.1 | 36.2 | 3.00 | 2.85 |
| Pipes | q40 | 73 | 5.9 | 76.3 | 37.6 | 3.51 | 3.31 |
| Pipes | q20 | 48 | 3.9 | 74.8 | 42.0 | 4.55 | 4.18 |

![payload versus depth error](results/payload_vs_error.png)

Per-scene visual grids with a shared color scale per scene:
[Motorcycle](results/motorcycle_grid.png), [Piano](results/piano_grid.png),
[Pipes](results/pipes_grid.png).

Compression barely matters until the texture starts to go. Quality 95 cuts the
payload by about 4x and moves the error by at most 0.03 cm. The curve has a
knee around quality 40: from there down, error and coverage fall away faster.
**Piano was the most sensitive scene** (bad2px up 13.1 points png to q20,
common-pixel error 2.5x), because JPEG wipes out the faint texture on its flat
surfaces - the dark piano body, the wall, the lampshade - first. In the grids
the failures sit in flat low-texture regions (lost coverage, wrong patches),
at depth discontinuities (blocky artifacts shift the matched edge), and in the
far background, where at small disparity a fraction of a pixel of disparity
error is a lot of depth.
