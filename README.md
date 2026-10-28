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
