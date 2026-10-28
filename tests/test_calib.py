from neural_stereo.calib import parse_calib

SAMPLE = """cam0=[1998.842 0 588.364; 0 1998.842 505.864; 0 0 1]
cam1=[1998.842 0 653.919; 0 1998.842 505.864; 0 0 1]
doffs=65.555
baseline=193.001
width=741
height=497
ndisp=70
isint=0
vmin=5
vmax=65
"""


def test_parses_all_fields() -> None:
    calib = parse_calib(SAMPLE)
    assert calib.focal_length_px == 1998.842
    assert calib.cx0 == 588.364
    assert calib.cy0 == 505.864
    assert calib.doffs == 65.555
    assert calib.baseline_mm == 193.001
    assert calib.width == 741
    assert calib.height == 497
    assert calib.ndisp == 70
