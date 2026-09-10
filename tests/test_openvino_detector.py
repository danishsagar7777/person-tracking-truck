from pathlib import Path

import pytest

from src.detection.openvino_detector import OpenVINOPersonDetector


MODEL_PATH = "yolo26n_openvino_model"


def test_openvino_model_directory_exists():
    if not Path(MODEL_PATH).exists():
        pytest.skip(
            "OpenVINO model has not been exported yet."
        )

    assert Path(MODEL_PATH).is_dir()


def test_openvino_detector_initialization():
    if not Path(MODEL_PATH).exists():
        pytest.skip(
            "OpenVINO model has not been exported yet."
        )

    detector = OpenVINOPersonDetector(
        model_path=MODEL_PATH,
        image_size=640,
        confidence=0.25,
        iou=0.45,
    )

    assert detector.image_size == 640
    assert detector.confidence == 0.25
    assert detector.iou == 0.45
    assert detector.PERSON_CLASS_ID == 0
