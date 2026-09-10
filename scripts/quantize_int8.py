from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov


MODEL_XML = Path("yolo26n_openvino_model/yolo26n.xml")
CALIBRATION_DIR = Path("data/calibration")
OUTPUT_DIR = Path("models/yolo26n_int8")

IMAGE_SIZE = 640


def load_calibration_image(image_path: Path) -> np.ndarray:
    """
    Load one calibration image and prepare it for OpenVINO/NNCF.

    Returns:
        NumPy array in NCHW format:
        [1, 3, 640, 640]
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read calibration image: {image_path}"
        )

    # OpenCV loads images as BGR.
    # Convert to RGB to match the usual YOLO preprocessing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the model input size.
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert uint8 [0,255] -> float32 [0,1].
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW.
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension.
    image = np.expand_dims(image, axis=0)

    return image


def main():
    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"OpenVINO model not found: {MODEL_XML}"
        )

    if not CALIBRATION_DIR.exists():
        raise FileNotFoundError(
            f"Calibration directory not found: {CALIBRATION_DIR}"
        )

    calibration_images = sorted(
        list(CALIBRATION_DIR.glob("*.jpg"))
        + list(CALIBRATION_DIR.glob("*.jpeg"))
        + list(CALIBRATION_DIR.glob("*.png"))
    )

    if not calibration_images:
        raise RuntimeError(
            f"No calibration images found in {CALIBRATION_DIR}"
        )

    print("=" * 60)
    print("OPENVINO INT8 QUANTIZATION")
    print("=" * 60)
    print(f"Model: {MODEL_XML}")
    print(f"Calibration images: {len(calibration_images)}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    print()
    print("Loading OpenVINO model...")

    core = ov.Core()

    model = core.read_model(str(MODEL_XML))

    print("OpenVINO model loaded successfully.")

    print()
    print("Preparing calibration dataset...")

    calibration_data = []

    for image_path in calibration_images:
        image = load_calibration_image(image_path)
        calibration_data.append(image)

    print(
        f"Prepared {len(calibration_data)} calibration images."
    )

    # NNCF expects data that can be passed to the model.
    # The previous implementation passed pathlib.Path objects,
    # which caused the OpenVINO TypeError.
    dataset = nncf.Dataset(calibration_data)

    print()
    print("Starting INT8 quantization...")
    print("Statistics collection may take some time.")
    print()

    quantized_model = nncf.quantize(
        model,
        dataset,
        preset=nncf.QuantizationPreset.MIXED,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_xml = OUTPUT_DIR / "yolo26n_int8.xml"

    ov.save_model(
        quantized_model,
        str(output_xml),
    )

    print()
    print("=" * 60)
    print("INT8 QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"Saved model: {output_xml}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
