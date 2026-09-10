from pathlib import Path

import openvino as ov


MODEL_PATH = Path("models/yolo26n_int8/yolo26n_int8.xml")


def main():
    print("=" * 60)
    print("OPENVINO INT8 MODEL INSPECTION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"INT8 model not found: {MODEL_PATH}"
        )

    core = ov.Core()

    print(f"Model: {MODEL_PATH}")
    print()

    model = core.read_model(str(MODEL_PATH))

    print("INPUTS")
    print("-" * 60)

    for index, tensor in enumerate(model.inputs):
        print(f"Input index : {index}")
        print(f"Shape       : {tensor.get_partial_shape()}")
        print(f"Element type: {tensor.get_element_type()}")

        try:
            print(f"Name        : {tensor.get_any_name()}")
        except RuntimeError:
            print("Name        : <unnamed>")

        print()

    print("OUTPUTS")
    print("-" * 60)

    for index, tensor in enumerate(model.outputs):
        print(f"Output index: {index}")
        print(f"Shape       : {tensor.get_partial_shape()}")
        print(f"Element type: {tensor.get_element_type()}")

        try:
            print(f"Name        : {tensor.get_any_name()}")
        except RuntimeError:
            print("Name        : <unnamed>")

        print()

    print("=" * 60)
    print("MODEL INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
