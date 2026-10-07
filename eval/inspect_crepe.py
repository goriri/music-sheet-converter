import inspect
import sys
import torch
import numpy as np

try:
    import torchcrepe
    print(f"torchcrepe version: {getattr(torchcrepe, '__version__', 'unknown')}")
    print(f"torchcrepe file: {torchcrepe.__file__}")

    # Inspect torchcrepe.predict source code
    import inspect
    print("\n=== torchcrepe.predict source ===")
    print(inspect.getsource(torchcrepe.predict))

    # Inspect torchcrepe.decode.viterbi source code
    print("\n=== torchcrepe.decode.viterbi source ===")
    print(inspect.getsource(torchcrepe.decode.viterbi))

    # Inspect torchcrepe.decode.weighted_argmax source code
    print("\n=== torchcrepe.decode.weighted_argmax source ===")
    print(inspect.getsource(torchcrepe.decode.weighted_argmax))

except Exception as e:
    print(f"Error inspecting torchcrepe: {e}")
