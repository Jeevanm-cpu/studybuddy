"""Day 1 sanity check: is Python native ARM64, and is the NPU visible?"""
import platform
import sys

print("Python:", sys.version.split()[0], "| machine:", platform.machine())
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print("ONNX Runtime:", ort.__version__)
    print("Providers:", providers)
    print("NPU (QNN) ready:", "QNNExecutionProvider" in providers)
except ImportError:
    print("onnxruntime not installed yet")
