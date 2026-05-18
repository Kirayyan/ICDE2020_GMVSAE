"""TensorFlow 1.x compatibility shim for TensorFlow 2.x runtimes."""
import tensorflow as _tf

keras = _tf.keras

if hasattr(_tf, "compat") and hasattr(_tf.compat, "v1"):
    tf = _tf.compat.v1
    tf.disable_v2_behavior()
else:
    tf = _tf

__all__ = ['tf', 'keras']
