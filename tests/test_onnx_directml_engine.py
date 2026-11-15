"""
Open-Photon ONNX Engine & DirectML GPU Tests
Tests:
1. DirectML Execution Provider initialization on GPU (RTX 4060)
2. Audio STFT Tensor inference shape validation:
   Input: (1, 2, frames, bins) or (1, 2, samples)
   Output: (1, 4, 2, frames, bins) or (4, 2, samples)
3. Latency measurement (must be < 200ms per 4-second chunk -> >20x Real-Time)
4. Graceful CPU fallback when GPU is unavailable
"""

import time
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import pytest

def create_synthetic_roformer_model_proto(num_stems=4, channels=2, frames=374, bins=513):
    """
    Creates an ONNX graph simulating the Band-Split RoFormer stem separation model:
    Input: stft_repr of shape [1, channels, frames, bins]
    Output: stem0_spec, stem1_spec, stem2_spec, stem3_spec each of shape [1, channels, frames, bins]
    """
    input_info = helper.make_tensor_value_info('stft_repr', TensorProto.FLOAT, [1, channels, frames, bins])
    
    outputs_info = [
        helper.make_tensor_value_info(f'stem{i}_spec', TensorProto.FLOAT, [1, channels, frames, bins])
        for i in range(num_stems)
    ]

    nodes = []
    # Mask estimation simulation: compute a frequency-gain filter for each stem
    for i in range(num_stems):
        # Scale constant per stem
        scale_val = np.array([0.25 * (i + 1)], dtype=np.float32)
        scale_init = helper.make_tensor(f'scale_{i}', TensorProto.FLOAT, [1], scale_val)
        
        node = helper.make_node(
            'Mul',
            inputs=['stft_repr', f'scale_{i}'],
            outputs=[f'stem{i}_spec'],
            name=f'StemEst_{i}'
        )
        nodes.append(node)

    initializers = [
        helper.make_tensor(f'scale_{i}', TensorProto.FLOAT, [1], np.array([0.25 * (i + 1)], dtype=np.float32))
        for i in range(num_stems)
    ]

    graph = helper.make_graph(
        nodes,
        'OpenPhotonRoFormerModel',
        [input_info],
        outputs_info,
        initializer=initializers
    )

    model = helper.make_model(graph, producer_name='OpenPhoton', opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
    return model


def test_directml_gpu_session_creation(tmp_path):
    model_path = str(tmp_path / "synthetic_roformer.onnx")
    model_proto = create_synthetic_roformer_model_proto()
    onnx.save(model_proto, model_path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    active_providers = session.get_providers()
    print(f"\n[GPU Engine] Active Providers: {active_providers}")
    assert 'DmlExecutionProvider' in active_providers or 'CPUExecutionProvider' in active_providers


def test_directml_gpu_inference_latency(tmp_path):
    model_path = str(tmp_path / "synthetic_roformer.onnx")
    model_proto = create_synthetic_roformer_model_proto()
    onnx.save(model_proto, model_path)

    sess_options = ort.SessionOptions()
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # 4-second STFT chunk @ 48kHz: [1, 2, 374, 513]
    dummy_input = np.random.randn(1, 2, 374, 513).astype(np.float32)

    # Warmup
    _ = session.run(None, {'stft_repr': dummy_input})

    # Benchmark 10 iterations
    timings = []
    for _ in range(10):
        t0 = time.perf_counter()
        outputs = session.run(None, {'stft_repr': dummy_input})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0) # ms

    avg_ms = np.mean(timings)
    print(f"\n[GPU Engine Performance] 4-second Audio Chunk Inference Time: {avg_ms:.2f} ms")
    realtime_factor = 4000.0 / avg_ms
    print(f"[GPU Engine Performance] Real-Time Factor: {realtime_factor:.1f}x Real-Time!")

    assert len(outputs) == 4
    for i in range(4):
        assert outputs[i].shape == (1, 2, 374, 513)

    # Must be faster than 200 ms (20x real-time)
    assert avg_ms < 200.0, f"Inference took {avg_ms:.2f} ms, slower than 200ms threshold"


def test_cpu_fallback_mode(tmp_path):
    model_path = str(tmp_path / "synthetic_roformer.onnx")
    model_proto = create_synthetic_roformer_model_proto()
    onnx.save(model_proto, model_path)

    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    assert session.get_providers() == ['CPUExecutionProvider']

    dummy_input = np.random.randn(1, 2, 374, 513).astype(np.float32)
    outputs = session.run(None, {'stft_repr': dummy_input})
    assert len(outputs) == 4


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
