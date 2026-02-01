import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import json

class OptimizedCreditModel:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def apply_pruning(self, amount=0.3):
        """Применение прунинга к модели"""
        model = torch.load(self.model_path)
        model.eval()
        
        # Прунинг весов
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        pruned_path = self.model_path.replace('.pth', '_pruned.pth')
        torch.save(model.state_dict(), pruned_path)
        
        # Сохранение размера модели
        original_size = self.get_model_size(self.model_path)
        pruned_size = self.get_model_size(pruned_path)
        
        print(f"Размер модели до прунинга: {original_size:.2f} MB")
        print(f"Размер модели после прунинга: {pruned_size:.2f} MB")
        print(f"Сжатие: {((original_size - pruned_size)/original_size)*100:.1f}%")
        
        return pruned_path
    
    def apply_quantization(self, onnx_path):
        """Квантизация ONNX модели"""
        # Dynamic quantization
        quantized_model_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(
            onnx_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8
        )
        
        # Benchmark производительности
        self.benchmark_models(onnx_path, quantized_model_path)
        
        return quantized_model_path
    
    def benchmark_models(self, original_path, optimized_path):
        """Сравнение производительности моделей"""
        test_data = np.random.randn(1000, 23).astype(np.float32)
        
        metrics = {}
        
        for name, path in [("Original", original_path), ("Optimized", optimized_path)]:
            session = ort.InferenceSession(path)
            inputs = {session.get_inputs()[0].name: test_data}
            
            # Inference time
            import time
            start = time.time()
            for _ in range(100):
                _ = session.run(None, inputs)
            inference_time = time.time() - start
            
            # Memory usage
            import os
            model_size = os.path.getsize(path) / (1024 * 1024)  # MB
            
            metrics[name] = {
                "inference_time": inference_time,
                "model_size_mb": model_size,
                "throughput": 1000 / inference_time
            }
        
        # Сохранение результатов
        with open('reports/optimization_benchmark.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics