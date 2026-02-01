import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
from sklearn.metrics import accuracy_score

# 1. Создаем нейронную сеть
class CreditScoringNN(nn.Module):
    def __init__(self, input_size=23, hidden_size=64, output_size=2):
        super(CreditScoringNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.layer3 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x

# 2. Обучение/загрузка модели
model = CreditScoringNN()
# model.load_state_dict(torch.load('models/trained/credit_nn_model.pth'))
model.eval()

# 3. Конвертация в ONNX
dummy_input = torch.randn(1, 23)
torch.onnx.export(
    model,
    dummy_input,
    "models/trained/credit_model.onnx",
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 4. Валидация конвертации
def validate_onnx_conversion():
    # Проверка формата
    onnx_model = onnx.load("models/trained/credit_model.onnx")
    onnx.checker.check_model(onnx_model)
    
    # Сравнение предсказаний
    ort_session = ort.InferenceSession("models/trained/credit_model.onnx")
    
    test_input = np.random.randn(10, 23).astype(np.float32)
    
    # PyTorch inference
    torch_output = model(torch.from_numpy(test_input)).detach().numpy()
    
    # ONNX Runtime inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Проверка совпадения
    mse = np.mean((torch_output - ort_output)**2)
    print(f"Validation MSE between PyTorch and ONNX: {mse:.6f}")
    
    if mse < 1e-5:
        print("✓ Конвертация прошла успешно")
    else:
        print("✗ Ошибка конвертации")
    
    return mse < 1e-5

# 5. Benchmark тесты
def benchmark_inference():
    test_data = np.random.randn(1000, 23).astype(np.float32)
    
    # PyTorch CPU
    start = time.time()
    for _ in range(100):
        _ = model(torch.from_numpy(test_data))
    torch_time = time.time() - start
    
    # ONNX CPU
    ort_session = ort.InferenceSession("models/trained/credit_model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: test_data}
    
    start = time.time()
    for _ in range(100):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = time.time() - start
    
    print(f"PyTorch CPU время: {torch_time:.2f}s")
    print(f"ONNX CPU время: {onnx_time:.2f}s")
    print(f"Ускорение ONNX: {torch_time/onnx_time:.2f}x")
    
    return {"pytorch_time": torch_time, "onnx_time": onnx_time}

if __name__ == "__main__":
    validate_onnx_conversion()
    benchmark_inference()