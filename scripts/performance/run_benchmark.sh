#!/bin/bash

echo "Запуск benchmark тестов..."

# CPU тесты
echo "=== CPU Benchmark ==="
python scripts/performance/benchmark_cpu.py

# GPU тесты (если доступно)
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Benchmark ==="
    python scripts/performance/benchmark_gpu.py
fi

# Нагрузочное тестирование
echo "=== Нагрузочное тестирование ==="
locust --headless -u 100 -r 10 --run-time 1m --csv=reports/load_test

echo "Benchmark завершен. Результаты в reports/"
