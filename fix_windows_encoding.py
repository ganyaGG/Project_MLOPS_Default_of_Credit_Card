#!/usr/bin/env python
# fix_windows_encoding.py
import os
import sys
from pathlib import Path

def detect_bom(filepath):
    """Определить BOM в файле"""
    bom_types = {
        b'\xff\xfe': 'UTF-16 LE',
        b'\xfe\xff': 'UTF-16 BE', 
        b'\xef\xbb\xbf': 'UTF-8 BOM',
        b'\x00\x00\xfe\xff': 'UTF-32 BE',
        b'\xff\xfe\x00\x00': 'UTF-32 LE'
    }
    
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            
        for bom, encoding in bom_types.items():
            if header.startswith(bom):
                return encoding, bom
                
        # Проверим, это UTF-16 без BOM?
        if len(header) >= 2 and header[1] == 0 and header[0] != 0:
            return 'UTF-16 LE (без BOM?)', None
            
        return 'UTF-8 (без BOM)', None
        
    except Exception as e:
        return f'Ошибка: {e}', None

def convert_to_utf8_no_bom(filepath):
    """Конвертировать файл в UTF-8 без BOM"""
    try:
        # Пробуем прочитать с разными кодировками
        encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 
                           'utf-16-be', 'cp1251', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Записываем в UTF-8 без BOM
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ {filepath}: конвертирован из {encoding}")
                return True
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️  {filepath}: ошибка с {encoding} - {e}")
                continue
        
        print(f"❌ {filepath}: не удалось определить кодировку")
        return False
        
    except Exception as e:
        print(f"❌ {filepath}: общая ошибка - {e}")
        return False

def main():
    print("🔧 Исправление проблем с кодировкой в Windows")
    print("=" * 60)
    
    # Список файлов для проверки
    files_to_check = [
        '.gitignore',
        '.dvcignore',
        'requirements.txt',
        'pyproject.toml',
        'dvc.yaml',
        'params.yaml',
        'run_api.py',
        'run_pipeline.py',
        'test_prediction.py',
        'test_simple.py'
    ]
    
    # Добавляем все Python файлы из src
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                files_to_check.append(os.path.join(root, file))
    
    # Проверяем и исправляем
    fixed_count = 0
    problem_files = []
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            encoding, bom = detect_bom(filepath)
            
            if 'UTF-16' in encoding or 'BOM' in encoding:
                print(f"\n⚠️  Проблема: {filepath}")
                print(f"   Обнаружена кодировка: {encoding}")
                
                if convert_to_utf8_no_bom(filepath):
                    fixed_count += 1
                else:
                    problem_files.append(filepath)
    
    # Отчет
    print("\n" + "=" * 60)
    print(f"📊 Результаты:")
    print(f"✅ Исправлено файлов: {fixed_count}")
    
    if problem_files:
        print(f"❌ Проблемные файлы ({len(problem_files)}):")
        for file in problem_files:
            print(f"   - {file}")
    
    # Проверка black
    print("\n🔍 Проверка black...")
    os.system("python -m black --check src/")
    
    # Создаем новый .gitignore если не существует
    if not os.path.exists('.gitignore'):
        print("\n📝 Создаю новый .gitignore...")
        create_gitignore()

def create_gitignore():
    """Создать новый .gitignore"""
    gitignore_content = """# Данные
data/raw/
data/processed/
data/expectations/

# Модели
models/
*.joblib
*.pkl
*.h5

# MLflow
mlruns/
mlflow/

# Логи и кеши
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb

# DVC
.dvc/
.dvcignore

# Python
*.egg-info/
dist/
build/

# Тесты
.coverage
htmlcov/
.pytest_cache/

# Операционные системы
.DS_Store
Thumbs.db

# Окружение
.env
.env.local
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Создан новый .gitignore в кодировке UTF-8")

if __name__ == "__main__":
    main()