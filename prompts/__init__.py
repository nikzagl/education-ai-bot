# Этот файл делает папку prompts модулем Python
import os
from pathlib import Path

# Получаем путь к папке с промптами
PROMPTS_DIR = Path(__file__).parent

def load_prompt(filename: str) -> str:
    """Загружает промпт из файла"""
    file_path = PROMPTS_DIR / filename
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ Промпт {filename} не найден, создаю пустой файл")
        # Создаем пустой файл, чтобы не было ошибки
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("")
        return ""

# Загружаем все промпты при импорте
QUESTION_PROMPT = load_prompt("question_generation.txt")
TASK_PROMPT = load_prompt("task_generation.txt")
TASK_TRAINING_PROMPT = load_prompt("task_training.txt")
SOLUTION_GENERATION_PROMPT = load_prompt("solution_generation.txt")
LATEX_CORRECTION_PROMPT = load_prompt("latex_correction.txt")
SYSTEM_PROMPT = load_prompt("system.txt")