"""
Конфигурация и константы приложения
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ========== TELEGRAM БОТА ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ========== ПАРАМЕТРЫ ТЕСТИРОВАНИЯ ==========
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_TEST", 3))
MAX_PRACTICE = int(os.getenv("MAX_PRACTICE_TASKS", 3))

# ========== LLM ПАРАМЕТРЫ ==========
LLM_MODEL = "mistral-large-latest"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# ========== LaTeX PREAMBLE ==========
LATEX_PREAMBLE = r"""
\documentclass[margin=0cm,varwidth]{standalone}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage[english,russian]{babel}
\usepackage{color}
\pagestyle{empty}
\begin{document}
""".strip()

# ========== ИЗОБРАЖЕНИЯ ==========
IMAGE_RESOLUTION = 150
IMAGE_PADDING = 20
IMAGE_WIDTH = 800
TEMP_IMAGES_DIR = Path("temp_images")
TEMP_IMAGES_DIR.mkdir(exist_ok=True)

# ========== ЦВЕТА ==========
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ========== БАЗА ЗНАНИЙ ==========
KB_BASE_PATH = "knowledge_base"

# ========== LaTeX CORRECTION ==========
LATEX_CORRECTION_MAX_ATTEMPTS = 3

# ========== ЛИМИТЫ ==========
USER_TEST_HISTORY_LIMIT = 20
