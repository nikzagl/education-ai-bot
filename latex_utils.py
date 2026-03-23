"""
Утилиты для работы с LaTeX формулами и документами
"""

import re
import logging
import json
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image, ImageOps

from config import LATEX_PREAMBLE, LATEX_CORRECTION_MAX_ATTEMPTS, LLM_MODEL, LLM_MAX_TOKENS
from aiogram.types import BufferedInputFile

try:
    from sympy import preview
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)



def remove_emojis(text: str) -> str:
    """
    Удаляет эмодзи из текста для совместимости с LaTeX
    
    Args:
        text: Текст с потенциальными эмодзи
    
    Returns:
        Текст без эмодзи
    """
    import unicodedata
    
    # Удаляем символы категории "So" (Symbol, other), которые включают эмодзи
    cleaned = "".join(
        char for char in text
        if unicodedata.category(char) not in ('So', 'Co', 'Cn')
    )
    
    # Дополнительная очистка от известных эмодзи диапазонов
    cleaned = re.sub(r'[\U0001F300-\U0001F9FF]', '', cleaned)
    cleaned = re.sub(r'[\U0001F600-\U0001F64F]', '', cleaned)  # Smileys & Emotion
    cleaned = re.sub(r'[\U0001F300-\U0001F5FF]', '', cleaned)  # Symbols & Pictographs
    cleaned = re.sub(r'[\U0001F680-\U0001F6FF]', '', cleaned)  # Transport & Map
    cleaned = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', cleaned)  # Flags
    
    # Убираем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def sanitize_latex_content(latex_str: str) -> str:
    """
    Очищает LaTeX контент от множественных document declarations и пакетов
    Это предотвращает ошибки компиляции при объединении LaTeX блоков.
    
    КРИТИЧНО: вызывается ДО передачи в sympy.preview()
    
    Args:
        latex_str: Потенциально грязный LaTeX контент
    
    Returns:
        Очищенный LaTeX контент
    """
    if not latex_str:
        return ""
    
    content = latex_str.strip()
    
    # Удаляем LaTeX document declarations (документы не могут быть вложены)
    content = re.sub(r'\\documentclass(\[[^\]]*\])?\{[^}]*\}', '', content)
    content = re.sub(r'\\usepackage(\[[^\]]*\])?\{[^}]*\}', '', content)
    content = re.sub(r'\\begin\{document\}', '', content)
    content = re.sub(r'\\end\{document\}', '', content)
    content = re.sub(r'\\pagestyle\{[^}]*\}', '', content)
    
    # Удаляем эмодзи
    content = remove_emojis(content)
    
    # Убираем лишние пробелы и пустые строки
    content = re.sub(r'\n\s*\n', '\n', content)
    content = content.strip()
    
    return content


def test_latex_compilation(latex_content: str) -> Tuple[bool, str]:
    """
    Проверяет, может ли LaTeX быть скомпилирован.
    
    Args:
        latex_content: LaTeX текст для проверки
    
    Returns:
        Кортеж (успешность, сообщение об ошибке или пустая строка)
    """
    if not latex_content or not SYMPY_AVAILABLE:
        return False, "LaTeX содержимое пусто или sympy не установлен"
    
    try:
        test_obj = BytesIO()        
        logger.debug(f"🧪 Тестирование компиляции LaTeX...")
        
        preview(
            latex_content,
            preamble=LATEX_PREAMBLE,
            output='png',
            viewer='BytesIO',
            outputbuffer=test_obj,
            resolution=100,  # Низкое разрешение для быстрой проверки
            quiet=True
        )
        
        test_obj.seek(0)
        if test_obj.tell() > 0 or len(test_obj.getvalue()) > 0:
            logger.debug(f"✅ LaTeX успешно скомпилирован")
            return True, ""
        else:
            return False, "Ошибка компиляции: пустой вывод"
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"⚠️ LaTeX компиляция не удалась: {error_msg}")
        return False, error_msg


def extract_latex_content(response: str) -> str:
    """
    Извлекает содержимое LaTeX документа из ответа LLM
    
    Args:
        response: Ответ от LLM, который может содержать LaTeX документ
    
    Returns:
        Чистое содержимое LaTeX без служебных элементов
    """
    latex_content = response
    
    # Удаляем ``` markdown блоки если есть
    if "```latex" in latex_content:
        match = re.search(r'```latex\n(.*?)\n```', latex_content, re.DOTALL)
        if match:
            latex_content = match.group(1)
    elif "```" in latex_content:
        match = re.search(r'```\n(.*?)\n```', latex_content, re.DOTALL)
        if match:
            latex_content = match.group(1)
    
    # Извлекаем содержимое между \begin{document} и \end{document}
    if "\\begin{document}" in latex_content:
        match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if match:
            latex_content = match.group(1).strip()
    
    return latex_content.strip()


def correct_latex_with_llm(latex_content: str, llm_instance=None, 
                           max_attempts: int = LATEX_CORRECTION_MAX_ATTEMPTS) -> Optional[str]:
    """
    Корректирует LaTeX текст через LLM перед рендерингом.
    
    Args:
        latex_content: LaTeX документ для корректировки
        llm_instance: Инстанс LLM для использования
        max_attempts: Максимальное количество попыток исправления
    
    Returns:
        Исправленный LaTeX текст или None при ошибке
    """
    if not latex_content or not llm_instance:
        return None
    
    # Импортируем здесь, чтобы избежать циклических импортов
    try:
        from prompts import LATEX_CORRECTION_PROMPT
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать промпты")
        return None
    
    current_latex = latex_content.strip()
    attempt = 0
    last_error = ""
    
    while attempt < max_attempts:
        attempt += 1
        logger.debug(f"🔧 Попытка корректировки {attempt}/{max_attempts}...")
        
        try:
            
            # Отправляем на корректировку через LLM
            corrected = llm_instance.ask(
                LATEX_CORRECTION_PROMPT,
                latex_content=current_latex,
                latex_error=last_error if last_error else "Нет информации об ошибках"
            )
            
            if not corrected:
                logger.warning(f"⚠️ Попытка {attempt}: LLM не вернул ответ")
                continue
            
            corrected = corrected.strip()
            corrected_content = corrected
            
            # Извлекаем содержимое без преамбулы если LLM вернул полный документ
            if "\\begin{document}" in corrected and "\\end{document}" in corrected:
                start_idx = corrected.find("\\begin{document}") + len("\\begin{document}")
                end_idx = corrected.find("\\end{document}")
                if start_idx < end_idx:
                    corrected_content = corrected[start_idx:end_idx].strip()
            
            # Проверяем компиляцию
            success, error_msg = test_latex_compilation(corrected_content)
            if success:
                logger.debug(f"✅ LaTeX успешно исправлен на попытке {attempt}")
                return corrected_content
            else:
                logger.warning(f"⚠️ Попытка {attempt}: LaTeX содержит ошибки, повторяю...")
                last_error = error_msg
                current_latex = corrected_content
                continue
                
        except Exception as e:
            logger.warning(f"⚠️ Попытка {attempt}: Ошибка при корректировке: {e}")
            last_error = str(e)
            continue
    
    logger.warning(f"⚠️ Не удалось исправить LaTeX после {max_attempts} попыток")
    return current_latex


def render_latex_to_bytes(latex_str: str, resolution: int = 150) -> Optional[bytes]:
    """
    Рендерит LaTeX строку в PNG через sympy
    
    Args:
        latex_str: LaTeX формула
        resolution: Разрешение в DPI
    
    Returns:
        Байты PNG изображения или None
    """
    if not SYMPY_AVAILABLE or not latex_str:
        return None
    
    try:
        # Очищаем LaTeX
        latex_str = sanitize_latex_content(latex_str.strip('$').strip())
        if not latex_str:
            return None
        
        obj = BytesIO()
        
        # Рендерим
        preview(
            latex_str,
            output='png',
            viewer='BytesIO',
            outputbuffer=obj,
            resolution=resolution,
            quiet=True
        )
        
        obj.seek(0)
        return obj.read()
    
    except Exception as e:
        logger.error(f"❌ Ошибка при рендеринге LaTeX: {e}")
        return None
