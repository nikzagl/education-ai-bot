"""
Утилиты для обработки изображений
"""

import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import (
    TEMP_IMAGES_DIR, IMAGE_RESOLUTION, IMAGE_PADDING,
    IMAGE_WIDTH, COLOR_WHITE, COLOR_BLACK, LATEX_PREAMBLE
)
from latex_utils import (
    sanitize_latex_content, test_latex_compilation, 
    correct_latex_with_llm,
    render_latex_to_bytes, remove_emojis, extract_latex_content
)

from utils import llm

from aiogram.types import BufferedInputFile

try:
    from sympy import preview
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def process_image_with_padding(pil_image: Image.Image, padding: int = IMAGE_PADDING) -> Image.Image:
    """
    Обработать изображение: обрезать белые края и добавить padding
    
    Args:
        pil_image: PIL Image объект
        padding: Размер padding в пиксели
    
    Returns:
        Обработанное изображение
    """
    try:
        img = pil_image.convert('RGB')
        
        # Обрезаем белые края (артефакты)
        img_inverted = ImageOps.invert(img)
        bbox = img_inverted.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        w, h = img.size
        # Добавляем padding для лучшего вида
        padded = Image.new('RGB', (w + padding * 2, h + padding * 2), COLOR_WHITE)
        padded.paste(img, (padding, padding))
        
        return padded
    except Exception as e:
        logger.error(f"⚠️ Ошибка при обработке изображения: {e}")
        return pil_image


def bytes_to_buffered_file(image_bytes: bytes, filename: str = "image.png") -> BufferedInputFile:
    """
    Преобразует байты изображения в BufferedInputFile для Telegram
    
    Args:
        image_bytes: Байты PNG
        filename: Имя файла
    
    Returns:
        BufferedInputFile объект
    """
    return BufferedInputFile(image_bytes, filename=filename)


def latex_to_image_bytes(latex_content: str, resolution: int = IMAGE_RESOLUTION) -> Optional[bytes]:
    """
    Преобразует LaTeX в PNG изображение (байты)
    Убирает служебную информацию и эмодзи перед рендерингом
    
    Args:
        latex_content: LaTeX текст
        resolution: Разрешение в DPI
    
    Returns:
        Байты PNG или None
    """
    if not SYMPY_AVAILABLE or not latex_content:
        return None
    
    try:
        # Убираем служебную информацию: заголовки, эмодзи, цифры
        # Оставляем только math часть
        latex_content = remove_emojis(latex_content)
        latex_content = sanitize_latex_content(latex_content)
        logger.debug(f"🔍 Чистый LaTeX для рендеринга: {latex_content}")
        if not latex_content:
            return None
        
        obj = BytesIO()
        
        # Проверяем компиляцию сначала
        success, error_msg = test_latex_compilation(latex_content)
        if not success:
            logger.debug(f"⚠️ LaTeX содержит ошибки, пропускаю")
            return None
        else:
            logger.debug(f"✅ LaTeX успешно компилируется, рендерю...")
        # Рендерим
        preview(
            latex_content,
            preamble=LATEX_PREAMBLE,
            output='png',
            viewer='BytesIO',
            outputbuffer=obj,
            resolution=resolution,
            quiet=True
        )
        
        obj.seek(0)
        img_bytes = obj.read()
        
        if img_bytes:
            # Обработаем изображение
            img = Image.open(BytesIO(img_bytes))
            img = process_image_with_padding(img, IMAGE_PADDING)
            
            output_bytes = BytesIO()
            img.save(output_bytes, format='PNG')
            output_bytes.seek(0)
            return output_bytes.read()
        
        return img_bytes
    except Exception as e:
        logger.error(f"❌ Ошибка при преобразовании LaTeX в изображение: {e}")
        return None


def text_to_image(text: str, title: str = "", width: int = IMAGE_WIDTH, 
                  bg_color: tuple = COLOR_WHITE, text_color: tuple = COLOR_BLACK) -> Optional[str]:
    """
    Преобразует текст в изображение для отправки в Telegram
    
    Args:
        text: Текст для преобразования в изображение
        title: Заголовок изображения (опционально)
        width: Ширина изображения в пикселях
        bg_color: Цвет фона (RGB кортеж)
        text_color: Цвет текста (RGB кортеж)
    
    Returns:
        Путь к сохраненному изображению или None если ошибка
    """
    try:
        # Пытаемся загрузить шрифт
        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            text_font = ImageFont.truetype("arial.ttf", 20)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Оборачиваем текст
        lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            if len(test_line) > 80:
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Расчет высоты
        padding = 40
        line_height = 35
        title_height = 50 if title else 0
        height = padding * 2 + title_height + len(lines) * line_height + 20
        
        # Создаем изображение
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        y = padding
        
        # Добавляем заголовок
        if title:
            draw.text((padding, y), title, fill=text_color, font=title_font)
            y += title_height
        
        # Добавляем текст
        for line in lines:
            draw.text((padding, y), line, fill=text_color, font=text_font)
            y += line_height
        
        # Сохраняем изображение
        file_name = f"image_{uuid.uuid4().hex}.png"
        file_path = TEMP_IMAGES_DIR / file_name
        img.save(file_path, format='PNG')
        
        return str(file_path)
    
    except Exception as e:
        logger.error(f"❌ Ошибка при преобразовании текста в изображение: {e}")
        return None


def latex_document_to_image(response: str, llm_instance=None) -> Optional[BufferedInputFile]:
    """
    Преобразует LaTeX документ из ответа LLM в изображение
    
    Args:
        response: Полный ответ от LLM (может содержать LaTeX документ)
        llm_instance: Инстанс LLM для корректировки (опционально)
    
    Returns:
        BufferedInputFile объект для отправки в Telegram или None
    """
    if not response or not SYMPY_AVAILABLE:
        return None
    
    try:
        # Извлекаем содержимое LaTeX
        latex_content = extract_latex_content(response)
        
        if not latex_content:
            latex_content = remove_emojis(response)
        
        if not latex_content:
            return None
        
        latex_content = sanitize_latex_content(latex_content)
        
        # Проверяем компиляцию
        logger.debug(f"🔧 Корректировка LaTeX документа...")
        corrected = correct_latex_with_llm(latex_content, llm_instance)
        if corrected:
            latex_content = sanitize_latex_content(corrected)
        
        # Рендерим
        img_bytes = latex_to_image_bytes(latex_content, IMAGE_RESOLUTION)
        if img_bytes:
            return bytes_to_buffered_file(img_bytes, filename="document.png")
        
        return None
    
    except Exception as e:
        logger.error(f"❌ Ошибка при создании изображения из LaTeX: {e}")
        return None


def question_to_image(question_data: Dict[str, Any], topic: str = "", 
                      topic_index: int = 0, total_topics: int = 0) -> Optional[BufferedInputFile]:
    """
    Преобразует вопрос в красивое изображение с поддержкой LaTeX формул
    
    Args:
        question_data: Словарь с вопросом и вариантами ответа
        topic: Название темы
        topic_index: Номер текущего вопроса
        total_topics: Всего вопросов
    
    Returns:
        BufferedInputFile объект или None
    """
    if not question_data or "question" not in question_data:
        return None
    
    # Формируем ТОЛЬКО LaTeX часть (вопрос + варианты) БЕЗ заголовков и эмодзи
    latex_text = f"{question_data['question']}\n\n"
    
    for i, opt in enumerate(question_data.get("options", []), 1):
        latex_text += f"{i}. {opt}\n"
    
    # Преобразуем в изображение (служебная информация уберется в latex_to_image_bytes)
    return latex_document_to_image(latex_text, llm)



def task_to_image(task_text: str, topic: str = "", task_number: int = 0) -> Optional[BufferedInputFile]:
    """
    Преобразует задачу в красивое изображение с поддержкой LaTeX
    БЕЗ служебной информации (заголовки добавляются отдельно в bot.py)
    
    Args:
        task_text: Текст задачи (может содержать LaTeX формулы в $...$)
        topic: Название темы (опционально, НЕ добавляется в изображение)
        task_number: Номер задачи (опционально, НЕ добавляется в изображение)
    
    Returns:
        BufferedInputFile объект или None
    """
    # Преобразуем ТОЛЬКО содержимое в изображение БЕЗ заголовков
    # Служебная информация (тема, номер) уберется в latex_to_image_bytes
    return latex_document_to_image(task_text, llm)
