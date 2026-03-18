"""
Вспомогательные функции и классы
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from io import BytesIO
import tempfile
import uuid
import subprocess
import shutil

from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from mistralai.client import Mistral
from dotenv import load_dotenv
from aiogram.types import BufferedInputFile

try:
    import sympy
    from sympy import latex as sympy_latex
    from sympy.printing.latex import latex
    from sympy import preview
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ sympy не установлен, будут использоваться базовые изображения")

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ langchain не установлен, используется прямое API Mistral")

from prompts import (
    QUESTION_PROMPT, 
    TASK_PROMPT, 
    TEACHER_SYSTEM_PROMPT,
    TUTOR_SYSTEM_PROMPT
)

load_dotenv()

# Настройка логирования для отладки
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Создаем обработчик для вывода на консоль
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# ========== PYDANTIC MODELS FOR STRUCTURED OUTPUT ==========

class QuestionAnswer(BaseModel):
    """Модель для структурированного вопроса с вариантами ответов"""
    question: str = Field(..., description="Текст вопроса с LaTeX формулами")
    options: List[str] = Field(..., description="Список из 4 вариантов ответа")
    correct: int = Field(..., description="Индекс правильного ответа (0, 1, 2 или 3)")


class KnowledgeBase:
    """
    Класс для работы с базой знаний в JSON
    """
    
    def __init__(self, base_path: str = "knowledge_base"):
        self.base_path = Path(base_path)
        self.topics = self._load_json("topics.json")
        self.curriculum = self._load_json("curriculum.json")
        self.dependencies = self._load_json("dependencies.json")
        self.difficulty = self._load_json("difficulty.json")
    
    def _load_json(self, filename: str) -> Dict:
        """Загружает JSON-файл"""
        file_path = self.base_path / filename
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Файл {filename} не найден")
            return {}
        except json.JSONDecodeError:
            print(f"⚠️ Ошибка парсинга {filename}")
            return {}
    
    def get_topics(self, grade: int, subject: str) -> List[Dict]:
        """Возвращает список тем для класса и предмета"""
        try:
            return self.topics["grades"][str(grade)][subject]["topics"]
        except KeyError:
            return []
    
    def get_topic_names(self, grade: int, subject: str) -> List[str]:
        """Возвращает только названия тем"""
        topics = self.get_topics(grade, subject)
        return [t["name"] for t in topics]
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """Находит тему по ID"""
        for grade in self.topics.get("grades", {}).values():
            for subject in grade.values():
                for topic in subject.get("topics", []):
                    if topic.get("id") == topic_id:
                        return topic
        return None
    
    def get_topic_by_name(self, grade: int, subject: str, name: str) -> Optional[Dict]:
        """Находит тему по названию"""
        topics = self.get_topics(grade, subject)
        for topic in topics:
            if topic["name"] == name:
                return topic
        return None
    
    def get_prerequisites(self, topic_id: str) -> List[str]:
        """Возвращает ID тем, которые нужно знать перед данной"""
        return self.dependencies.get("prerequisites", {}).get(topic_id, [])
    
    def get_difficulty_level(self, level: int) -> Dict:
        """Возвращает описание уровня сложности"""
        return self.difficulty.get("levels", {}).get(str(level), {})
    
    def get_topics_by_difficulty(self, grade: int, subject: str, difficulty: int) -> List[Dict]:
        """Фильтрует темы по сложности"""
        topics = self.get_topics(grade, subject)
        return [t for t in topics if t.get("difficulty") == difficulty]
    
    def get_quarter_topics(self, grade: int, subject: str, quarter: str) -> List[str]:
        """Возвращает темы для конкретной четверти"""
        try:
            return self.curriculum["by_quarter"][str(grade)][subject][quarter]
        except KeyError:
            return []
    
    def get_learning_objectives(self, topic_id: str) -> List[str]:
        """Возвращает цели обучения по теме"""
        topic = self.get_topic_by_id(topic_id)
        return topic.get("learning_objectives", []) if topic else []
    
    def get_examples(self, topic_id: str) -> List[str]:
        """Возвращает примеры задач по теме"""
        topic = self.get_topic_by_id(topic_id)
        return topic.get("examples", []) if topic else []
    
    def get_common_mistakes(self, topic_id: str) -> List[str]:
        """Возвращает типичные ошибки по теме"""
        topic = self.get_topic_by_id(topic_id)
        return topic.get("common_mistakes", []) if topic else []
    
    def format_topic_info(self, topic: Dict) -> str:
        """Форматирует информацию о теме для промпта"""
        info = f"Тема: {topic['name']}\n"
        info += f"Описание: {topic.get('description', 'Нет описания')}\n"
        
        if topic.get('learning_objectives'):
            info += "Цели обучения:\n"
            for obj in topic['learning_objectives'][:3]:
                info += f"- {obj}\n"
        
        if topic.get('common_mistakes'):
            info += "Типичные ошибки:\n"
            for mistake in topic['common_mistakes'][:3]:
                info += f"- {mistake}\n"
        
        return info


class SimpleLLM:
    """
    Класс для работы с LLM через langchain и Mistral AI
    """
    
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("❌ MISTRAL_API_KEY не найден в переменных окружения")
        
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatMistralAI(
                model="mistral-large-latest",
                api_key=api_key,
                temperature=0.7,
                max_tokens=1000
            )
            self.use_langchain = True
        else:
            self.client = Mistral(api_key=api_key)
            self.use_langchain = False
        self.kb = KnowledgeBase()
    
    def ask(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Запрос к LLM с подстановкой параметров
        """
        # Подставляем параметры в промпт
        if kwargs:
            try:
                prompt = prompt.format(**kwargs)
            except KeyError as e:
                logger.error(f"Отсутствует параметр в промпте: {e}")
                return None
            except Exception as e:
                logger.error(f"Ошибка форматирования промпта: {e}")
                return None
        
        system = system_prompt or TEACHER_SYSTEM_PROMPT
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"📨 LLM REQUEST (ask)")
        logger.debug(f"{'='*80}")
        logger.debug(f"System Prompt:\n{system[:200]}...")
        logger.debug(f"User Prompt:\n{prompt[:300]}...")
        
        try:
            if self.use_langchain:
                logger.debug("Using LangChain with ChatMistralAI")
                # Создаем prompt template с системным промптом
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system),
                    ("user", "{input}")
                ])
                
                # Создаем цепь
                chain = prompt_template | self.llm
                
                # Выполняем запрос
                response = chain.invoke({"input": prompt})
                response_text = response.content if hasattr(response, 'content') else str(response)
                logger.debug(f"✅ Response received (LangChain)\n{response_text[:500]}...")
                return response_text
            else:
                logger.debug("Using direct Mistral API")
                # Используем прямое API Mistral
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content
                logger.debug(f"✅ Response received (Mistral API)\n{response_text[:500]}...")
                return response_text
        except Exception as e:
            logger.error(f"❌ LLM Error in ask(): {e}", exc_info=True)
            return None
    
    def ask_json(self, prompt: str, system_prompt: Optional[str] = None, pydantic_model: Optional[type] = None, **kwargs) -> Optional[Dict]:
        """
        Запрос с гарантией структурированного JSON-ответа используя Pydantic модели
        
        Args:
            prompt: Промпт для LLM
            system_prompt: Системный промпт
            pydantic_model: Pydantic модель для структурирования (по умолчанию QuestionAnswer)
            **kwargs: Параметры для форматирования промпта
        
        Returns:
            Словарь с распарсенными данными или None
        """
        # Подставляем параметры
        if kwargs:
            try:
                prompt = prompt.format(**kwargs)
            except KeyError as e:
                logger.error(f"Отсутствует параметр в промпте: {e}")
                return None
            except Exception as e:
                logger.error(f"Ошибка форматирования промпта: {e}")
                return None
        
        system = system_prompt or TEACHER_SYSTEM_PROMPT
        model = pydantic_model or QuestionAnswer
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"📨 LLM REQUEST (ask_json)")
        logger.debug(f"{'='*80}")
        logger.debug(f"Pydantic Model: {model.__name__}")
        logger.debug(f"System Prompt:\n{system[:200]}...")
        logger.debug(f"User Prompt:\n{prompt[:300]}...")
        
        try:
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                logger.debug("Using LangChain with structured output (Pydantic)")
                # Используем LangChain с Pydantic структурированием
                # Создаем LLM с поддержкой структурированного вывода
                structured_llm = self.llm.with_structured_output(model)
                
                # Создаем prompt template с системным промптом
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system),
                    ("user", "{input}")
                ])
                
                # Создаем цепь с Pydantic парсером
                chain = prompt_template | structured_llm
                
                try:
                    # Выполняем запрос с структурированным выводом
                    response = chain.invoke({"input": prompt})
                    logger.debug(f"Raw response type: {type(response)}")
                    
                    # Если response это объект Pydantic модели, преобразуем в dict
                    if isinstance(response, BaseModel):
                        result = response.model_dump()
                        logger.debug(f"✅ Pydantic validation successful\nResult: {result}")
                        return result
                    elif isinstance(response, dict):
                        logger.debug(f"✅ Dict response received\nResult: {response}")
                        return response
                    else:
                        logger.error(f"Неожиданный тип ответа: {type(response)}")
                        return None
                        
                except Exception as e:
                    logger.warning(f"LangChain structured output failed: {e}\nFalling back to JSON parsing...")
                    # Fallback на старый метод парсинга JSON
                    return self._fallback_json_parse(prompt, system)
            else:
                logger.debug("Using direct Mistral API with Pydantic validation")
                # Используем прямое API Mistral с Pydantic парсингом
                return self._mistral_direct_json(prompt, system, model)
                
        except Exception as e:
            logger.error(f"❌ LLM Error in ask_json(): {e}", exc_info=True)
            return None
    
    def _mistral_direct_json(self, prompt: str, system: str, model: type) -> Optional[Dict]:
        """
        Использует прямое Mistral API с Pydantic парсингом JSON
        """
        try:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"📨 Direct Mistral JSON Request")
            logger.debug(f"{'='*80}")
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
            logger.debug(f"📥 Raw response from Mistral:\n{response_text}")
            
            # Пытаемся распарсить JSON с Pydantic валидацией
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.debug(f"📦 Extracted JSON:\n{json_str}")
                json_data = json.loads(json_str)
            else:
                json_data = json.loads(response_text)
            
            logger.debug(f"✅ JSON parsed successfully")
            
            # Валидируем через Pydantic модель
            validated = model(**json_data)
            result = validated.model_dump()
            logger.debug(f"✅ Pydantic validation successful\nResult: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}\nResponse was: {response_text[:300] if 'response_text' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"❌ Error in _mistral_direct_json(): {e}", exc_info=True)
            return None
    
    def _fallback_json_parse(self, prompt: str, system: str) -> Optional[Dict]:
        """
        Fallback для парсинга JSON без Pydantic структурирования
        """
        try:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"📨 Fallback JSON Parse (no Pydantic)")
            logger.debug(f"{'='*80}")
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
            logger.debug(f"📥 Raw response:\n{response_text}")
            
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.debug(f"📦 Extracted JSON:\n{json_str}")
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)
            
            logger.debug(f"✅ JSON parsed successfully\nResult: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}\nResponse: {response_text[:300] if 'response_text' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"❌ Error in _fallback_json_parse(): {e}", exc_info=True)
            return None


class UserDataManager:
    """
    Менеджер данных пользователей (в памяти)
    В реальном проекте заменить на БД
    """
    
    def __init__(self):
        self.users = {}  # user_id -> profile
    
    def get_user(self, user_id: int) -> Dict:
        """Получает профиль пользователя"""
        if user_id not in self.users:
            self.users[user_id] = {
                "username": None,
                "grade": None,
                "subject": None,
                "test_history": [],
                "created_at": datetime.now().isoformat(),
                "total_tests": 0,
                "total_correct": 0,
                "total_questions": 0
            }
        return self.users[user_id]
    
    def update_user(self, user_id: int, **kwargs):
        """Обновляет данные пользователя"""
        user = self.get_user(user_id)
        user.update(kwargs)
    
    def add_test_result(self, user_id: int, grade: int, subject: str, 
                        correct: int, total: int, weak_topics: List[str]):
        """Добавляет результат теста в историю"""
        user = self.get_user(user_id)
        
        test_result = {
            "grade": grade,
            "subject": subject,
            "correct": correct,
            "total": total,
            "weak_topics": weak_topics,
            "date": datetime.now().isoformat()
        }
        
        user["test_history"].append(test_result)
        user["total_tests"] += 1
        user["total_correct"] += correct
        user["total_questions"] += total
        
        # Ограничиваем историю
        if len(user["test_history"]) > 20:
            user["test_history"] = user["test_history"][-20:]
    
    def get_statistics(self, user_id: int) -> Dict:
        """Возвращает статистику пользователя"""
        user = self.get_user(user_id)
        
        if not user["test_history"]:
            return {"message": "Пока нет пройденных тестов"}
        
        success_rate = 0
        if user["total_questions"] > 0:
            success_rate = round(user["total_correct"] / user["total_questions"] * 100)
        
        # Собираем все слабые темы
        all_weak_topics = []
        for test in user["test_history"]:
            all_weak_topics.extend(test["weak_topics"])
        
        from collections import Counter
        weak_topics_counter = Counter(all_weak_topics)
        top_weak = weak_topics_counter.most_common(5)
        
        return {
            "total_tests": user["total_tests"],
            "total_questions": user["total_questions"],
            "total_correct": user["total_correct"],
            "success_rate": success_rate,
            "top_weak_topics": top_weak,
            "last_test": user["test_history"][-1] if user["test_history"] else None
        }


# Создаем глобальные экземпляры
kb = KnowledgeBase()
llm = SimpleLLM()
user_manager = UserDataManager()


def format_question_as_latex(question_data: Dict[str, Any]) -> str:
    """
    Форматирует вопрос в LaTeX-формулу для красивого отображения.
    
    Args:
        question_data: Словарь с ключами 'question', 'options', 'correct'
    
    Returns:
        Форматированная строка с LaTeX
    """
    if not question_data or "question" not in question_data:
        return ""
    
    question_text = question_data.get("question", "")
    options = question_data.get("options", [])
    
    # Преобразуем математические выражения в LaTeX
    # Ищем выражения в фигурных скобках типа {x^2 + 2x + 1} и преобразуем их в LaTeX
    latex_question = question_text
    
    # Заменяем ^на ^ (для степеней)
    latex_question = re.sub(r'\^(\d+)', r'$^{\1}$', latex_question)
    
    # Заменяем дроби a/b на $\frac{a}{b}$
    latex_question = re.sub(r'(\d+)/(\d+)', r'$\\frac{\1}{\2}$', latex_question)
    
    # Заменяем sqrt на \sqrt
    latex_question = re.sub(r'sqrt\(([^)]+)\)', r'$\\sqrt{\1}$', latex_question)
    
    # Форматируем варианты ответов
    formatted_options = []
    for i, opt in enumerate(options):
        # Применяем тот же LaTeX-форматинг к опциям
        latex_opt = opt
        latex_opt = re.sub(r'\^(\d+)', r'$^{\1}$', latex_opt)
        latex_opt = re.sub(r'(\d+)/(\d+)', r'$\\frac{\1}{\2}$', latex_opt)
        latex_opt = re.sub(r'sqrt\(([^)]+)\)', r'$\\sqrt{\1}$', latex_opt)
        formatted_options.append(latex_opt)
    
    return {
        "question": latex_question,
        "options": formatted_options,
        "correct": question_data.get("correct", 0)
    }


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


def render_latex_to_image(latex_str: str) -> Optional[BufferedInputFile]:
    """
    Рендерит LaTeX формулу в изображение используя MikTeX и sympy
    Возвращает BufferedInputFile готовый для отправки в Telegram
    
    Args:
        latex_str: LaTeX формула (например: "$x^2 + 2x + 1$")
    
    Returns:
        BufferedInputFile объект или None
    """
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        # Удаляем символы долларов если они есть
        latex_str = latex_str.strip('$').strip()
        
        # Удаляем эмодзи которые предотвращают рендеринг LaTeX
        latex_str = remove_emojis(latex_str)
        
        if not latex_str:
            return None
        
        # Создаем BytesIO для вывода
        obj = BytesIO()
        
        try:
            # Используем sympy preview с MikTeX напрямую в BytesIO
            # Добавляем параметры для чистого рендеринга без артефактов
            preview(
                latex_str,
                preamble="""
              \\documentclass[margin=0cm,varwidth]{standalone}
              \\usepackage[T2A]{fontenc}
              \\usepackage[utf8]{inputenc}
              \\usepackage{amsmath}
              \\usepackage{amssymb}
              \\usepackage[english,russian]{babel}
              \\pagestyle{empty}
              \\begin{document}
              """,
                output="png", 
                viewer='BytesIO', 
                outputbuffer=obj,
                resolution=150)  # Увеличиваем разрешение для качества
            
            obj.seek(0)
            
            # Добавляем padding к изображению и удаляем артефакты
            try:
                img = Image.open(obj).convert('RGB')
                
                # Обрезаем белые края (артефакты)
                from PIL import ImageOps
                img = ImageOps.invert(img)
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                img = ImageOps.invert(img)
                
                w, h = img.size
                # Добавляем padding для лучшего вида
                padding = 10
                padded = Image.new('RGB', (w + padding*2, h + padding*2), 'white')
                padded.paste(img, (padding, padding))
                
                # Сохраняем обратно в BytesIO
                obj.seek(0)
                obj.truncate(0)
                padded.save(obj, format='PNG')
                obj.seek(0)
                
                # Возвращаем BufferedInputFile для Telegram
                photo = BufferedInputFile(obj.read(), filename="formula.png")
                return photo
            except Exception as e:
                print(f"⚠️ Ошибка при обработке изображения: {e}")
                obj.seek(0)
                photo = BufferedInputFile(obj.read(), filename="formula.png")
                return photo
                
        except Exception as e:
            print(f"⚠️ Ошибка при рендеринге LaTeX с sympy: {e}")
            return None
    
    except Exception as e:
        print(f"❌ Ошибка при обработке LaTeX: {e}")
        import traceback
        traceback.print_exc()
        return None


def text_with_latex_to_image(text: str, title: str = "") -> Optional[BufferedInputFile]:
    """
    Преобразует текст с LaTeX формулами в красивое изображение
    Использует MikTeX для рендеринга всего текста как LaTeX документа
    
    Args:
        text: Текст с LaTeX формулами (может содержать $...$)
        title: Заголовок изображения
    
    Returns:
        BufferedInputFile объект или None
    """
    if not text or not SYMPY_AVAILABLE:
        return None
    
    try:
        obj = BytesIO()
        # Создаем LaTeX документ с текстом
        # Заменяем LaTeX формулы в правильный формат
        content = text.strip("$").strip()
        
        # Удаляем эмодзи которые предотвращают рендеринг LaTeX
        content = remove_emojis(content)
        
        if not content:
            return None
        
        try:
            # Используем sympy preview для рендеринга всего текста
            preview(
                content,
                output='png',
                viewer='BytesIO',
                outputbuffer=obj,
                resolution=150,  # Увеличиваем разрешение для качества
                preamble="""
              \\documentclass[margin=0cm,varwidth]{standalone}
              \\usepackage[T2A]{fontenc}
              \\usepackage[utf8]{inputenc}
              \\usepackage{amsmath}
              \\usepackage{amssymb}
              \\usepackage[english,russian]{babel}
              \\pagestyle{empty}
              \\begin{document}
              """)
            
            obj.seek(0)
            
            # Добавляем padding к изображению и удаляем артефакты
            try:
                img = Image.open(obj).convert('RGB')
                
                # Обрезаем белые края (артефакты/стикеры)
                from PIL import ImageOps
                img = ImageOps.invert(img)
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                img = ImageOps.invert(img)
                
                w, h = img.size
                # Добавляем padding для лучшего вида
                padding = 15
                padded = Image.new('RGB', (w + padding*2, h + padding*2), 'white')
                padded.paste(img, (padding, padding))
                
                # Сохраняем обратно в BytesIO
                obj.seek(0)
                obj.truncate(0)
                padded.save(obj, format='PNG')
                obj.seek(0)
                
                # Возвращаем BufferedInputFile для Telegram
                photo = BufferedInputFile(obj.read(), filename="content.png")
                return photo
            except Exception as e:
                print(f"⚠️ Ошибка при обработке изображения: {e}")
                obj.seek(0)
                photo = BufferedInputFile(obj.read(), filename="content.png")
                return photo
                
        except Exception as e:
            print(f"⚠️ Ошибка при рендеринге текста с LaTeX: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"❌ Ошибка при создании изображения с LaTeX: {e}")
        import traceback
        traceback.print_exc()
        return None


def text_to_image(text: str, title: str = "", width: int = 800, bg_color: tuple = (255, 255, 255), text_color: tuple = (0, 0, 0)) -> Optional[str]:
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
        # Пытаемся загрузить шрифт, если его нет используем default
        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            text_font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback на стандартный шрифт
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Оборачиваем текст
        lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            # Приблизительная проверка ширины строки
            test_line = " ".join(current_line)
            # Проверяем примерно вмещается ли в строку
            if len(test_line) > 80:  # Примерный лимит
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
        
        # Добавляем заголовок если есть
        if title:
            draw.text((padding, y), title, fill=text_color, font=title_font)
            y += title_height
        
        # Добавляем текст
        for line in lines:
            draw.text((padding, y), line, fill=text_color, font=text_font)
            y += line_height
        
        # Создаем временную папку для изображений если её нет
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        
        # Сохраняем изображение
        file_name = f"image_{uuid.uuid4().hex}.png"
        file_path = temp_dir / file_name
        img.save(file_path, format='PNG')
        
        return str(file_path)
    
    except Exception as e:
        print(f"❌ Ошибка при преобразовании текста в изображение: {e}")
        return None


def question_to_image(question_data: Dict[str, Any], topic: str = "", topic_index: int = 0, total_topics: int = 0) -> Optional[BufferedInputFile]:
    """
    Преобразует вопрос в красивое изображение с поддержкой LaTeX формул
    
    Args:
        question_data: Словарь с вопросом и вариантами ответа (содержат LaTeX если нужны формулы)
        topic: Название темы
        topic_index: Номер текущего вопроса
        total_topics: Всего вопросов
    
    Returns:
        BufferedInputFile объект для отправки в Telegram или None
    """
    if not question_data or "question" not in question_data:
        return None
    
    # Формируем заголовок
    header = f"❓ Вопрос {topic_index + 1} из {total_topics}"
    if topic:
        header += f"\n📌 Тема: {topic}"
    
    # Формируем текст вопроса с вариантами
    full_text = f"{header}\n\n{question_data['question']}\n\n"
    
    for i, opt in enumerate(question_data.get("options", []), 1):
        full_text += f"{i}. {opt}\n"
    
    full_text += "\nВыбери номер ответа (1-4)"
    
    return text_with_latex_to_image(full_text, title="")


def task_to_image(task_text: str, topic: str = "", task_number: int = 0) -> Optional[BufferedInputFile]:
    """
    Преобразует задачу в красивое изображение с поддержкой LaTeX формул
    
    Args:
        task_text: Текст задачи (может содержать LaTeX формулы в $...$)
        topic: Название темы
        task_number: Номер задачи
    
    Returns:
        BufferedInputFile объект для отправки в Telegram или None
    """
    title = f"🎯 Задание {task_number}: {topic}" if topic else f"🎯 Задание {task_number}"
    return text_with_latex_to_image(task_text, title=title)


def extract_latex_content(response: str) -> str:
    """
    Извлекает содержимое LaTeX документа из ответа LLM
    
    Args:
        response: Ответ от LLM, который может содержать LaTeX документ
    
    Returns:
        Чистое содержимое LaTeX без служебных элементов
    """
    # Удаляем ``` markdown блоки если есть
    latex_content = response
    
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


def latex_document_to_image(response: str) -> Optional[BufferedInputFile]:
    """
    Преобразует LaTeX документ из ответа LLM в изображение
    
    Args:
        response: Полный ответ от LLM (может содержать LaTeX документ)
    
    Returns:
        BufferedInputFile объект для отправки в Telegram или None
    """
    if not response or not SYMPY_AVAILABLE:
        return None
    
    try:
        # Извлекаем содержимое LaTeX
        latex_content = extract_latex_content(response)
        
        if not latex_content:
            # Если не удалось извлечь, используем всё как есть, убирая эмодзи
            latex_content = remove_emojis(response)
        
        if not latex_content:
            return None
        
        obj = BytesIO()
        
        # Создаем полный LaTeX документ если нужно
        if "\\documentclass" not in latex_content:
            full_latex = f"""\\documentclass[margin=0.5cm]{{standalone}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{amsfonts}}
\\usepackage[english,russian]{{babel}}
\\usepackage{{color}}
\\pagestyle{{empty}}
\\begin{{document}}
{latex_content}
\\end{{document}}"""
        else:
            full_latex = latex_content
        
        try:
            preview(
                full_latex,
                output='png',
                viewer='BytesIO',
                outputbuffer=obj,
                resolution=150,
                quiet=True
            )
            
            obj.seek(0)
            
            # Обработка изображения: обрезка белых краёв и добавление padding
            try:
                img = Image.open(obj).convert('RGB')
                
                # Обрезаем белые края
                from PIL import ImageOps
                img = ImageOps.invert(img)
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                img = ImageOps.invert(img)
                
                w, h = img.size
                # Добавляем padding для лучшего вида
                padding = 20
                padded = Image.new('RGB', (w + padding*2, h + padding*2), 'white')
                padded.paste(img, (padding, padding))
                
                # Сохраняем обратно в BytesIO
                obj.seek(0)
                obj.truncate(0)
                padded.save(obj, format='PNG')
                obj.seek(0)
                
                # Возвращаем BufferedInputFile для Telegram
                photo = BufferedInputFile(obj.read(), filename="document.png")
                return photo
            except Exception as e:
                print(f"⚠️ Ошибка при обработке изображения: {e}")
                obj.seek(0)
                photo = BufferedInputFile(obj.read(), filename="document.png")
                return photo
                
        except Exception as e:
            print(f"⚠️ Ошибка при рендеринге LaTeX документа: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"❌ Ошибка при создании изображения из LaTeX документа: {e}")
        import traceback
        traceback.print_exc()
        return None