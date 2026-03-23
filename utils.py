"""
Вспомогательные функции и классы
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

from pydantic import BaseModel, Field
from mistralai.client import Mistral
from dotenv import load_dotenv

from config import (
    MISTRAL_API_KEY, LLM_MODEL, LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
     KB_BASE_PATH
)

from prompts import SYSTEM_PROMPT

from user_manager import UserDataManager


try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ langchain не установлен, используется прямое API Mistral")

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
    
    def __init__(self, base_path: str = KB_BASE_PATH):
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
        api_key = MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("❌ MISTRAL_API_KEY не найден в переменных окружения")
        
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatMistralAI(
                model=LLM_MODEL,
                api_key=api_key,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
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
        
        system = system_prompt or SYSTEM_PROMPT
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"📨 LLM REQUEST (ask)")
        logger.debug(f"{'='*80}")
        logger.debug(f"System Prompt:\n{system}")
        logger.debug(f"\nUser Prompt:\n{prompt}")
        logger.debug(f"{'='*80}\n")
        
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
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS
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
        
        system = system_prompt or SYSTEM_PROMPT
        model = pydantic_model or QuestionAnswer
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"📨 LLM REQUEST (ask_json)")
        logger.debug(f"{'='*80}")
        logger.debug(f"Pydantic Model: {model.__name__}")
        logger.debug(f"System Prompt:\n{system}")
        logger.debug(f"\nUser Prompt:\n{prompt}")
        logger.debug(f"{'='*80}\n")
        
        try:
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                logger.debug("Using LangChain with structured output (Pydantic)")
                # Используем LangChain с Pydantic структурированием
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
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
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
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
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


# Создаем глобальные экземпляры
kb = KnowledgeBase()
llm = SimpleLLM()
user_manager = UserDataManager()
