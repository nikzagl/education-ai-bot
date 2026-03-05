"""
Вспомогательные функции и классы
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from prompts import (
    QUESTION_PROMPT, 
    TASK_PROMPT, 
    RECOMMENDATIONS_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    TUTOR_SYSTEM_PROMPT
)

load_dotenv()


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
    Простой класс для работы с LLM
    """
    
    def __init__(self):
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.model = "gpt-oss-120b"
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
                print(f"⚠️ Отсутствует параметр в промпте: {e}")
            except Exception as e:
                print(f"⚠️ Ошибка форматирования промпта: {e}")
        
        system = system_prompt or TEACHER_SYSTEM_PROMPT
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return None
    
    def ask_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[Dict]:
        """
        Запрос с гарантией JSON-ответа
        """
        # Подставляем параметры
        if kwargs:
            try:
                prompt = prompt.format(**kwargs)
            except KeyError as e:
                print(f"⚠️ Отсутствует параметр в промпте: {e}")
            except Exception as e:
                print(f"⚠️ Ошибка форматирования промпта: {e}")
        
        system = system_prompt or TEACHER_SYSTEM_PROMPT
        system += " Отвечай строго в формате JSON, без пояснений."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Ошибка JSON LLM: {e}")
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