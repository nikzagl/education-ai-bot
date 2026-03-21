"""
Менеджер данных пользователей
"""

from datetime import datetime
from typing import Dict, List
from collections import Counter
from config import USER_TEST_HISTORY_LIMIT


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
        if len(user["test_history"]) > USER_TEST_HISTORY_LIMIT:
            user["test_history"] = user["test_history"][-USER_TEST_HISTORY_LIMIT:]
    
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
