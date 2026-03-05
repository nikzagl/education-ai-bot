"""
Состояния FSM для бота
"""

from aiogram.fsm.state import State, StatesGroup


class StudentState(StatesGroup):
    """
    Состояния ученика в боте
    """
    # Начальные состояния
    choosing_grade = State()           # Выбор класса
    choosing_subject = State()         # Выбор предмета
    main_menu = State()                 # Главное меню
    
    # Состояния тестирования
    test_in_progress = State()          # Тест идет
    test_generating = State()           # Генерация вопроса
    test_answering = State()            # Ожидание ответа
    
    # Состояния тренировки
    practice_generating = State()       # Генерация задач
    practice_showing = State()          # Показ задач
    
    # Результаты
    showing_results = State()           # Показ результатов
    viewing_progress = State()          # Просмотр прогресса
    learning_topics = State()           # Изучение тем
    
    # Настройки
    changing_settings = State()         # Смена настроек