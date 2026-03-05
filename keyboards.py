"""
Клавиатуры для бота
"""

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from typing import List


def get_class_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура выбора класса (1-11)
    """
    buttons = []
    row = []
    
    for i in range(1, 12):
        row.append(KeyboardButton(text=str(i)))
        if i % 4 == 0 or i == 11:
            buttons.append(row)
            row = []
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True,
        input_field_placeholder="Выбери класс"
    )


def get_subject_keyboard(subjects: List[str]) -> ReplyKeyboardMarkup:
    """
    Клавиатура выбора предмета
    """
    buttons = [[KeyboardButton(text=subject)] for subject in subjects]
    buttons.append([KeyboardButton(text="🔙 Назад к выбору класса")])
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True,
        input_field_placeholder="Выбери предмет"
    )


def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    """
    Главное меню
    """
    buttons = [
        [KeyboardButton(text="📝 Пройти тест")],
        [KeyboardButton(text="📚 Список тем")],
        [KeyboardButton(text="📊 Мой прогресс")],
        [KeyboardButton(text="⚙️ Сменить класс/предмет")]
    ]
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True,
        input_field_placeholder="Выбери действие"
    )


def get_test_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура во время теста
    """
    buttons = [
        [KeyboardButton(text="1"), KeyboardButton(text="2"), 
         KeyboardButton(text="3"), KeyboardButton(text="4")],
        [KeyboardButton(text="❌ Прервать тест")]
    ]
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True,
        input_field_placeholder="Выбери ответ (1-4)"
    )


def get_practice_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура после теста
    """
    buttons = [
        [KeyboardButton(text="✅ Тренировать слабые темы")],
        [KeyboardButton(text="📝 Пройти тест заново")],
        [KeyboardButton(text="🏠 Главное меню")]
    ]
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True
    )


def get_back_to_menu_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура возврата в меню
    """
    buttons = [[KeyboardButton(text="🏠 Главное меню")]]
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True
    )


def get_confirm_keyboard() -> ReplyKeyboardMarkup:
    """
    Клавиатура подтверждения
    """
    buttons = [
        [KeyboardButton(text="✅ Да")],
        [KeyboardButton(text="❌ Нет")]
    ]
    
    return ReplyKeyboardMarkup(
        keyboard=buttons,
        resize_keyboard=True
    )