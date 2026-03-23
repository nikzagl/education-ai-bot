#!/usr/bin/env python
"""
Основной файл бота с FSM
"""

import os
import logging
import asyncio
import re
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardRemove, FSInputFile

from states import StudentState
from keyboards import *
from utils import llm, kb, user_manager
from image_utils import question_to_image, latex_document_to_image, task_to_image
from config import MAX_QUESTIONS, MAX_PRACTICE

from prompts import *

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Константы
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_TEST", 3))
MAX_PRACTICE = int(os.getenv("MAX_PRACTICE_TASKS", 3))
# Базовая преамбула для случаев, когда LaTeX создаётся вручную (не из промпта)


# ========== ХЕЛПЕРЫ ==========

async def safe_edit_message(message: types.Message, text: str, **kwargs):
    """Безопасное редактирование сообщения"""
    try:
        await message.edit_text(text, **kwargs)
    except Exception:
        await message.answer(text, **kwargs)


# ========== ОБРАБОТЧИКИ КОМАНД ==========

@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    """Начало работы - выбор класса"""
    user_id = message.from_user.id
    username = message.from_user.first_name
    
    # Сохраняем username
    user_manager.update_user(user_id, username=username)
    
    await state.set_state(StudentState.choosing_grade)
    await message.answer(
        f"👋 Привет, {username}!\n\n"
        f"Я бот-помощник для школьников. Помогу проверить знания "
        f"и подтянуть слабые места с помощью ИИ.\n\n"
        f"📚 В каком ты классе? (1-11)",
        reply_markup=get_class_keyboard()
    )


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    """Отмена текущего действия"""
    current_state = await state.get_state()
    
    if current_state is None:
        await message.answer(
            "Нет активного действия.",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    await state.clear()
    await message.answer(
        "❌ Действие отменено. Используй /start для начала работы.",
        reply_markup=ReplyKeyboardRemove()
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Справка"""
    help_text = (
        "🤖 *Помощь по боту*\n\n"
        "/start - Начать работу\n"
        "/cancel - Отменить текущее действие\n"
        "/help - Показать эту справку\n\n"
        "*Как пользоваться:*\n"
        "1. Выбери класс и предмет\n"
        "2. Пройди тест из 3-5 вопросов\n"
        "3. Получи рекомендации по слабым темам\n"
        "4. Тренируйся на задачах\n\n"
        "Бот использует ИИ для генерации уникальных вопросов!"
    )
    
    await message.answer(help_text, parse_mode="Markdown")


# ========== ВЫБОР КЛАССА ==========

@dp.message(StudentState.choosing_grade, 
            lambda m: m.text and m.text.isdigit() and 1 <= int(m.text) <= 11)
async def process_grade(message: types.Message, state: FSMContext):
    """Обработка выбора класса"""
    user_id = message.from_user.id
    grade = int(message.text)
    
    # Сохраняем класс
    user_manager.update_user(user_id, grade=grade)
    await state.update_data(grade=grade)
    
    # Получаем доступные предметы
    subjects = []
    for subject in ["Математика", "Физика", "Русский язык", "Химия", "Биология"]:
        if kb.get_topics(grade, subject):
            subjects.append(subject)
    
    if not subjects:
        await message.answer(
            f"😕 Извини, для {grade} класса пока нет доступных предметов. "
            f"Попробуй другой класс.",
            reply_markup=get_class_keyboard()
        )
        return
    
    await state.set_state(StudentState.choosing_subject)
    await message.answer(
        f"📚 Отлично, {grade} класс! Теперь выбери предмет:",
        reply_markup=get_subject_keyboard(subjects)
    )


@dp.message(StudentState.choosing_grade)
async def process_grade_invalid(message: types.Message):
    """Неверный ввод при выборе класса"""
    await message.answer(
        "❌ Пожалуйста, выбери класс, нажав на кнопку с цифрой (1-11).",
        reply_markup=get_class_keyboard()
    )


# ========== ВЫБОР ПРЕДМЕТА ==========

@dp.message(StudentState.choosing_subject, lambda m: m.text == "🔙 Назад к выбору класса")
async def process_back_to_grade(message: types.Message, state: FSMContext):
    """Возврат к выбору класса"""
    await state.set_state(StudentState.choosing_grade)
    await message.answer(
        "Выбери класс:",
        reply_markup=get_class_keyboard()
    )


@dp.message(StudentState.choosing_subject)
async def process_subject(message: types.Message, state: FSMContext):
    """Обработка выбора предмета"""
    user_id = message.from_user.id
    subject = message.text
    
    data = await state.get_data()
    grade = data.get("grade") or user_manager.get_user(user_id).get("grade")
    
    # Проверяем предмет
    topics = kb.get_topics(grade, subject)
    if not topics:
        await message.answer(
            f"😕 Для {subject} в {grade} классе пока нет тем. Выбери другой предмет.",
            reply_markup=get_subject_keyboard(
                ["Математика", "Физика", "Русский язык", "Химия", "Биология"]
            )
        )
        return
    
    # Сохраняем предмет
    user_manager.update_user(user_id, subject=subject)
    await state.update_data(subject=subject)
    
    await state.set_state(StudentState.main_menu)
    await message.answer(
        f"✅ *Готово!*\n\n"
        f"Класс: {grade}\n"
        f"Предмет: {subject}\n\n"
        f"Что будем делать?",
        parse_mode="Markdown",
        reply_markup=get_main_menu_keyboard()
    )


# ========== ГЛАВНОЕ МЕНЮ ==========

@dp.message(StudentState.main_menu, lambda m: m.text == "📚 Список тем")
async def process_show_topics(message: types.Message, state: FSMContext):
    """Показ списка тем"""
    user_id = message.from_user.id
    data = await state.get_data()
    
    grade = data.get("grade") or user_manager.get_user(user_id).get("grade")
    subject = data.get("subject") or user_manager.get_user(user_id).get("subject")
    
    topics = kb.get_topic_names(grade, subject)
    
    if not topics:
        await message.answer(
            f"😕 Для {subject} в {grade} классе пока нет тем.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    # Разбиваем на части, если много тем
    topics_text = f"📚 *Темы {subject}, {grade} класс:*\n\n"
    
    for i, topic in enumerate(topics, 1):
        topics_text += f"{i}. {topic}\n"
        
        # Если текст слишком длинный, отправляем частями
        if len(topics_text) > 3000:
            await message.answer(topics_text, parse_mode="Markdown")
            topics_text = ""
    
    if topics_text:
        await message.answer(
            topics_text + "\n\nХочешь проверить знания? Нажми '📝 Пройти тест'",
            parse_mode="Markdown"
        )
    
    await message.answer(
        "Выбери действие:",
        reply_markup=get_main_menu_keyboard()
    )


@dp.message(StudentState.main_menu, lambda m: m.text == "📊 Мой прогресс")
async def process_progress(message: types.Message, state: FSMContext):
    """Показ прогресса ученика"""
    user_id = message.from_user.id
    stats = user_manager.get_statistics(user_id)
    
    if "message" in stats:
        await message.answer(
            "📊 У тебя пока нет пройденных тестов. Пройди первый тест!",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    progress_text = (
        f"📊 *Твой прогресс*\n\n"
        f"📝 Всего тестов: {stats['total_tests']}\n"
        f"❓ Вопросов отвечено: {stats['total_questions']}\n"
        f"✅ Правильных ответов: {stats['total_correct']}\n"
        f"📈 Успеваемость: {stats['success_rate']}%\n\n"
    )
    
    if stats['top_weak_topics']:
        progress_text += "*Темы для повторения:*\n"
        for topic, count in stats['top_weak_topics']:
            progress_text += f"• {topic} (ошибок: {count})\n"
    
    if stats['last_test']:
        last = stats['last_test']
        progress_text += (
            f"\n*Последний тест:*\n"
            f"{last['subject']}, {last['grade']} класс\n"
            f"✅ {last['correct']}/{last['total']}\n"
        )
    
    await message.answer(progress_text, parse_mode="Markdown")
    await message.answer(
        "Выбери действие:",
        reply_markup=get_main_menu_keyboard()
    )


@dp.message(StudentState.main_menu, lambda m: m.text == "⚙️ Сменить класс/предмет")
async def process_change_settings(message: types.Message, state: FSMContext):
    """Смена класса или предмета"""
    await state.set_state(StudentState.choosing_grade)
    await message.answer(
        "В каком ты классе?",
        reply_markup=get_class_keyboard()
    )


@dp.message(StudentState.main_menu, lambda m: m.text == "📝 Пройти тест")
async def start_test(message: types.Message, state: FSMContext):
    """Начинаем тестирование"""
    user_id = message.from_user.id
    data = await state.get_data()
    
    grade = data.get("grade") or user_manager.get_user(user_id).get("grade")
    subject = data.get("subject") or user_manager.get_user(user_id).get("subject")
    
    if not grade or not subject:
        await message.answer(
            "❌ Сначала выбери класс и предмет! Используй /start",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    # Получаем темы
    topics = kb.get_topic_names(grade, subject)
    
    if not topics:
        await message.answer(
            f"😕 Для {subject} в {grade} классе пока нет тем.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    # Берем первые N тем
    test_topics = topics[:MAX_QUESTIONS]
    
    # Сохраняем данные теста
    await state.update_data(
        test_topics=test_topics,
        current_topic_index=0,
        correct_topics=[],
        wrong_topics=[],
        current_question=None,
        questions_asked=0
    )
    
    await state.set_state(StudentState.test_generating)
    
    await message.answer(
        f"📋 *Начинаем тест по {subject}!*\n"
        f"Будет {len(test_topics)} вопросов.\n"
        f"Генерирую первый вопрос... 🔮",
        parse_mode="Markdown",
        reply_markup=get_test_keyboard()
    )
    
    # Генерируем первый вопрос
    await generate_question(message, state)


async def generate_question(message: types.Message, state: FSMContext):
    """Генерация вопроса по текущей теме"""
    data = await state.get_data()
    
    topic_index = data.get("current_topic_index", 0)
    test_topics = data.get("test_topics", [])
    
    if topic_index >= len(test_topics):
        await finish_test(message, state)
        return
    
    topic_name = test_topics[topic_index]
    grade = data.get("grade")
    subject = data.get("subject")
    
    # Получаем дополнительную информацию о теме
    topic_info = kb.get_topic_by_name(grade, subject, topic_name)
    
    # Формируем промпт с контекстом
    examples = ""
    mistakes = ""
    
    if topic_info:
        if topic_info.get("examples"):
            examples = f"\nПримеры: {', '.join(topic_info['examples'][:2])}"
        if topic_info.get("common_mistakes"):
            mistakes = f"\nТипичные ошибки: {', '.join(topic_info['common_mistakes'][:2])}"
    
    question_data = llm.ask_json(
        QUESTION_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        topic=topic_name,
        grade=grade,
        subject=subject,
        examples=examples,
        mistakes=mistakes
    )
    
    if not question_data or "error" in question_data or "question" not in question_data:
        # Если ошибка, пропускаем тему
        await message.answer(
            f"❌ Не удалось сгенерировать вопрос по теме '{topic_name}'. Пропускаем..."
        )
        await state.update_data(current_topic_index=topic_index + 1)
        await generate_question(message, state)
        return
    
    # Сохраняем вопрос
    await state.update_data(
        current_question=question_data,
        questions_asked=data.get("questions_asked", 0) + 1
    )
    
    await state.set_state(StudentState.test_answering)
    
    # Применяем LaTeX форматирование
    
    
    # Преобразуем вопрос в изображение
    question_image = question_to_image(
        question_data,
        topic=topic_name,
        topic_index=topic_index,
        total_topics=len(test_topics)
    )
    print(f"Сгенерирован вопрос по теме: {topic_name}")
    
    if question_image:
        try:
            # Формируем заголовок для caption
            caption = f"❓ Вопрос {topic_index + 1} из {len(test_topics)}\n📌 Тема: {topic_name}\n\nВыбери номер ответа (1-4)"
            
            await message.answer_photo(
                photo=question_image,
                caption=caption,
                reply_markup=get_test_keyboard()
            )
        except Exception as e:
            logger.error(f"Ошибка отправки фото: {e}")
            # Fallback на текстовое сообщение
            options_text = "\n".join([
                f"{i+1}. {opt}" for i, opt in enumerate(question_data["options"])
            ])
            await message.answer(
                f"❓ *Вопрос {topic_index + 1} из {len(test_topics)}*\n"
                f"📌 Тема: {topic_name}\n\n"
                f"{question_data['question']}\n\n"
                f"{options_text}\n\n"
                f"*Выбери номер ответа (1-4)*",
                parse_mode="Markdown",
                reply_markup=get_test_keyboard()
            )
    else:
        # Fallback на текстовое сообщение если не получилось создать изображение
        options_text = "\n".join([
            f"{i+1}. {opt}" for i, opt in enumerate(question_data["options"])
        ])
        await message.answer(
            f"❓ *Вопрос {topic_index + 1} из {len(test_topics)}*\n"
            f"📌 Тема: {topic_name}\n\n"
            f"{question_data['question']}\n\n"
            f"{options_text}\n\n"
            f"*Выбери номер ответа (1-4)*",
            parse_mode="Markdown",
            reply_markup=get_test_keyboard()
        )


@dp.message(StudentState.test_answering, lambda m: m.text == "❌ Прервать тест")
async def abort_test(message: types.Message, state: FSMContext):
    """Прерывание теста"""
    await state.set_state(StudentState.main_menu)
    await message.answer(
        "❌ Тест прерван. Возвращаю в главное меню.",
        reply_markup=get_main_menu_keyboard()
    )


@dp.message(StudentState.test_answering, 
            lambda m: m.text and m.text.isdigit() and 1 <= int(m.text) <= 4)
async def process_test_answer(message: types.Message, state: FSMContext):
    """Обработка ответа на вопрос"""
    answer = int(message.text) - 1  # 0-3
    
    data = await state.get_data()
    current_question = data.get("current_question")
    topic_index = data.get("current_topic_index", 0)
    test_topics = data.get("test_topics", [])
    
    if not current_question:
        await message.answer("❌ Ошибка данных. Начинаем тест заново.")
        await start_test(message, state)
        return
    
    topic = test_topics[topic_index]
    correct_topics = data.get("correct_topics", [])
    wrong_topics = data.get("wrong_topics", [])
    
    # Проверяем ответ
    if answer == current_question["correct"]:
        await message.answer("✅ *Правильно!*", parse_mode="Markdown")
        correct_topics.append(topic)
    else:
        correct_option = current_question["options"][current_question["correct"]]
        await message.answer(
            f"❌ *Неправильно.*\n\nПравильный ответ: *{correct_option}*",
            parse_mode="Markdown"
        )
        wrong_topics.append(topic)
    
    # Обновляем состояние
    await state.update_data(
        current_topic_index=topic_index + 1,
        correct_topics=correct_topics,
        wrong_topics=wrong_topics
    )
    
    # Переходим к следующему вопросу
    await state.set_state(StudentState.test_generating)
    await generate_question(message, state)


@dp.message(StudentState.test_answering)
async def process_invalid_answer(message: types.Message):
    """Неверный ввод во время теста"""
    await message.answer(
        "❌ Пожалуйста, выбери номер ответа от 1 до 4.",
        reply_markup=get_test_keyboard()
    )


async def finish_test(message: types.Message, state: FSMContext):
    """Завершение теста и показ результатов"""
    data = await state.get_data()
    
    correct = data.get("correct_topics", [])
    wrong = data.get("wrong_topics", [])
    total = len(correct) + len(wrong)
    
    user_id = message.from_user.id
    grade = data.get("grade")
    subject = data.get("subject")
    
    # Сохраняем в историю
    user_manager.add_test_result(
        user_id=user_id,
        grade=grade,
        subject=subject,
        correct=len(correct),
        total=total,
        weak_topics=wrong
    )
    
    # Формируем отчет
    report = (
        f"📊 *Результаты теста*\n\n"
        f"✅ Правильно: {len(correct)} из {total}\n"
        f"❌ Ошибки: {len(wrong)} из {total}\n\n"
    )
    
    if wrong:
        report += "*Темы для повторения:*\n"
        report += "\n".join([f"• {topic}" for topic in wrong])
        report += "\n\n"
    else:
        report += "🎉 *Отличная работа! Ты знаешь все темы!*\n\n"
    
    await state.set_state(StudentState.showing_results)
    await message.answer(report, parse_mode="Markdown")
    
    # Если есть ошибки, предлагаем тренировку
    if wrong:
        await message.answer(
            "🎯 Хочешь потренироваться на этих темах?",
            reply_markup=get_practice_keyboard()
        )
    else:
        await state.set_state(StudentState.main_menu)
        await message.answer(
            "Что делаем дальше?",
            reply_markup=get_main_menu_keyboard()
        )


# ========== ТРЕНИРОВКА ==========

@dp.message(StudentState.showing_results, lambda m: m.text == "✅ Тренировать слабые темы")
async def start_practice(message: types.Message, state: FSMContext):
    """Начинаем тренировку по слабым темам"""
    data = await state.get_data()
    weak_topics = data.get("wrong_topics", [])
    
    if not weak_topics:
        await message.answer(
            "🎉 У тебя нет слабых тем! Можешь пройти тест заново.",
            reply_markup=get_main_menu_keyboard()
        )
        await state.set_state(StudentState.main_menu)
        return
    
    # Инициализируем практику
    practice_topics = weak_topics[:MAX_PRACTICE]
    await state.update_data(
        practice_topics=practice_topics,
        current_practice_index=0,
        practice_completed=0,
        practice_skipped=0
    )
    
    await message.answer(
        f"🔮 Начинаем тренировку по {len(practice_topics)} темам!\n"
        f"После каждого неверного ответа я покажу решение."
    )
    
    # Генерируем первое задание
    await generate_practice_task(message, state)


async def generate_practice_task(message: types.Message, state: FSMContext):
    """Генерирует одно практическое задание"""
    data = await state.get_data()
    practice_topics = data.get("practice_topics", [])
    current_index = data.get("current_practice_index", 0)
    
    if current_index >= len(practice_topics):
        # Завершаем тренировку
        completed = data.get("practice_completed", 0)
        skipped = data.get("practice_skipped", 0)
        await state.set_state(StudentState.main_menu)
        await message.answer(
            f"✅ Тренировка завершена!\n\n"
            f"✓ Решено правильно: {completed}\n"
            f"⊘ Пропущено: {skipped}",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    topic = practice_topics[current_index]
    grade = data.get("grade")
    subject = data.get("subject")
    
    # Получаем информацию о теме
    topic_info = kb.get_topic_by_name(grade, subject, topic)
    examples = ""
    if topic_info and topic_info.get("examples"):
        examples = f"\nПримеры: {', '.join(topic_info['examples'][:2])}"
    
    # Генерируем задание (без вариантов ответа!)
    task = llm.ask(
        TASK_TRAINING_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        topic=topic,
        grade=grade,
        subject=subject,
        examples=examples
    )
    
    if task:
        logger.info(f"Практическое задание по теме: {topic}")
        logger.info(f"Задача: {task}")
        
        # Сохраняем задание для проверки ответа
        await state.update_data(
            current_practice_task=task,
            current_practice_topic=topic
        )
        
        # Отправляем задание
        try:
            task_image = task_to_image(task)
            if task_image:
                await message.answer_photo(
                    photo=task_image,
                    caption=f"🎯 Задание {current_index + 1} из {len(practice_topics)}: {topic}\n\n📝 Твой ответ:",
                    reply_markup=get_back_to_menu_keyboard()
                )
            else:
                await message.answer(
                    f"🎯 Задание {current_index + 1} из {len(practice_topics)}: {topic}\n\n{task}\n\n📝 Напиши свой ответ:",
                    parse_mode="Markdown",
                    reply_markup=get_back_to_menu_keyboard()
                )
        except Exception as e:
            logger.error(f"Ошибка отправки задания: {e}")
            await message.answer(
                f"🎯 Задание {current_index + 1} из {len(practice_topics)}: {topic}\n\n{task}\n\n📝 Напиши свой ответ:",
                parse_mode="Markdown",
                reply_markup=get_back_to_menu_keyboard()
            )
        
        # Переходим на ожидание ответа
        await state.set_state(StudentState.practice_answering)
    else:
        await message.answer(f"❌ Не удалось сгенерировать задание по теме '{topic}'. Пропускаю...")
        await state.update_data(
            current_practice_index=current_index + 1,
            practice_skipped=data.get("practice_skipped", 0) + 1
        )
        await generate_practice_task(message, state)


@dp.message(StudentState.practice_answering, lambda m: m.text == "🏠 Главное меню")
async def practice_abort(message: types.Message, state: FSMContext):
    """Пользователь вышел из практики"""
    await state.set_state(StudentState.main_menu)
    await message.answer(
        "🏠 Главное меню:",
        reply_markup=get_main_menu_keyboard()
    )


@dp.message(StudentState.practice_answering)
async def process_practice_answer(message: types.Message, state: FSMContext):
    """Обработка ответа на практическое задание"""
    student_answer = message.text.strip()
    data = await state.get_data()
    
    task_text = data.get("current_practice_task")
    topic = data.get("current_practice_topic")
    current_index = data.get("current_practice_index", 0)
    practice_topics = data.get("practice_topics", [])
    
    if not task_text or not topic:
        await message.answer("❌ Ошибка. Начинаем заново.")
        await state.set_state(StudentState.main_menu)
        await start_practice(message, state)
        return
    
    # Проверяем ответ через LLM
    check_prompt = f"""Ученик решал задачу и дал ответ.

Задача:
{task_text}

Ответ ученика:
{student_answer}

Проверь, правильный ли ответ. Просто ответь одним словом:
- "ПРАВИЛЬНО" если решение верное
- "НЕПРАВИЛЬНО" если решение неверное или неполное"""
    
    check_result = llm.ask(
        check_prompt,
        system_prompt=SYSTEM_PROMPT
    )
    
    is_correct = check_result and "НЕПРАВИЛЬНО" not in check_result.upper()
    
    if is_correct:
        await message.answer(f"✅ *Правильно!*\n\nМолодец! Переходим к следующему заданию.", parse_mode="Markdown")
        await state.update_data(
            current_practice_index=current_index + 1,
            practice_completed=data.get("practice_completed", 0) + 1
        )
    else:
        await message.answer(
            f"❌ Не совсем правильно...\n\n"
            f"Сейчас покажу решение 👇",
            parse_mode="Markdown"
        )
        
        # Генерируем решение
        solution = llm.ask(
            SOLUTION_GENERATION_PROMPT,
            system_prompt=SYSTEM_PROMPT,
            task_text=task_text
        )
        
        if solution:
            logger.info(f"Решение для задачи '{topic}': {solution}")
            try:
                solution_image = latex_document_to_image(solution)
                if solution_image:
                    await message.answer_photo(
                        photo=solution_image,
                        caption="📚 Вот как нужно решать:"
                    )
                else:
                    await message.answer(
                        f"📚 *Вот как нужно решать:*\n\n{solution}",
                        parse_mode="Markdown"
                    )
            except Exception as e:
                logger.error(f"Ошибка отправки решения: {e}")
                await message.answer(
                    f"📚 *Вот как нужно решать:*\n\n{solution}",
                    parse_mode="Markdown"
                )
            
            await asyncio.sleep(1)
        
        # Переходим к следующему заданию
        await state.update_data(
            current_practice_index=current_index + 1
        )
    
    # Генерируем следующее задание
    await asyncio.sleep(1)
    await generate_practice_task(message, state)


@dp.message(StudentState.showing_results, lambda m: m.text == "📝 Пройти тест заново")
async def restart_test(message: types.Message, state: FSMContext):
    """Повторное прохождение теста"""
    await start_test(message, state)


@dp.message(StudentState.showing_results, lambda m: m.text == "🏠 Главное меню")
@dp.message(StudentState.practice_generating, lambda m: m.text == "🏠 Главное меню")
@dp.message(StudentState.practice_showing, lambda m: m.text == "🏠 Главное меню")
@dp.message(StudentState.practice_answering, lambda m: m.text == "🏠 Главное меню")
async def back_to_main_menu(message: types.Message, state: FSMContext):
    """Возврат в главное меню"""
    await state.set_state(StudentState.main_menu)
    await message.answer(
        "🏠 Главное меню:",
        reply_markup=get_main_menu_keyboard()
    )


# ========== ОБРАБОТКА НЕИЗВЕСТНЫХ КОМАНД ==========

@dp.message()
async def handle_unknown(message: types.Message, state: FSMContext):
    """Обработка всех неизвестных сообщений"""
    current_state = await state.get_state()
    
    if current_state is None:
        await message.answer(
            "❓ Неизвестная команда. Используй /start для начала работы",
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        # Если есть состояние, но сообщение не подходит
        await message.answer(
            "❌ Пожалуйста, используй кнопки или /cancel для отмены",
            reply_markup=get_back_to_menu_keyboard()
        )


# ========== ЗАПУСК ==========

async def main():
    """Запуск бота"""
    logger.info("🤖 Бот запускается...")
    
    # Проверка наличия необходимых файлов
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.error("❌ TELEGRAM_BOT_TOKEN не найден в .env")
        return
    
    if not os.getenv("MISTRAL_API_KEY"):
        logger.error("❌ MISTRAL_API_KEY не найден в .env")
        return
    
    logger.info("✅ Бот готов к работе!")
    
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())