# import telebot
# from telebot import types
import asyncio 
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.filters import CommandStart
import pickle
import googletrans
import catboost
from sentence_transformers import SentenceTransformer
import keras.models
import sqlite3

with open("token.txt", "r") as f:
    token = f.readline()


bot = Bot(token)
dp = Dispatcher()
# dp = telebot.TeleBot(token)

with open("models/linear_logregress", "rb") as f:
    linear_logregress_model = pickle.load(f)


grad_boost_model = catboost.CatBoostClassifier()
grad_boost_model.load_model("models/catboost_model.bin")


neuro_network_model = keras.models.load_model("models/neuro_network_model.keras")


emb_model = SentenceTransformer('all-MiniLM-L6-v2')


translator = googletrans.Translator()


class_order = [3, 2, 6, 0, 5, 1, 4]


classes = ['тревога', 'биполярочка', 'депрессия', 'все в норме',
       'расстройство личности', 'стресс', 'суицидальные наклонности']


models = [linear_logregress_model, grad_boost_model, neuro_network_model]


models_i = {"linear": 0, "grad_boost": 1, "neuro_network": 2}


models_ru = ["линейная модель", "градиентный бустинг", "нейронная сеть"]
roc_aucs = [0.93, 0.97, 0.94]


print("ready")


@dp.message(CommandStart())
async def start(mes: types.Message):

    add_id(mes.chat.id)

    await mes.answer(f"Привет, {mes.from_user.first_name}!\nНапиши мне, что думаешь, и я скажу, какое у тебя психическое расстройство")


@dp.message(Command("menu"))
async def menu(mes: types.Message):

    add_id(mes.chat.id)

    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Задать модель", callback_data="set_model"),
            types.InlineKeyboardButton(text = "Задать способ вывода", callback_data="set_output")
        ],
        [
            types.InlineKeyboardButton(text = "Добавить в группу", url="https://t.me/durkachatbot?startgroup=botstart")
        ]
    ])

    model = get_model(mes.chat.id)
    message_str = f"Текущая модель: {models_ru[model]}\nROC AUC: {roc_aucs[model]}"

    output = get_output_method(mes.chat.id)
    message_str += "\n\nСпособ вывода:\n"
    if output or not model == 1:
        message_str += "вероятности диагнозов"
    else:
        message_str += "диагноз"
    
    await mes.answer(message_str, reply_markup=markup)

@dp.callback_query(F.data == "set_model")
async def callback_message(callback: types.CallbackQuery):
    await set_model(callback.message)

@dp.callback_query(F.data == "set_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output(callback.message)


@dp.message(Command("info"))
async def info(mes: types.Message):

    add_id(mes.chat.id)

    message_str = ("Я методами машинного обучения определяю, какое психическое расстройство у автора текста." +
                   " Просто напиши сюда, что думаешь, и я скажу, какое у тебя расстройство\n\n" 
                   "Меня написал очень талантливый молодой человек @MRabbit в качестве своего ml pet-проекта для дороги в IT\n\n" + 
                   "Я обучался на датаесе англоязычных сообщений из сотсетей\n" + 
                   "https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health\n\n"+
                   "Мой репозиторий открыт и каждый может убедится, что ваши сообщения остаются только между нами)\n"+
                   "https://github.com/maxusha-zaytsusha/durkachatbot\n\n"+
                   "Так же меня можно добавить в группу и я буду говорить, какое расстройство у автора сообщения!")

    # markup = types.InlineKeyboardMarkup([[types.InlineKeyboardButton("Добавить в группу", url="https://t.me/durkachatbot?startgroup=start")]])

    await mes.answer(message_str)


@dp.message(Command("set_model"))
async def set_model(mes: types.Message):

    add_id(mes.chat.id)

    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Линейная модель", callback_data="linear")
        ],
        [
            types.InlineKeyboardButton(text = "Градиентный бустинг", callback_data="grad_boost"),
            types.InlineKeyboardButton(text = "Нейронная сеть", callback_data="neuro_network")
        ]
    ])

    message_str = ("Выбери модель\n\n" + 
                   "<pre>"+
                   "|       Модель        |  ROC AUC |\n" + 
                   "|---------------------|----------|\n" + 
                   "|   линейная модель   |   0.93   |\n" + 
                   "| градиентный бустинг |   0.97   |\n" + 
                   "|   нейронная сеть    |   0.94   |</pre>")

    await mes.answer(message_str, reply_markup=markup, parse_mode="HTML")


@dp.callback_query(F.data == "linear")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(0, callback)

@dp.callback_query(F.data == "grad_boost")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(1, callback)

@dp.callback_query(F.data == "neuro_network")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(2, callback)


@dp.message(Command("set_output"))
async def set_output(mes: types.Message):

    add_id(mes.chat.id)

    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Показывать вероятности", callback_data="proba_output"),
            types.InlineKeyboardButton(text = "Показывать диагноз", callback_data="diagnosis_output")
        ]
    ])

    message_str = ("Выбери, как ты хочешь, чтобы отвечал бот.\n" + 
                   "Бот может писать диагноз или показывать вроятности каждого диагноза для текста (недоступно для случайного леса).")
    await mes.answer(message_str, reply_markup=markup)

@dp.callback_query(F.data == "proba_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output_method(1, callback)

@dp.callback_query(F.data == "diagnosis_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output_method(0, callback)


@dp.message(Command("model"))
async def get_model(mes: types.Message):

    add_id(mes.chat.id)

    i = get_model(mes.chat.id)

    await mes.answer(f"Текущая модель: {models_ru[i]}\nROC AUC: {roc_aucs[i]}")


@dp.message()
async def request(mes: types.Message):

    add_id(mes.chat.id)

    model = models[get_model(mes.chat.id)]
    pred = predict(mes.text, model)

    if get_output_method(mes.chat.id):
        await mes.answer(format_probas(pred))
    else:
        await mes.answer("у тебя " + probas_to_class(pred))

@dp.message(Command("predict"))
async def request2(mes: types.Message):

    add_id(mes.chat.id)

    model = models[get_model(mes.chat.id)]
    pred = predict(mes.text, model)

    if get_output_method(mes.chat.id):
        await mes.answer(format_probas(pred))
    else:
        await mes.answer("у тебя " + probas_to_class(pred))


# @dp.callback_query(F.data == "catalog")
# async def callback_message(callback: types.CallbackQuery):

#     if callback.data in models_i:
#         await set_model_by_id(models_i[callback.data], callback)

#     match callback.data:
#         case "set_model":
#             await set_model(callback.message)
#         case "set_output":
#             await set_output(callback.message)
#         case "diagnosis_output":
#             await set_output_method(0, callback)
#         case "proba_output":
#             await set_output_method(1, callback)

# @dp.callback_query(F.data == "catalog")
# async def callback_message(callback: types.CallbackQuery):
#     async def 

def predict(text, model):

    text = translator.translate(text, dest="en").text
    embeding = emb_model.encode([text])

    if model == neuro_network_model:
        res = model.predict(embeding, verbose=0)
    else:
        res = model.predict_proba(embeding)
    return res[0]


def class_to_diagnosis(s):
    match s:
        case "Stress":
            return "стресс"
        case "Bipolar":
            return "биполярочка"
        case "Anxiety":
            return "тревога"
        case "Personality disorder":
            return "расстройство личности"
        case "Normal":
            return "все в норме"
        case "Suicidal":
            return "суицидальные наклонности"
    return "депрессия" 


def format_probas(proba):

    res = ""

    for i in range(7):
        res += f'{classes[class_order[i]]}:  {proba[class_order[i]]:.2}\n'
    return res[:-1]


def probas_to_class(proba):

    i_max = 0
    val_max = proba[0]

    for i in range(1, 7):
        if proba[i] > val_max:
            val_max = proba[i]
            i_max = i
    return classes[i_max]


def add_id(user_id):

    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = " + str(user_id))
    row = cur.fetchall()

    if not row:
        cur.execute("INSERT INTO users (id, model, output_method) VALUES (" + str(user_id) + ", 0, 0)")
        conn.commit()


async def set_model_by_id(model_i, callback):

    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("UPDATE users SET model = " + str(model_i) + " WHERE id = " + str(callback.message.chat.id))
    conn.commit()

    await callback.answer("Выбранно: " + models_ru[model_i])


def get_model(user_id):

    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT model FROM users WHERE id = " + str(user_id))

    return cur.fetchall()[0][0]


def get_output_method(user_id):

    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT output_method FROM users WHERE id = " + str(user_id))

    return cur.fetchall()[0][0]


async def set_output_method(method, callback):

    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("UPDATE users SET output_method = " + str(method) + " WHERE id = " + str(callback.message.chat.id))
    conn.commit()

    if method:
        await callback.answer("Теперь бот выводит вероятности")
    else:
        await callback.answer("Теперь бот выводит диагноз")


async def error(callback):

    await callback.answer("Произошла ошибка!")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())