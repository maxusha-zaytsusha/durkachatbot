import asyncio 
from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.filters.command import Command
from aiogram.filters import CommandStart, JOIN_TRANSITION
from aiogram.filters.chat_member_updated import ChatMemberUpdatedFilter
import pickle
import googletrans
import catboost
from sentence_transformers import SentenceTransformer
import keras.models
import sqlite3

# получаем токен бота и создаем объекты бота и деспетчера
with open("token.txt", "r") as f:
    token = f.readline()

bot = Bot(token)
dp = Dispatcher()
router = Router()


# загружаем модели для определения диагноза
with open("models/linear_logregress", "rb") as f:
    linear_logregress_model = pickle.load(f)

grad_boost_model = catboost.CatBoostClassifier()
grad_boost_model.load_model("models/catboost_model.bin")

neuro_network_model = keras.models.load_model("models/neuro_network_model.keras")


# загружаем модель для создания эмбедингов а так же объект переводчика
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

translator = googletrans.Translator()


# диагнозы и их порадок вывода пользователю
class_order = [3, 2, 6, 0, 5, 1, 4]

classes = ['тревога', 'биполярочка', 'депрессия', 'все в норме',
       'расстройство личности', 'стресс', 'суицидальные наклонности']


# модели, их индексы и качество
models = [linear_logregress_model, grad_boost_model, neuro_network_model]

models_i = {"linear": 0, "grad_boost": 1, "neuro_network": 2}

models_ru = ["линейная модель", "градиентный бустинг", "нейронная сеть"]

roc_aucs = [0.93, 0.97, 0.94]


print("ready")

# обработка команды /start
@dp.message(CommandStart())
async def start(mes: types.Message):

    # id чата добавляется в базу, если его там нет
    add_id(mes.chat.id)
    
    await hello(mes, False)
    

async def hello(mes, group):

    if not group:
        text = (f"Привет, {mes.from_user.first_name}, я твой ИИ санитар!\nНапиши мне, что думаешь, и я скажу, какое у тебя психическое расстройство.")
    else:
        text = (f"Привет всем! Я ваш ИИ санитар.\nЯ определяю психическое расстройство по сообщению пациента. "+
                 "Чтобы я увидел ваше сообщение и дал вам диагноз, начните его с обращения 'Санитар', например:\n"+
                 "Санитар, мне кажется, что меня не существует.\n" + 
                 "Так же вы можете ответить собеседнику словом 'Санитар', и я напишу его диагноз")

    # выводиться сообщение
    await bot.send_message(chat_id=mes.chat.id, text=text)


# обработка команды /menu
@dp.message(Command("menu"))
async def menu(mes: types.Message):

    # делаем кнопки
    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Задать модель", callback_data="set_model"),
            types.InlineKeyboardButton(text = "Задать способ вывода", callback_data="set_output")
        ],
        [
            types.InlineKeyboardButton(text = "Добавить в группу", url="https://t.me/durkachatbot?startgroup=botstart")
        ]
    ])

    # получаем текушую модель и способ вывода для этого чата
    model = get_model(mes.chat.id)
    text = f"Текущая модель: {models_ru[model]}\nROC AUC: {roc_aucs[model]}"

    output = get_output_method(mes.chat.id)
    text += "\n\nСпособ вывода:\n"
    if output or not model == 1:
        text += "вероятности диагнозов"
    else:
        text += "диагноз"
    
    # выводим сообщение
    await mes.answer(text, reply_markup=markup)

@router.my_chat_member(ChatMemberUpdatedFilter(member_status_changed=JOIN_TRANSITION))
async def bot_added(event: types.ChatMemberUpdated, bot: Bot):
    
    # id чата добавляется в базу, если его там нет
    add_id(event.chat.id)

    hello(event, True)

# обработка обытия нажатия на кнопку "Задать модель"
@dp.callback_query(F.data == "set_model")
async def callback_message(callback: types.CallbackQuery):
    await set_model(callback.message)

# обработка обытия нажатия на кнопку "Задать способ вывода"
@dp.callback_query(F.data == "set_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output(callback.message)

# обработка команды /info
@dp.message(Command("info"))
async def info(mes: types.Message):

    # сообщение для лички
    if mes.chat.type == "private":
        text = ("Я определяю психическое расстройство у автора сообщения методами машинного обучения. "
                       "Просто напиши сюда, что думаешь, и я попробую поставить тебе диагноз.\n\n" 
                       "Я обучался на датаесе англоязычных сообщений из сотсетей\n"
                       "https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health\n\n"
                       "Мой репозиторий открыт и каждый может убедится, что ваши сообщения остаются только между нами)\n"
                       "https://github.com/maxusha-zaytsusha/durkachatbot\n\n"
                       "Так же меня можно добавить в группу и я буду говорить, какое расстройство у автора сообщения!"
                       "Чтобы я мог видеть сообщения в группе, сделай меня администратором, и я буду ставить диагноз сообщениям, начинающихся со слова 'Санитар'. "
                       "Так же ты можешь ответить на сообщение словом 'Санитар' и я поставлю диагноз автору сообщения\n\n"
                       "Автор @MRabbit")
        
    # сообщение для группы
    else:
        text = ("Я определяю психическое расстройство у автора сообщения методами машинного обучения. "
                       "Чтобы я мог видеть сообщения в группе, сделай меня администратором, и я буду ставить диагноз сообщениям, начинающихся со слова 'Санитар'. " 
                       "Так же ты можешь ответить на сообщение словом 'Санитар' и я поставлю диагноз автору сообщения.\n\n"
                       "Я обучался на датаесе англоязычных сообщений из сотсетей\n" 
                       "https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health\n\n"
                       "Мой репозиторий открыт и каждый может убедится, что ваши сообщения остаются только между нами)\n"
                       "https://github.com/maxusha-zaytsusha/durkachatbot\n\n"
                       "Автор @MRabbit")

    # отправляем сообщение с информацией о боте
    await mes.answer(text)

# обработка команды /set_model 
@dp.message(Command("set_model"))
async def set_model(mes: types.Message):

    # делаем кнопки
    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Линейная модель", callback_data="linear")
        ],
        [
            types.InlineKeyboardButton(text = "Градиентный бустинг", callback_data="grad_boost"),
            types.InlineKeyboardButton(text = "Нейронная сеть", callback_data="neuro_network")
        ]
    ])

    # делаем текст сообщения с табличкой
    text = ("Выбери модель\n\n" + 
                   "<pre>"+
                   "|       Модель        |  ROC AUC |\n" + 
                   "|---------------------|----------|\n" + 
                   "|   линейная модель   |   0.93   |\n" + 
                   "| градиентный бустинг |   0.97   |\n" + 
                   "|   нейронная сеть    |   0.94   |</pre>")

    # выводим сообщение
    await mes.answer(text, reply_markup=markup, parse_mode="HTML")

# обработка нажатия на кнопку "Линейная модель"
@dp.callback_query(F.data == "linear")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(0, callback)

# обработка нажатия на кнопку "Градиентный бустинг"
@dp.callback_query(F.data == "grad_boost")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(1, callback)

# обработка нажатия на кнопку "Нейронная сеть"
@dp.callback_query(F.data == "neuro_network")
async def callback_message(callback: types.CallbackQuery):
    await set_model_by_id(2, callback)

# обработка команды /set_output 
@dp.message(Command("set_output"))
async def set_output(mes: types.Message):

    # id чата добавляется в базу, если его там нет
    add_id(mes.chat.id)

    # делаем кнопки
    markup = types.InlineKeyboardMarkup(inline_keyboard = [
        [
            types.InlineKeyboardButton(text = "Показывать вероятности", callback_data="proba_output"),
            types.InlineKeyboardButton(text = "Показывать диагноз", callback_data="diagnosis_output")
        ]
    ])

    text = ("Выбери, как ты хочешь, чтобы отвечал бот.\n" + 
                   "Бот может писать диагноз или показывать вроятности каждого диагноза для текста.")

    # выводим сообщение
    await mes.answer(text, reply_markup=markup)

# обработка нажатия на кнопку "Показывать вероятности"
@dp.callback_query(F.data == "proba_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output_method(1, callback)

# обработка нажатия на кнопку "Показывать диагноз"
@dp.callback_query(F.data == "diagnosis_output")
async def callback_message(callback: types.CallbackQuery):
    await set_output_method(0, callback)


# обработка сообщения, начинающегося со слова "Санитар"
@dp.message(lambda message: message.text.startswith('Санитар') if message.text else False)
async def сorpsman(mes: types.Message):

    # id чата добавляется в базу, если его там нет
    add_id(mes.chat.id)

    prefix = "у тебя "

    # если сообщение в личке, то анализируем текст полностью
    if mes.chat.type == 'private':
        await request(mes, mes.text, prefix)
    else:

        # убираем из текста "Санитар", а так же запятую и пробел после него, если есть 
        text = mes.text[7:]
        if text and text[0] == ",": text = text[1:]
        if text and text[0] == " ": text = text[1:]

        # если больше ничего не осталось в сообщении, проверяем, является ли сообщение ответом
        if not text:
            if mes.reply_to_message:

                # если является, анализируем сообщение, на которое ответили
                prefix = "у него "
                text = mes.reply_to_message.text
            else:
                return

        # выводим диагноз
        await request(mes, text, prefix)
        

# обработа собщения
@dp.message()
async def some_message(mes: types.Message):
    
    # если сообщение из нового чата (бота только что добавили) добавляем id в базу и пишем приветствие
    if is_new_chat(mes.chat.id):
        add_id(mes.chat.id)
        await hello(mes, True)
    else:
        # иначе, если сообщение в личке, выводим диагноз
        if mes.chat.type == 'private':
            await request(mes, mes.text, "у тебя ")


# функция вывода диагноза
async def request(mes: types.Message, text, prefix):
    
    # получаем модель для заданного id чата
    model = models[get_model(mes.chat.id)]

    # делаем предсказание
    pred = predict(text, model)

    # выводим диагноз согласно методу для данного id
    if get_output_method(mes.chat.id):
        await mes.answer(format_probas(pred))
    else:
        await mes.answer(prefix + probas_to_class(pred))


# функция предсказания диагноза потексту text с помощью модели model
def predict(text, model):

    # переводим текст на английский
    text = translator.translate(text, dest="en").text

    # делаем эмбединги
    embeding = emb_model.encode([text])

    # предсказываем значения softmax
    if model == neuro_network_model:
        res = model.predict(embeding, verbose=0)
    else:
        res = model.predict_proba(embeding)
    return res[0]


# функция делает строку с вероятностями по массиву вероятностей proba
def format_probas(proba):

    res = ""

    # проодимся по массиву и добавляем в строку диагнозы в определенном порядке
    for i in range(7):
        res += f'{classes[class_order[i]]}:  {proba[class_order[i]]:.2}\n'
    return res[:-1]


# функция находит самый вероятный диагноз и выводит его
def probas_to_class(proba):

    i_max = 0
    val_max = proba[0]

    for i in range(1, 7):
        if proba[i] > val_max:
            val_max = proba[i]
            i_max = i
    return classes[i_max]


# функция добавляет строку с новым id в базу, если его еще нет
# в базе хранятся id чатов, индексы моделеи и индексы способов вывода для каждого чата
def add_id(user_id):

    # запрашиваем строку с поданным id
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = " + str(user_id))
    row = cur.fetchall()

    # если строки нет, добавляем ее с дефолтными значениями в базу
    if not row:
        cur.execute("INSERT INTO users (id, model, output_method) VALUES (" + str(user_id) + ", 1, 0)")
        conn.commit()


# функция возвращает True, если данного id нет в базе, иначе False
def is_new_chat(user_id):

    # запрашиваем строку с поданным id
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = " + str(user_id))
    row = cur.fetchall()

    # если строки нет, добавляем ее с дефолтными значениями в базу
    if not row:
        return True
    return False


# функция задает модель для определенного id чата
async def set_model_by_id(model_i, callback):

    # меняем значение индекса модели для заданного id в базе
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("UPDATE users SET model = " + str(model_i) + " WHERE id = " + str(callback.message.chat.id))
    conn.commit()

    # выводим всплывающий текст
    await callback.answer("Выбранно: " + models_ru[model_i])


# функция получает индекс модели для заданного id
def get_model(user_id):

    # получаем индекс из базы
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT model FROM users WHERE id = " + str(user_id))

    return cur.fetchall()[0][0]


# функция получает способ вывода для заданного id
def get_output_method(user_id):

    # получаем индекс способа вывода
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("SELECT output_method FROM users WHERE id = " + str(user_id))

    return cur.fetchall()[0][0]


# функция задает способ вывода для определенного id чата
async def set_output_method(method, callback):

    # задаем новый индекс вывода для id в базе
    conn = sqlite3.connect("base.sql")
    cur = conn.cursor()
    cur.execute("UPDATE users SET output_method = " + str(method) + " WHERE id = " + str(callback.message.chat.id))
    conn.commit()

    # выводим всплывающий текст
    if method:
        await callback.answer("Теперь бот выводит вероятности")
    else:
        await callback.answer("Теперь бот выводит диагноз")


# функция выводит всплывающей текст об ошибке
async def error(callback):

    await callback.answer("Произошла ошибка!")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())