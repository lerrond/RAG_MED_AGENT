import streamlit as st
from loguru import logger
import os

st.set_page_config(page_title = "MedBro", page_icon="👨🏻‍⚕️")
logger.add("log/st.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

st.title(" ‍🔬  ИИ-ассистент для студентов медицинских университетов")
st.sidebar.success(" ⬆️    Узнайте о проекте чуть больше ")

uploaded_file = st.sidebar.file_uploader("👇  При желании загрузите свой PDF-файл", type="pdf")

# загрузка своего пдф
if uploaded_file is not None:
    file_path = os.path.join("pdf/new", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Файл сохранён: {uploaded_file.name}")

question_input = st.chat_input("Введите вопрос:", key="input_text_field")
response_area = st.empty()

# если ввели какой-то вопрос, то запускаем файл мэйн
if question_input:
    logger.debug(f'question_input={question_input}')

    # отображаем ответ пользователя и временный ответ ассистента
    st.chat_message("user").markdown(question_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("_Изучаю книги..._")

    import main as sr

    @st.cache_data
    def load_all():
        db = sr.indexed_df()
        logger.debug('Данные загружены')
        return db

    #  подгружаем базу данных
    db = load_all()
    inputs = {"question": question_input, "max_retries": 3}

    # получаем ответ от модели
    for event in sr.graph.stream(inputs, stream_mode="values"):
        logger.debug(event)
        if "generation" in event and hasattr(event["generation"], "content"):
            model_response = event["generation"].content
            response_placeholder.markdown(f"**Ассистент:** {model_response}")
            break

    # кнопки лайка/дизлайка под ответом
    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     if st.button("👍", key="like_button"):
    #         logger.debug("Ответ получил лайк!")
    #         st.success("Спасибо за ваш отзыв! Будем работать дальше.")
    # with col2:
    #     if st.button("👎", key="dislike_button"):
    #         logger.debug("Ответ получил дизлайк!")
    #         st.warning("Спасибо за ваш отзыв! Мы будем работать над улучшением.")

st.sidebar.image("pages/pokemon-turtwig-turtwig.gif")