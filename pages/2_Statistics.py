import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Анализ анкет", layout="wide", page_icon="📊")
st.sidebar.success(" ⬆️    Узнайте о проекте чуть больше ")

st.title("📊 Анализ анкет студентов медицинских университетов")

st.markdown("В ходе выполнения проекта мы провели опрос среди студентов 2-го медицинского " 
            "университета - РНИМУ им. Н.И. Пирогова. Целью опроса было выявить самые "
            "сложные предметы, встречающиеся на пути у студентов до 3-го курса. Именно 3-ий курс считается в медицинских университетах "
            "_экватором_. После него учеба идет чуточку легче, чем это было _до_.")

st.markdown("В опросе приняло участие примерно 50 человек с педиатрического факультета - можете ознакомиться "
            "со статистикой ниже. Учебники по самым сложным предметам были использованы в дальнейшем для обучения модели. ")

st.divider()
# Загрузка данных
dataframe = pd.read_csv('Anketa_Dlya_Studentov_Meda.xls', header=0, names=['0', 'курс', '1 курс', '2 курс', '3 курс', 'time', 'find'])
dataframe.drop(columns='0', inplace=True)
dataframe.replace([
    'Всё и сразу',
    'все вместе, потому что каждый раз информация находится в разных источниках',
    'Всё вместе. '
], 'Всё вместе', inplace=True)

# 1. Где ищут информацию
st.header("🔍 Где студенты ищут информацию")
fig_find = px.pie(dataframe.groupby('find')['time'].count().reset_index(),
                  values='time', names='find', color_discrete_sequence=px.colors.sequential.Greens_r)
fig_find.update_traces(hovertemplate='%{label}')
fig_find.update_layout(
    legend=dict(
    font=dict(size=16),
    x=-0.75,
))
st.plotly_chart(fig_find, use_container_width=True)

st.divider()


st.header("🕒 Сколько времени уходит у студентов на поиск информации")
fig_time = px.pie(dataframe.groupby('time')['find'].count().reset_index(),
                  values='find', names='time', color_discrete_sequence=px.colors.sequential.Greens_r)
fig_time.update_traces(hovertemplate='%{label}')
fig_time.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
    font=dict(size=13)
))
st.plotly_chart(fig_time, use_container_width=True)

st.divider()

# 3. По курсам
st.header("🎓 Распределение по курсам")
fig_course = px.pie(dataframe.groupby('курс')['find'].count().reset_index(),
                    values='find', names='курс', color_discrete_sequence=px.colors.sequential.Greens_r)
fig_course.update_traces(hovertemplate='%{label}')
fig_course.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),
    legend=dict(
    font=dict(size=16),
    x=0.65,
))
st.plotly_chart(fig_course, use_container_width=True)

# Обработка курсов по сложностям
def process_course_column(column_name, default_text):
    data = dataframe[column_name].replace(np.nan, default_text).values
    split_data = np.array([])
    for item in data:
        split_data = np.append(item.split(';'), split_data)
    subjects, counts = np.unique(split_data, return_counts=True)
    return pd.DataFrame({'предмет': subjects, 'число': counts})

st.divider()

# 4. Сложности 1 курса
st.header("📘 Сложные предметы на 1 курсе")
s1_df = process_course_column('1 курс', '—')
fig_s1 = px.bar(s1_df.sort_values(by='число'),
                y='предмет', x='число', orientation='h',
                hover_data=['число'], color_discrete_sequence=px.colors.sequential.Plasma)
fig_s1.update_layout(yaxis_title=None)
st.plotly_chart(fig_s1, use_container_width=True)

st.divider()

# 5. Сложности 2 курса
st.header("📗 Сложные предметы на 2 курсе")
s2_df = process_course_column('2 курс', 'Еще не дошел до 2-го курса')
fig_s2 = px.bar(s2_df.sort_values(by='число'),
                y='предмет', x='число', orientation='h',
                hover_data=['число'], color_discrete_sequence=px.colors.sequential.Plasma)
fig_s2.update_layout(yaxis_title=None)
st.plotly_chart(fig_s2, use_container_width=True)

st.divider()

# 6. Сложности 3 курса
st.header("📙 Сложные предметы на 3 курсе")
s3_df = process_course_column('3 курс', 'Еще не дошел до 3-го курса')
fig_s3 = px.bar(s3_df.sort_values(by='число'),
                y='предмет', x='число', orientation='h',
                hover_data=['число'], color_discrete_sequence=px.colors.sequential.Plasma)
fig_s3.update_layout(yaxis_title=None)
st.plotly_chart(fig_s3, use_container_width=True)