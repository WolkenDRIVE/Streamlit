import streamlit as st
import os
#pip install os-sys
import csv
import pandas as pd
import io
import pandas_profiling
import openai
import requests
import json
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model
st.set_page_config(layout="wide", page_title="MyStreamlit", page_icon="🧊")

yagpt_folder_id = "b1gvh5i9c7bk2s20uvpu"
yagpt_api_key = "AQVNz19Pq3E0NXWOkxoVK4NiowvmqOjfl3NGZzos"
def yagpt_get_response(question):
    prompt = {
        "modelUri": f"gpt://{yagpt_folder_id}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "user",
                "text": question
            }
        ]
    }
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {yagpt_api_key}",
    }

    response = requests.post(url, headers=headers, json=prompt)
    result = response.text
    result_obj = json.loads(result)
    return result_obj["result"]["alternatives"][0]["message"]["text"]


#from streamlit_extras.buy_me_a_coffee import button 

#button(username="Geeks", floating=False, width=250)

with st.sidebar:
    st.image("босс.jpg")
    st.title("Solo")
    choice = st.radio("Навигация", ["Загрузка", "Анализ", "Машинное обучение", "Скачать", "Чат"])

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('./dataset.csv', index_col=None)

if choice == "Загрузка":
    st.title("Загрузка данных")
    file = st.file_uploader("Загрузите свой набор данных", type="csv")
    if file is not None:
        try:
            data = io.StringIO(file.read().decode("utf-8"))
            df = pd.read_csv(data, sep=",", index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)
        except  pd.errors.EmptyDataError:
            st.error("Файл csv пустой")
               


#pip install pandas_profiling
#pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

if choice == "Анализ":
    st.title("Автоматизированный исследовательский анализ данных")
    prolife_df = df.profile_report()
    st_profile_report(prolife_df)

#pip install pycaret

if choice == "Машинное обучение":
    chosen_targets = st.multiselect('Выберите целевые столбцы', df.columns)
    if st.button('Запустить обучение'):
        for target in chosen_targets:
            st.subheader(f"Модель для целевого столбца: {target}")
            df_target = df.dropna(subset=[target])
            setup(df_target, target=target, verbose=False)
            setup_df = pull()
            st.info("Это настройка эксперимента Solo")
            st.dataframe(setup_df.dropna())
            best_model = compare_models()  
            compare_df = pull()
            st.info("Это модель Solo") 
            st.dataframe(compare_df.dropna())
            best_model
            save_model(best_model, f'best_model{target}')

file_path = 'best_model.pkl'    

if choice == "Скачать":
    st.title("Ваша обученная модель скачалась в папку проекта")
    with open(file_path, 'rb') as file:
        st.download_button('Скачать модель', file, file_name="best_model.pkl")

if choice == "Чат":
    #with st.chat_message(name="assistant", avatar="🧊"):
        #st.write("Привет")
    #openai.api_key=st.secrets["OPENAI_API_KEY"]
    #if "openai_model" not in st.session_state:
        #st.session_state["openai_model"] = "gpt-3.5"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    if prompt := st.chat_input("Что хочеть узнать?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = yagpt_get_response(prompt)
            #for response in openai.ChatCompletion.create(
                #model=st.session_state["openai_model"],
                #messages=[
                    #{"role": m["role"], "content": m["content"]}
                    #for m in st.session_state.messages
                #],
                #stream=True,
            #):
                #full_response += response.choices[0].delta.get("content", "")
                #message_placeholder.markdown(full_response + " ")
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

#st.write("Привет мир!")