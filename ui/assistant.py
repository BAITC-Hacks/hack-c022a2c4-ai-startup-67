"""Small optional assistant; API requests happen only on explicit submission."""
import os
import streamlit as st
from assistant.service import answer_question, UNAVAILABLE


def setting(name, default=''):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def assistant_panel(tools, gid, data_version):
    st.markdown('#### Ask about this client')
    key = setting('OPENAI_API_KEY')
    if not key:
        st.info(UNAVAILABLE)
        return
    st.caption('По кнопке вопрос и выбранные результаты инструментов отправляются в OpenAI. '
               'AI объясняет готовые оценки; его ответ нужно сверять с данными.')
    with st.form('assistant_question', clear_on_submit=False):
        question = st.text_input('Вопрос', placeholder='Почему этот клиент в приоритете? Какие ограничения важны?', max_chars=2000)
        submitted = st.form_submit_button('Объяснить')
    if submitted:
        with st.spinner('Разбор сохранённых результатов…'):
            result = answer_question(question, gid, tools, api_key=key, model=setting('OPENAI_MODEL', 'gpt-4.1-mini'))
        st.session_state['ai_result'] = (gid, data_version, result)
    saved = st.session_state.get('ai_result')
    if saved and saved[:2] == (gid, data_version):
        result = saved[2]
        if result.get('error'):
            st.info(result['error'])
        else:
            st.markdown(result['answer'])
        if result['evidence']:
            with st.expander('Данные, использованные для ответа'):
                st.json(result['evidence'])
