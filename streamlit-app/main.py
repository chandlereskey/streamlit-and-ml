import streamlit as st


pg = st.navigation([
    st.Page('todo.py', title='Todo List', icon='📝'),
    st.Page('number_prediction.py', title='Number Prediction', icon='🔍')
])
pg.run()