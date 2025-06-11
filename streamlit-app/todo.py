import streamlit as st

st.title('Todo List')

todo = st.text_input('Add your todo')

if st.button('Add Task'):
    if todo:
        st.session_state['todo_list'].append(
            {
                'title': todo,
                'complete': False
            }
        )
    todo = ''

def handle_checkbox_click(value, index):
    print(f'Checkbox clicked: {value} at index {index}')
    st.session_state['todo_list'][index]['complete'] = not value

if 'todo_list' not in st.session_state:
    st.session_state['todo_list'] = []

for i, todo in enumerate(st.session_state['todo_list']):
    st.checkbox('Complete?', todo['complete'], key=f'checkbox_{i}', on_change=handle_checkbox_click, args=(todo['complete'],i))
    if todo['complete']:
        if not st.button('Delete', key=f'delete_{i}', on_click=lambda idx=i: st.session_state['todo_list'].pop(idx)):
            st.success(todo['title'])
    else:
        st.info(todo['title'])
