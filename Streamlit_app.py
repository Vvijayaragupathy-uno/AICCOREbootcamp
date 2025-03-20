import streamlit as st
import os
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# Import exercises if needed
from excerise_1 import exercise1
from excerise_2 import exercise2

# ✅ Ensure session state is initialized
if "page" not in st.session_state:
    st.session_state.page = "home"  # Start at the main home page

def navigate():
    """Handles page navigation logic."""
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "week1":
        display_week1_exercises()
        back_to_home()
    elif st.session_state.page == "week2":
        display_week2_exercises()
        back_to_home()
    elif st.session_state.page == "week3":
        display_week3_exercises()
        back_to_home()
    elif st.session_state.page == "week4":
        display_week4_exercises()
        back_to_home()
    elif st.session_state.page == "exercise1":
        exercise1()
        back_to_home()
    elif st.session_state.page == "exercise2":
        exercise2()
        back_to_home()

def show_home():
    """Displays the main home page with navigation buttons."""
    st.markdown("<h1 style='text-align: center;'>AI BOOTcamp Lab</h1>", unsafe_allow_html=True)

    container = st.container(height=300)

    col1, col2 = container.columns(2)

    with col1:
        if st.button("Week 1: LLM Performance Comparison", use_container_width=True):
            st.session_state.page = "week1"
            st.rerun()

    with col2:
        if st.button("Week 2: Prompt Engineering Techniques", use_container_width=True):
            st.session_state.page = "week2"
            st.rerun()

    col3, col4 = container.columns(2)

    with col3:
        if st.button("Week 3", use_container_width=True):
            st.session_state.page = "week3"
            st.rerun()

    with col4:
        if st.button("Week 4", use_container_width=True):
            st.session_state.page = "week4"
            st.rerun()

def display_week1_exercises():
    """Displays exercises for Week 1."""
    st.markdown("<h2 style='text-align: center;'>Week 1 Exercises</h2>", unsafe_allow_html=True)
    
    container = st.container(height=300)
    
    col1, col2, col3, col4 = container.columns(4)

    with col1:
        if st.button("Exercise 1", use_container_width=True):
            st.session_state.page = "exercise1"
            st.rerun()

    with col2:
        if st.button("Exercise 2", use_container_width=True):
            st.session_state.page = "exercise2"
            st.rerun()

    with col3:
        if st.button("Exercise 3", use_container_width=True):
            st.write("Exercise 3 is not yet available.")

    with col4:
        if st.button("Exercise 4", use_container_width=True):
            st.write("Exercise 4 is not yet available.")

def display_week2_exercises():
    """Displays exercises for Week 2."""
    st.markdown("<h2 style='text-align: center;'>Week 2 Exercises Coming Soon</h2>", unsafe_allow_html=True)

def display_week3_exercises():
    """Displays exercises for Week 3."""
    st.markdown("<h2 style='text-align: center;'>Week 3 Exercises Coming Soon</h2>", unsafe_allow_html=True)

def display_week4_exercises():
    """Displays exercises for Week 4."""
    st.markdown("<h2 style='text-align: center;'>Week 4 Exercises Coming Soon</h2>", unsafe_allow_html=True)

def back_to_home():
    """Adds a button to go back to the home page."""
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ✅ Call navigation function to route the pages
navigate()
