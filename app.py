import streamlit as st

st.title("Streamlit Components Demo")

st.header("Welcome to the demo app!")
st.write("This app showcases various Streamlit components.")

if st.button("Click me"):
    st.write("Button clicked!")

st.checkbox("Check me")
st.radio("Choose one", options=["Option 1", "Option 2", "Option 3"])
st.selectbox("Select an option", options=["A", "B", "C"])
st.multiselect("Select multiple options", options=["X", "Y", "Z"])
st.slider("Slide me", min_value=0, max_value=100, value=50)
st.text_input("Enter some text")
st.text_area("Enter a longer text")
st.date_input("Pick a date")
st.time_input("Pick a time")
st.file_uploader("Upload a file")
