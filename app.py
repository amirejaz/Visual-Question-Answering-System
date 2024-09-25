import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.set_page_config(layout="wide", page_title="VQA")

# ViLT code
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        img = Image.open(BytesIO(image)).convert("RGB")
        # prepare inputs
        encoding = processor(img, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer
    except Exception as e:
        return str(e)

st.title("Visual Question Answering System")
st.write("Upload an image and enter a question to get an answer")

col1, col2 = st.columns(2)

# Upload image
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:  # Check if an image was uploaded
        st.image(uploaded_file, use_column_width=True)

with col2:
    question = st.text_input("Question")

    if uploaded_file and question:  # Ensure both image and question are provided
        if st.button("ASK QUESTION"):
            image = Image.open(uploaded_file)
            img_byte_array = BytesIO()
            image.save(img_byte_array, format="jpeg")
            img_bytes = img_byte_array.getvalue()

            # Get the answer
            answer = get_answer(img_bytes, question)
            st.info(f"Your Question: {question}")
            st.success(f"Answer: {answer}")
