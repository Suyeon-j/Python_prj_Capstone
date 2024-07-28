import cv2
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import os

def cos_sim(emb1, emb2):
    emb1_array = np.array(emb1)
    emb2_array = np.array(emb2)
    similarity = cosine_similarity(emb1_array.reshape(1, -1), emb2_array.reshape(1, -1))
    return similarity[0][0]

def get_expression(img):
    feature_extractor = ViTImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
    model = ViTForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
    
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    class_names = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    return class_names[predicted_class_idx]

def plot_img(test_img, group_similarities):
    result_images = []
    result_captions = []
    for group, similarity_score in group_similarities[:3]:
        img_path = next(f for f in os.listdir('./m_img') if f.startswith(group))
        img = Image.open(os.path.join('./m_img', img_path))
        result_images.append(img)
        result_captions.append(f'{group}\n: {similarity_score:.2f}')
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(test_img, caption='Test Image', use_column_width=True)
    with col2:
        st.image(result_images[0], caption=result_captions[0], use_column_width=True)
    with col3:
        st.image(result_images[1], caption=result_captions[1], use_column_width=True)
    with col4:
        st.image(result_images[2], caption=result_captions[2], use_column_width=True)

def cos_sim(emb1, emb2):
    emb1_array = np.array(emb1)
    emb2_array = np.array(emb2)
    similarity = cosine_similarity(emb1_array.reshape(1, -1), emb2_array.reshape(1, -1))
    return similarity[0][0]

def sim1(input_img_path):
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    df = pd.read_csv("./facenet/m_emb_mtcnn.csv")

    group_embeddings = {}
    for _, row in df.iterrows():
        filename = row['filename']
        group = ''.join([char for char in filename if not char.isdigit()]).replace('.jpg', '')
        emb = np.array(eval(row['emb']))
        if emb.shape[0] == 512:
            if group not in group_embeddings:
                group_embeddings[group] = []
            group_embeddings[group].append(emb)

    avg_group_embeddings = {group: np.mean(embs, axis=0) for group, embs in group_embeddings.items()}

    img = Image.open(input_img_path)
    img = img.convert('RGB')

    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        img_aligned = mtcnn(img)
        embedding1 = resnet(img_aligned).detach().numpy().reshape(-1)

        if embedding1.shape[0] == 1024:
            embedding1 = embedding1[:512]

        group_sim = []
        for group, avg_emb in avg_group_embeddings.items():
            avg_emb = np.array(avg_emb).reshape(-1)
            similarity = cos_sim(embedding1, avg_emb)
            group_sim.append((group, similarity))

        group_sim.sort(key=lambda x: x[1], reverse=True)
        plot_img(img, group_sim)

    else:
        st.write('No faces detected in the input image.')


def sim2(input_img1, input_img2):
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    df = pd.read_csv("./facenet/m_emb_mtcnn.csv")

    embeddings_dict = {}
    for _, row in df.iterrows():
        filename = row['filename']
        group = ''.join([char for char in filename if not char.isdigit()]).replace('.jpg', '')
        emb = eval(row['emb'])
        embeddings_dict[filename] = (group, emb)

    def get_embedding(img_array):
        img = Image.fromarray(img_array)
        img = img.convert('RGB')
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            img_aligned = mtcnn(img)
            embedding = resnet(img_aligned).detach().numpy()
            return embedding
        else:
            st.write('No faces detected in the input image.')
            return None

    embedding1 = get_embedding(input_img1)
    embedding2 = get_embedding(input_img2)

    if embedding1 is None or embedding2 is None:
        return

    sim_scores1 = []
    sim_scores2 = []

    for filename, (group, emb) in embeddings_dict.items():
        sim1 = cos_sim(embedding1, emb)
        sim2 = cos_sim(embedding2, emb)
        sim_scores1.append((group, filename, sim1))
        sim_scores2.append((group, filename, sim2))

    sim_scores1.sort(key=lambda x: x[2], reverse=True)
    sim_scores2.sort(key=lambda x: x[2], reverse=True)
    print("sim1", sim_scores1)
    print("sim2", sim_scores2)

    group_highest_sim = {}

    for group, filename, sim in sim_scores1:
        if group not in group_highest_sim or group_highest_sim[group] < sim:
            group_highest_sim[group] = sim

    for group, filename, sim in sim_scores2:
        if group not in group_highest_sim or group_highest_sim[group] < sim:
            group_highest_sim[group] = sim

    sorted_groups = sorted(group_highest_sim.items(), key=lambda x: x[1], reverse=True)
    top_groups = sorted_groups[:3]

    plot_img(Image.fromarray(input_img2), top_groups)

def capture_and_classify():
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    img1 = None
    img2 = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        emotion = get_expression(pil_image)

        text = f"Emotion: {emotion}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        text_x = (frame_rgb.shape[1] - text_width) // 2
        text_y = frame_rgb.shape[0] - 30

        frame_rgb = cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(frame_rgb)

        if emotion == 'neutral' and img1 is None:
            img1 = frame_rgb
            st.write("Neutral expression captured.")
        elif emotion == 'happy' and img2 is None:
            img2 = frame_rgb
            st.write("Happy expression captured.")

        if img1 is not None and img2 is not None:
            cap.release()
            break

    if img1 is None:
        st.write("Neutral expression not found. Please try again.")
    if img2 is None:
        st.write("Happy expression not found. Please try again.")

    return img1, img2