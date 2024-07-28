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


def plot_img(test_img, group_similarities):
    result_images = []
    result_captions = []
    for group, similarity_score in group_similarities[:3]:
        img_path = next(f for f in os.listdir('./m_img_mtcnn') if f.startswith(group))
        img = Image.open(os.path.join('./m_img', img_path))
        result_images.append(img)
        result_captions.append(f'{group}\nSim: {similarity_score:.2f}')
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(test_img, caption='Test Image', use_column_width=True)
    with col2:
        st.image(result_images[0], caption=result_captions[0], use_column_width=True)
    with col3:
        st.image(result_images[1], caption=result_captions[1], use_column_width=True)
    with col4:
        st.image(result_images[2], caption=result_captions[2], use_column_width=True)


def sim(input_img):
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    df = pd.read_csv("./facenet/m_emb_mtcnn.csv")

    group_embeddings = {}
    for _, row in df.iterrows():
        filename = row['filename']
        group = ''.join([char for char in filename if not char.isdigit()]).replace('.jpg', '')
        emb = eval(row['emb'])
        if group not in group_embeddings:
            group_embeddings[group] = []
        group_embeddings[group].append(emb)

    avg_group_embeddings = {group: np.mean(embs, axis=0) for group, embs in group_embeddings.items()}

    img = Image.open(input_img)
    img = img.convert('RGB')

    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        img_aligned = mtcnn(img)
        embedding1 = resnet(img_aligned).detach().numpy()

        group_sim = []
        for group, avg_emb in avg_group_embeddings.items():
            similarity = cos_sim(embedding1, avg_emb)
            group_sim.append((group, similarity))

        group_sim.sort(key=lambda x: x[1], reverse=True)
        plot_img(img, group_sim)

    else:
        st.write('No faces detected in the input image.')