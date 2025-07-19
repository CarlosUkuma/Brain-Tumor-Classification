import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
from fpdf import FPDF
import datetime

# Carregar ou criar diretório do histórico
if not os.path.exists("historico_pacientes"):
    os.makedirs("historico_pacientes")

st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

st.title("🧠 Brain Tumor Classification")

# Sidebar - Lista de Pacientes
st.sidebar.subheader("Lista de Pacientes")
pacientes = [f"Paciente {i}" for i in range(1, 11)]
paciente_selecionado = st.sidebar.selectbox("Selecione um paciente", pacientes)

# Função para carregar modelo
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# Upload do modelo
modelo_path = st.sidebar.file_uploader("Carregar modelo (.h5)", type=["h5"])
if modelo_path:
    modelo = load_model('/content/drive/MyDrive/GoogleColaboratory/brain_tumor_inceptionv3.h5')

# Upload de imagem
col1, col2 = st.columns(2)
with col1:
    st.subheader("Carregue a Imagem de Ressonância")
    imagem_file = st.file_uploader("Upload da imagem", type=["png", "jpg", "jpeg"])
    if imagem_file:
        imagem = Image.open(imagem_file)
        st.image(imagem, caption="Imagem de Ressonância", width=250)

# Dados do paciente
with col2:
    st.subheader("Dados do paciente")
    nome = st.text_input("Nome")
    idade = st.number_input("Idade", min_value=0)
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
    tipo_sanguineo = st.selectbox("Tipo Sanguíneo", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    peso = st.number_input("Peso (kg)", min_value=0.0)

# Diagnóstico
st.subheader("Resultado do Diagnóstico")
if st.button("Gerar Diagnóstico") and imagem_file and modelo_path:
    img = imagem.resize((150, 150))  # depende do modelo
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = modelo.predict(img_array)
    diagnostico = f"Tumor Detected: {'Yes' if pred[0][0] > 0.5 else 'No'} (Confiança: {pred[0][0]:.2f})"

    st.success(diagnostico)

    # Salvar histórico
    historico_file = f"historico_pacientes/{paciente_selecionado}.csv"
    df = pd.DataFrame([{
        "Data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nome": nome,
        "Idade": idade,
        "Sexo": sexo,
        "Tipo Sanguíneo": tipo_sanguineo,
        "Peso": peso,
        "Diagnóstico": diagnostico
    }])

    if os.path.exists(historico_file):
        df_antigo = pd.read_csv(historico_file)
        df = pd.concat([df_antigo, df], ignore_index=True)

    df.to_csv(historico_file, index=False)

    st.dataframe(df)

    # Gerar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório Médico", ln=True, align="C")
    pdf.ln()

    for k, v in df.iloc[-1].items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

    pdf_path = f"historico_pacientes/{paciente_selecionado}_diagnostico.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Baixar PDF do Diagnóstico", f, file_name=os.path.basename(pdf_path))

# Mostrar histórico do paciente selecionado
if os.path.exists(f"historico_pacientes/{paciente_selecionado}.csv"):
    st.subheader(f"Histórico de {paciente_selecionado}")
    historico_df = pd.read_csv(f"historico_pacientes/{paciente_selecionado}.csv")
    st.dataframe(historico_df)
