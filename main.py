# ========== Importação de Bibliotécas ========== #
import streamlit as st
import openai
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import datetime
import tempfile
import cv2
import joblib
from PIL import Image
from fpdf import FPDF
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ========== Configuração inicial ========== #
# Título da página web
st.set_page_config(
    page_title="Diagnóstico de Tumores Cerebrais", layout="wide")

# Criação da pasta para salvar o histórico
if not os.path.exists("Histórico de Pacientes"):
    os.makedirs("Histórico de Pacientes")

# Validação do Bilhete de Identidade


def validar_bi(bi):
    return bool(re.match(r"^\d{9}[A-Z]{2}\d{3}$", bi))

# Verificação da existência de um B.I idêntico


def bi_ja_existe(bi):
    for file in os.listdir("Histórico de Pacientes"):
        if file.endswith(" Histórico.csv"):
            df = pd.read_csv(f"Histórico de Pacientes/{file}")
            if "Bilhete de Identidade" in df.columns and bi in df["Bilhete de Identidade"].values:
                return True
    return False

# Função de leitura do modelo


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# Função de leitura do label encoder


@st.cache_resource
def load_label_encoder(path):
    """Carregar o label encoder salvo durante o treinamento"""
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não foi possível carregar o label encoder: {e}")
        # Retornar encoder padrão baseado na ordem do treinamento
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.classes_ = np.array(
            ['glioma', 'meningioma', 'notumor', 'pituitary'])
        return encoder

# FUNÇÃO CORRIGIDA PARA PROCESSAR IMAGEM


def processar_imagem_para_predicao(image_pil):
    """
    Processa a imagem exatamente como no treinamento

    Args:
        image_pil: Imagem PIL

    Returns:
        np.array: Imagem processada pronta para predição
    """
    try:
        # Converter PIL para numpy array (já em RGB)
        img_array = np.array(image_pil)

        # Garantir que é RGB (3 canais)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Redimensionar para 224x224 (exatamente como no treinamento)
        img_array = cv2.resize(img_array, (224, 224))

        # Converter para float32
        img_array = img_array.astype(np.float32)

        # Expandir dimensão do batch
        img_array = np.expand_dims(img_array, axis=0)

        # CRÍTICO: Aplicar o mesmo pré-processamento do InceptionV3
        img_array = preprocess_input(img_array)

        return img_array

    except Exception as e:
        st.error(f"Erro no processamento da imagem: {e}")
        return None

# FUNÇÃO PARA FAZER PREDIÇÃO SEGURA


def fazer_predicao(modelo, img_array, label_encoder):
    """
    Fazer predição com tratamento de erro

    Args:
        modelo: Modelo TensorFlow carregado
        img_array: Imagem processada
        label_encoder: Encoder dos rótulos

    Returns:
        tuple: (classe_predita, confianca, todas_probabilidades)
    """
    try:
        # Fazer predição
        pred = modelo.predict(img_array, verbose=0)

        # Extrair resultados
        classe_index = np.argmax(pred)
        confianca = float(pred[0][classe_index])

        # Obter nome da classe
        classe_predita = label_encoder.classes_[classe_index]

        # Mapear nomes das classes para português
        mapeamento_classes = {
            'glioma': 'Glioma',
            'meningioma': 'Meningioma',
            'notumor': 'Nenhum Tumor',
            'pituitary': 'Pituitário'
        }

        classe_portugues = mapeamento_classes.get(
            classe_predita, classe_predita)

        # Criar dicionário com todas as probabilidades
        todas_probs = {}
        for i, classe in enumerate(label_encoder.classes_):
            nome_pt = mapeamento_classes.get(classe, classe)
            todas_probs[nome_pt] = float(pred[0][i])

        return classe_portugues, confianca, todas_probs

    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, 0, {}

# ========== Carregamento Automático do Modelo ========== #


@st.cache_resource
def carregar_modelo_automatico():
    """Carrega o modelo e label encoder automaticamente da pasta"""
    modelo = None
    label_encoder = None

    # Tentar carregar o modelo padrão
    if os.path.exists("brain_tumor_inceptionv3.h5"):
        try:
            modelo = load_model("brain_tumor_inceptionv3.h5")
        except Exception as e:
            st.error(f"❌ Erro ao carregar o modelo: {e}")
    else:
        st.error(
            "❌ Modelo 'brain_tumor_inceptionv3.h5' não encontrado na pasta do projeto!")

    # Tentar carregar o label encoder
    if os.path.exists("label_encoder.pkl"):
        try:
            label_encoder = load_label_encoder("label_encoder.pkl")
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar label encoder: {e}")
    else:
        st.warning(
            "⚠️ Arquivo 'label_encoder.pkl' não encontrado. Usando encoder padrão.")

    return modelo, label_encoder


# ========== Cabeçalho ========== #
col1, col2, col3 = st.columns([2, 6, 1])

with col2:
    st.title("Diagnóstico de Tumores Cerebrais")
st.divider()

# ========== Sidebar com ícone e informações do modelo ========== #
icon = Image.open(
    "Imagens/brain.png") if os.path.exists("Imagens/brain.png") else None
if icon:
    st.sidebar.image(icon, width=250)
    st.sidebar.divider()

# Carregamento automático do modelo
st.sidebar.markdown("## Status do Sistema")
modelo, label_encoder = carregar_modelo_automatico()

# Verificar se o modelo foi carregado com sucesso
if modelo is not None and label_encoder is not None:
    st.sidebar.success("🟢 Pronto para o diagnóstico!")
else:
    st.sidebar.error("🔴 O sistema não está pronto!")
    st.sidebar.markdown("**Verifique se os arquivos estão na pasta:**")
    st.sidebar.markdown("- `brain_tumor_inceptionv3.h5`")
    st.sidebar.markdown("- `label_encoder.pkl`")

# ========== Upload de imagem ========== #
col4, col5 = st.columns(2)

with col4:
    st.subheader("Imagem de Ressonância Magnética")
    imagem_file = st.file_uploader(
        "Faça o upload da imagem de ressonância magnética aqui", type=["png", "jpg", "jpeg"])
    if imagem_file:
        imagem = Image.open(imagem_file)
        st.image(imagem, width=180)
        st.success("Imagem carregada com sucesso!")
    else:
        st.info("Nenhuma imagem carregada.")

# ========== Registro e seleção de pacientes ========== #
with col5:
    st.subheader("Dados do Paciente")

    # Lista de pacientes
    arquivos_csv = [f for f in os.listdir(
        "Histórico de Pacientes") if f.endswith(" Histórico.csv")]
    pacientes_registrados = [
        f.replace(" Histórico.csv", "") for f in arquivos_csv]
    paciente_selecionado = st.sidebar.selectbox(
        "Consulte por pacientes registrados", [""] + sorted(pacientes_registrados))

    if not paciente_selecionado:
        st.sidebar.info("Registre um novo paciente.")
        st.session_state.show_historico = False

    # Variáveis padrão
    nome_val = ""
    bi_val = ""
    idade_val = 0
    sexo_val = ""
    tipo_sang_val = ""
    peso_val = 0.0

    # Preenchimento automático se paciente for selecionado
    if paciente_selecionado:
        hist_path = f"Histórico de Pacientes/{paciente_selecionado} Histórico.csv"
        if os.path.exists(hist_path):
            dados = pd.read_csv(hist_path)
            if not dados.empty:
                ultimo = dados.iloc[-1]
                nome_val = ultimo.get("Nome", "")
                bi_val = ultimo.get("Bilhete de Identidade", "")
                idade_val = ultimo.get("Idade", "")
                sexo_val = ultimo.get("Sexo", "")
                tipo_sang_val = ultimo.get("Tipo Sanguíneo", "")
                peso_val = ultimo.get("Peso", "")

    # Campos
    nome = st.text_input("Nome", value=nome_val,
                         disabled=bool(paciente_selecionado))
    bi = st.text_input("Bilhete de Identidade (Ex: 123456789AB123)",
                       value=bi_val, disabled=bool(paciente_selecionado))
    idade = st.number_input("Idade", min_value=0, value=idade_val)
    sexo = st.selectbox("Sexo", ["", "Masculino", "Feminino"], index=[
                        "", "Masculino", "Feminino"].index(sexo_val), disabled=bool(paciente_selecionado))
    tipo_sanguineo = st.selectbox("Tipo Sanguíneo", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=[
                                  "", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(tipo_sang_val), disabled=bool(paciente_selecionado))
    peso = st.number_input("Peso (kg)", min_value=0.0, value=peso_val)

    # Inicializar variáveis de controle no estado
    if "modo_edicao" not in st.session_state:
        st.session_state.modo_edicao = False
    if "modo_exclusao" not in st.session_state:
        st.session_state.modo_exclusao = False

    if paciente_selecionado:
        col_edit, col_delete = st.columns(2)

        if col_edit.button("Editar Paciente"):
            st.session_state.modo_edicao = True
            st.session_state.modo_exclusao = False

        if col_delete.button("Excluir Paciente"):
            st.session_state.modo_exclusao = True
            st.session_state.modo_edicao = False

    # Modo de edição
    if st.session_state.modo_edicao and paciente_selecionado:
        st.divider()
        st.subheader("Editar Informações do Paciente")

        novo_nome = st.text_input("Nome", value=nome_val, key="edit_nome")
        novo_bi = st.text_input("Bilhete de Identidade",
                                value=bi_val, key="edit_bi")
        nova_idade = st.number_input(
            "Idade", min_value=0, value=int(idade_val), key="edit_idade")
        novo_sexo = st.selectbox("Sexo", ["", "Masculino", "Feminino"], index=[
                                 "", "Masculino", "Feminino"].index(sexo_val), key="edit_sexo")
        novo_tipo_sang = st.selectbox("Tipo Sanguíneo", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=[
                                      "", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(tipo_sang_val), key="edit_tipo_sang")
        novo_peso = st.number_input(
            "Peso (kg)", min_value=0.0, value=float(peso_val), key="edit_peso")

        col_save, col_cancel = st.columns(2)
        if col_save.button("Salvar Edição"):
            erros = []

            # Validações dos campos
            if not novo_nome:
                erros.append("O campo 'Nome' é obrigatório.")
            if not novo_bi:
                erros.append("O campo 'Bilhete de Identidade' é obrigatório.")
            elif not validar_bi(novo_bi):
                erros.append("Formato de B.I. inválido. Ex: 123456789AB123")
            elif novo_bi != bi_val and bi_ja_existe(novo_bi):
                erros.append("Este B.I. já está registrado em outro paciente.")
            if nova_idade <= 0:
                erros.append("Idade inválida.")
            if not novo_sexo:
                erros.append("O campo 'Sexo' é obrigatório.")
            if not novo_tipo_sang:
                erros.append("O campo 'Tipo Sanguíneo' é obrigatório.")
            if novo_peso <= 0:
                erros.append("Peso inválido.")

            if erros:
                for erro in erros:
                    st.error(erro)
            else:
                try:
                    df = pd.read_csv(hist_path)
                    df["Nome"] = novo_nome
                    df["Bilhete de Identidade"] = novo_bi
                    df["Idade"] = nova_idade
                    df["Sexo"] = novo_sexo
                    df["Tipo Sanguíneo"] = novo_tipo_sang
                    df["Peso"] = novo_peso

                    novo_path = f"Histórico de Pacientes/{novo_nome} Histórico.csv"

                    # Verifica se novo nome já existe para evitar sobrescrita
                    if novo_nome != nome_val and os.path.exists(novo_path):
                        st.error("Já existe um paciente com esse nome.")
                    else:
                        df.to_csv(novo_path, index=False)

                        if novo_path != hist_path:
                            os.remove(hist_path)

                        st.success("Alterações salvas com sucesso.")
                        st.session_state.modo_edicao = False
                        st.rerun()

                except Exception as e:
                    st.error(f"Erro ao salvar alterações: {e}")

        if col_cancel.button("Cancelar Edição"):
            st.session_state.modo_edicao = False
            st.rerun()

    # Modo de exclusão
    if st.session_state.modo_exclusao and paciente_selecionado:
        st.divider()
        st.warning(
            "Tem certeza que deseja excluir este paciente? Essa ação é irreversível.")

        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button("Confirmar Exclusão"):
            try:
                os.remove(hist_path)
                st.success(
                    f"Paciente '{paciente_selecionado}' excluído com sucesso.")
                st.session_state.modo_exclusao = False
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao excluir: {e}")

        if col_cancel.button("Cancelar Exclusão"):
            st.session_state.modo_exclusao = False
            st.rerun()

# ========== Diagnóstico ========== #
st.divider()
if st.button("Gerar Diagnóstico"):
    if not imagem_file:
        st.error("Por favor, carregue uma imagem.")
    elif not nome:
        st.error("Por favor, preencha o campo 'Nome'.")
    elif not bi:
        st.error("Por favor, preencha o campo 'Bilhete de Identidade'.")
    elif idade == 0:
        st.error("Por favor, preencha o campo 'Idade'.")
    elif sexo == "":
        st.error("Por favor, preencha o campo 'Sexo'.")
    elif tipo_sanguineo == "":
        st.error("Por favor, preencha o campo 'Tipo Sanguíneo'.")
    elif peso == 0:
        st.error("Por favor, preencha o campo 'Peso'.")
    elif modelo is None:
        st.error(
            "Modelo não carregado. Verifique se o arquivo 'brain_tumor_inceptionv3.h5' está na pasta do projeto.")
    elif label_encoder is None:
        st.error("Label encoder não carregado.")
    elif not validar_bi(bi):
        st.error(
            "Formato de BI inválido! Ele deve conter 14 caracteres: 9 dígitos + 2 letras + 3 dígitos (Ex: 123456789AB123).")
    elif not paciente_selecionado and bi_ja_existe(bi):
        st.error(
            "Este Bilhete de Identidade já está registrado. Selecione o paciente correspondente para novo diagnóstico ou registre um novo paciente.")
    else:
        # PROCESSAR IMAGEM
        with st.spinner("Registrando o paciente, processando imagem e gerando diagnóstico..."):
            try:
                # Redimensionar imagem para exibição (mantém original para processamento)
                img_display = imagem.resize((224, 224))

                # Processar imagem para predição (FUNÇÃO CORRIGIDA)
                img_array = processar_imagem_para_predicao(img_display)

                if img_array is not None:
                    # Fazer predição (FUNÇÃO CORRIGIDA)
                    classe_predita, confianca, todas_probs = fazer_predicao(
                        modelo, img_array, label_encoder)

                    if classe_predita:
                        # Exibir resultado principal
                        diagnostico = f"Tumor identificado: {classe_predita} - Confiança no diagnóstico: {confianca:.2%}"

                        # Determinar cor baseada na confiança
                        if confianca >= 0.8:
                            st.success(diagnostico)
                        elif confianca >= 0.6:
                            st.warning(diagnostico)
                        else:
                            st.info(diagnostico)

                        # Mostrar todas as probabilidades
                        st.subheader("Probabilidades por Classe:")
                        prob_df = pd.DataFrame(list(todas_probs.items()),
                                               columns=['Classe', 'Probabilidade'])
                        prob_df['Probabilidade (%)'] = prob_df['Probabilidade'] * 100
                        prob_df = prob_df.sort_values(
                            'Probabilidade', ascending=False)

                        # Criar gráfico de barras
                        st.bar_chart(prob_df.set_index(
                            'Classe')['Probabilidade'])

                        # Tabela com probabilidades
                        st.dataframe(
                            prob_df[['Classe', 'Probabilidade (%)']].round(2))

                        # Salvar histórico
                        historico_file = f"Histórico de Pacientes/{nome} Histórico.csv"
                        df = pd.DataFrame([{
                            "Data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Nome": nome,
                            "Bilhete de Identidade": bi,
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

                        # Criação do diagnóstico em PDF
                        try:
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Times", size=14)
                            pdf.cell(200, 10, "Relatório Médico",
                                     ln=True, align="C")
                            pdf.cell(200, 10, "Diagnóstico de Tumores Cerebrais",
                                     ln=True, align="C")
                            pdf.ln()

                            # Adicionar informações do paciente e diagnóstico
                            for k, v in df.iloc[-1].items():
                                # Codificar caracteres especiais
                                texto = f"{k}: {str(v)}"
                                texto_encoded = texto.encode(
                                    'latin-1', 'replace').decode('latin-1')
                                pdf.cell(200, 10, texto_encoded, ln=True)

                            # Adicionar probabilidades
                            pdf.ln()
                            pdf.cell(
                                200, 10, "Probabilidades por Classe:", ln=True)
                            for classe, prob in todas_probs.items():
                                texto = f"{classe}: {prob*100:.2f}%"
                                texto_encoded = texto.encode(
                                    'latin-1', 'replace').decode('latin-1')
                                pdf.cell(200, 10, texto_encoded, ln=True)

                            pdf.ln()
                            pdf.cell(
                                200, 10, "Por favor, consulte o seu Radiologista para se informar dos próximos passos.", ln=True)

                            pdf_path = f"Histórico de Pacientes/{nome} Diagnóstico.pdf"
                            pdf.output(pdf_path)

                            with open(pdf_path, "rb") as f:
                                col10, col11 = st.columns([1, 1])
                                with col10:
                                    st.download_button("Baixar PDF do Diagnóstico",
                                                       f, file_name=os.path.basename(pdf_path))

                                with col11:
                                    if st.button("Fechar Diagnostico"):
                                        st.rerun()

                        except Exception as e:
                            st.warning(
                                f"PDF gerado, mas pode conter problemas de encoding: {e}")
                    else:
                        st.error("Erro ao gerar diagnóstico.")
                else:
                    st.error("Erro no processamento da imagem.")

            except Exception as e:
                st.error(f"Erro durante o diagnóstico: {str(e)}")
                st.error(
                    "Verifique se o modelo foi treinado corretamente e se os arquivos necessários estão presentes.")

# ========== Histórico ========== #
# Inicializar o controle
if "show_historico" not in st.session_state:
    st.session_state.show_historico = False

# Botão para mostrar o histórico
if paciente_selecionado:
    st.sidebar.button(
        "Ocultar Histórico" if st.session_state.show_historico else "Mostrar Histórico",
        on_click=lambda: st.session_state.update(
            {"show_historico": not st.session_state.show_historico})
    )

if st.session_state.show_historico and paciente_selecionado:
    st.divider()
    hist_path = f"Histórico de Pacientes/{paciente_selecionado} Histórico.csv"
    if os.path.exists(hist_path):
        st.subheader(f"Histórico Médico de {paciente_selecionado}")
        historico_df = pd.read_csv(hist_path)
        st.dataframe(historico_df)

        st.divider()
        idx = st.selectbox(
            "Selecione o diagnóstico para baixar",
            options=historico_df.index,
            format_func=lambda x: historico_df.loc[x, "Data"]
        )
        selected = historico_df.loc[idx]

        # Criação do diagnóstico em PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Times", size=14)
            pdf.cell(200, 10, "Relatório Médico", ln=True, align="C")
            pdf.cell(200, 10, "Diagnóstico de Tumores Cerebrais",
                     ln=True, align="C")
            pdf.ln()
            for k, v in selected.items():
                texto = f"{k}: {str(v)}"
                texto_encoded = texto.encode(
                    'latin-1', 'replace').decode('latin-1')
                pdf.cell(200, 10, texto_encoded, ln=True)

            pdf_path = f"Histórico de Pacientes/{paciente_selecionado} Diagnóstico.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Baixar Diagnóstico Selecionado", f, file_name=os.path.basename(pdf_path))
        except Exception as e:
            st.warning(f"Erro na geração do PDF: {e}")

# ========== Sobre ========== #
st.sidebar.divider()
st.sidebar.text("© Eng. C. Ukuma, ISPI 2025")

# Variável de controle do modal
if 'form' not in st.session_state:
    st.session_state.form = False

if st.sidebar.button("Detalhes do Projecto"):
    st.session_state.form = True

# Construção do modal


@st.dialog("I'm an Engineer, finally!")
def modal():
    st.subheader("Descrição do Projecto")
    st.markdown("Este projecto apresenta o desenvolvimento de uma aplicação inteligente para **Diagnóstico de Tumores Cerebrais** com base em **imagens de Ressonância Magnética (MRI)**. A solução utiliza técnicas de **Visão Computacional** e **Aprendizado Profundo**, com o modelo **InceptionV3**, para classificar imagens em quatro categorias principais: **glioma, meningioma, pituitário e nenhum tumor**. Este trabalho é parte do projecto de conclusão de curso em **Engenharia Informática**, desenvolvido no Instituto Superior Politécnico Independente (ISPI), apresentado por **Carlos Denilson Pascoal Ukuma**, orientado pelo Professor **Abel Zacarias**. A proposta visa demonstrar como as ferramentas de inteligência artificial podem ser aplicadas de forma prática e ética no contexto médico, oferecendo suporte ao diagnóstico por imagem com maior rapidez e eficiência.")
    st.divider()
    st.markdown("Este trabalho é dedicado, com todo o meu carinho, à minha **família**, que sempre foi meu porto seguro, me apoiando nos momentos mais difíceis, me incentivando nos momentos de dúvida, e celebrando comigo cada pequena vitória. Aos **meus colegas de turma**, que partilharam comigo noites de estudo, desafios e conquistas — somos mais que colegas, somos irmãos de jornada. E em especial, com profunda saudade e emoção, dedico este projecto à memória de **Celmira Samuel**, colega, amiga. Sua presença, mesmo que agora ausente fisicamente, continua viva em nossos corações. Que este trabalho carregue um pouco da luz e do exemplo que você deixou em nós.")

    col12, col13, col14 = st.columns([6, 5, 4])
    with col13:
        if st.button("Fechar"):
            st.session_state.form = False
            st.rerun()


# Visualização do modal
if st.session_state.form:
    modal()
