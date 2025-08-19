# Diagnóstico de Tumores Cerebrais com Deep Learning e Streamlit

Este projeto é uma aplicação web interativa desenvolvida com **Streamlit**, que utiliza um modelo de **aprendizado profundo baseado na arquitetura InceptionV3** para classificar imagens de ressonância magnética (MRI) em quatro categorias:

- Glioma
- Meningioma
- Pituitário
- Nenhum Tumor

O sistema permite o **upload de imagens e informações do paciente**, gera diagnóstico com **confiança percentual**, salva histórico por paciente e permite a **emissão de relatórios médicos em PDF**.

---

## 🧠 Modelo de Machine Learning

O modelo foi treinado com a arquitetura **InceptionV3** pré-treinada no ImageNet, utilizando **transfer learning**. Camadas densas foram adicionadas ao topo da rede para adaptá-la ao problema de classificação de tumores cerebrais. O modelo foi treinado com imagens redimensionadas para 224x224 pixels e salva no formato `.h5`.

**Dataset utilizado:** [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


## 📦 Requisitos

Crie um ambiente virtual e instale as dependências com:

```bash
pip install -r requirements.txt
```

### Arquivo `requirements.txt`:
```txt
streamlit==1.35.0
tensorflow==2.15.0
numpy==1.24.3
pandas==2.2.2
opencv-python==4.9.0.80
Pillow==10.2.0
fpdf==1.7.2
openai==1.30.1
```

---

## 🚀 Como Executar

1. Certifique-se de que o arquivo `brain_tumor_inceptionv3.h5` esteja na raiz do projeto (ou use o upload manual na sidebar).

2. Execute a aplicação com:

```bash
streamlit run app.py
```

3. Acesse no navegador: [http://localhost:8501](http://localhost:8501)


---

## 🖼️ Funcionalidades

- Upload de modelo `.h5` (ou uso do modelo padrão)
- Cadastro e consulta de pacientes
- Upload de imagem de MRI (JPG/PNG)
- Validação do Bilhete de Identidade (B.I.)
- Gera diagnóstico com confiança percentual
- Salva histórico médico por paciente (CSV + PDF)
- Permite edição e exclusão de pacientes
- Exporta relatórios PDF sob demanda


---

## 📁 Estrutura de Diretórios
```
.
├── app.py                     # Código principal do Streamlit
├── brain_tumor_inceptionv3.h5  # Modelo treinado (padrão)
├── requirements.txt          # Dependências
├── README.md                 # Documentação
├── Imagens/
│   └── brain.png            # Icone usado na sidebar
└── Histórico de Pacientes/
    ├── NomePaciente Histórico.csv
    └── NomePaciente Diagnóstico.pdf
```

---

## 🧾 Observação sobre Formato do B.I.
O sistema valida o bilhete de identidade usando o seguinte formato:
```
123456789AB123
(9 dígitos + 2 letras + 3 dígitos)
```

---

## 🎓 Sobre o Projeto
Este projeto faz parte do **Trabalho de Conclusão de Curso (TCC)** do curso de **Engenharia Informática**, apresentado por **Carlos Denilson Pascoal Ukuma** no **Instituto Superior Politécnico Independente (ISPI)**, orientado pelo professor **Abel Zacarias**.

A proposta visa demonstrar a aplicação prática e ética de ferramentas de IA no apoio ao diagnóstico médico por imagem.

---

## 📬 Contato
**Autor:** Carlos Denilson Pascoal Ukuma 
**Email:** carlosukuma@icloud.com
**Instituição:** ISPI - Instituto Superior Politécnico Independente  
**Ano:** 2025