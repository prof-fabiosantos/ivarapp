import streamlit as st
import cv2
import tempfile
import os
import time
import base64
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

# Configuração da API Gemini - substitua pela sua chave válida
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

FRAME_FOLDER = "full_frames"
os.makedirs(FRAME_FOLDER, exist_ok=True)
OUTPUT_FILE = "observations.txt"

if "log_text" not in st.session_state:
    st.session_state["log_text"] = None

if "acesso_verificado" not in st.session_state:
    st.session_state["acesso_verificado"] = False

if "video_processado" not in st.session_state:
    st.session_state["video_processado"] = False

if "link_pagamento_clcado" not in st.session_state:
    st.session_state["link_pagamento_clcado"] = False


def parse_irregularity(observation):
    detected_irregularities = []
    lines = observation.split("\n")

    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4:
            tipo = parts[1].strip()
            observado = parts[2].strip().lower()
            descricao = parts[3].strip() or "Não especificado"

            if observado == "yes":
                detected_irregularities.append((tipo, descricao))

    return detected_irregularities

def color_irregularity(value):
    if value.lower() != "não especificado":
        return "background-color: #8b0000; color: #ffffff;"
    return "background-color: transparent;"

def style_irregularities_table(df):
    return df.style.set_table_styles([
        {"selector": "th.col_heading.level0.col1", "props": [("min-width", "400px")]},
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "left")]}
    ]).applymap(color_irregularity, subset=["Descrição"])

def reset_state():
    st.session_state.clear()

def processar_video(uploaded_file):
    temp_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_path)
    st.write("Processando vídeo...")

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total_frames / fps
    SEND_INTERVAL = 1.5 if duration <= 30 else (2 if duration <= 180 else 4)

    st.write(f"⏱️ Duração do vídeo: {duration:.1f} segundos - Capturando frame a cada {SEND_INTERVAL} segundo(s)")

    frame_paths = []
    irregular_frames = []
    irregularities_table = []

    progress_bar = st.progress(0)

    def analyze_with_gemini(image_path, timestamp):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": (
                        "Analyze the image of the football play and respond in a structured format, according to the **Official FIFA Laws of the Game**.\n"
                        "If any irregularity is detected (**'Yes'**), provide details about the nature of the infraction.\n"
                        "No additional details are required.\n"
                        "\n"
                        "| Type of Irregularity | Observed? (Yes/No) | Description of Irregularity (If Yes) |\n"
                        "|-----------------------------|--------------------|--------------------------------------|\n"
                        "| Foul (pushing, reckless charge, excessive force, illegal obstruction, etc.) | No | - |\n"
                        "| Handball (deliberate hand or arm contact with the ball) | No | - |\n"
                        "| Offside (receiving the ball in an offside position followed by active play) | No | - |\n"
                        "| Penalty (foul or handball inside the defensive penalty area) | No | - |\n"
                        "| Unsporting behavior (simulation, violent conduct, dissent, excessive protests) | No | - |\n"
                        "| Irregular goal (goal scored after an infraction or with illegal hand/arm use) | No | - |\n"
                        "| Leaving the field without referee authorization | No | - |\n"
                        "| Goalkeeper infraction (holding the ball for more than 6 seconds) | No | - |\n"
                        "| Set-piece irregularity (improper throw-in, goal kick or corner kick violation) | No | - |\n"
                        "\n"
                        "For each detected irregularity, briefly describe what happened and specify which FIFA Law was violated."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )

            response = gemini_model.invoke([message])
            observation = response.content.strip()

            if "Yes" in observation:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
                    file.write(f"{timestamp}\n{observation}\n\n")
                return observation
            return None

        except Exception as e:
            return f"❌ Erro na análise: {e}"

    frame_count = 0
    last_sent_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        if time.time() - last_sent_time >= SEND_INTERVAL:
            last_sent_time = time.time()
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            frame_path = os.path.join(FRAME_FOLDER, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)

            observation = analyze_with_gemini(frame_path, timestamp)
            frame_paths.append(frame_path)

            if observation:
                detected_irregularities = parse_irregularity(observation)
                if detected_irregularities:
                    irregular_frames.append((frame_path, detected_irregularities))
                    irregularities_table.extend(detected_irregularities)

        progress_bar.progress(frame_count / total_frames)

    cap.release()
    st.success("Processamento concluído! ✅")

    if irregular_frames:       
        st.write("### 📊 Resumo das irregularidades detectadas")
        cols = st.columns(3)

        for idx, (frame_path, detected_irregularities) in enumerate(irregular_frames):
            with cols[idx % 3]:
                st.image(frame_path, caption=f"Irregularidade {idx+1}", use_container_width=True)
                for tipo, descricao in detected_irregularities:
                    st.write(f"🔸 **{tipo}**")
                    st.write(f"👉 {descricao}")

        st.write("### Tabela Resumida de Irregularidades")
        df = pd.DataFrame(irregularities_table, columns=["Tipo", "Descrição"])
        st.dataframe(style_irregularities_table(df))

        log_text = "\n".join([f"{tipo}: {descricao}" for tipo, descricao in irregularities_table])
        st.session_state["log_text"] = log_text
    else:
        st.write("✅ Nenhuma irregularidade detectada.")

    os.remove(temp_path)

st.title("🤖 iVAR ⚽🤾‍♂️")

st.write("""
Com o iVAR, você resolve de vez aquelas jogadas polêmicas do futebol. Envie o vídeo do lance e nossa IA analisa cada frame usando as Regras Oficiais da FIFA. Tecnologia de ponta para trazer justiça ao seu jogo!
""")

st.markdown("""
### 📋 Como funciona
1. Informe seu e-mail para gerar o link de pagamento;
2. Realize o pagamento de R$ 5,00;
3. Envie o vídeo (máximo 10 segundos, formato MP4);
4. Aguarde enquanto a IA analisa e te envia o resultado.
""")

st.write("Veja abaixo como a IA analisa cada lance:")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.image("frames.png", caption="Exemplo da análise automática da IA", width=370)

email = st.text_input("Digite seu e-mail para pagar e acessar o analisador:")

if not st.session_state["link_pagamento_clcado"]:
    if st.button("Gerar link de pagamento 💳💵"):
        with st.spinner("Gerando link de pagamento, aguarde..."):
            response = requests.post(
                "https://apistripe.onrender.com/create-checkout-session/",
                json={"email": email, "amount": 1000}
            )
            if response.status_code == 200:
                checkout_url = response.json()["checkout_url"]
                st.markdown(f"[Clique aqui para pagar]({checkout_url})")
                st.session_state["link_pagamento_clcado"] = True
            else:
                st.error("Erro ao iniciar pagamento!")


if not st.session_state["acesso_verificado"]:
    if st.button("📤Enviar o Vídeo 🎞️"):
        try:
            response = requests.post(f"https://apistripe.onrender.com/use-access/{email}")
            print(response)
            if response.status_code != 200:
                st.error(f"O pagamento não foi feito ⛔")
                #st.error(f"Erro da API: {response.status_code} - {response.text}")
                st.session_state["acesso_liberado"] = False
            else:
                data = response.json()
                if data.get("access", False):
                    st.session_state["acesso_liberado"] = data.get("access", False)
                    if st.session_state["acesso_liberado"]:
                        st.session_state["acesso_verificado"] = True
                else:
                    st.error(f"O pagamento não foi feito ⛔")
        except requests.RequestException as e:
            st.error(f"O pagamento não foi feito ⛔")
            #st.error(f"Erro de conexão com a API: {e}")
            st.session_state["acesso_liberado"] = False
else:
    st.success("✅ Acesso já verificado!")


if st.session_state.get("acesso_liberado") and not st.session_state["video_processado"]:
    uploaded_file = st.file_uploader("Faça o envio de um vídeo mp4 do lance duvidoso", type=["mp4"])
    if uploaded_file and st.button("🕴️ Analisar o vídeo"):
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["processar_video_ativado"] = True

if st.session_state.get("processar_video_ativado"):
    processar_video(st.session_state["uploaded_file"])
    st.session_state["processar_video_ativado"] = False

if st.session_state.get("log_text"):
    st.session_state["acesso_verificado"] = True
    st.download_button("📥 Baixar Log de Observações", st.session_state["log_text"], "observations.txt")
    st.session_state["uploaded_file"] = None  # Reset explícito

if st.button("🚪 Sair"):
    st.session_state.clear()
    st.session_state["uploaded_file"] = None  # Reset explícito
    st.rerun()
