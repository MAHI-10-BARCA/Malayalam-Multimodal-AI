import streamlit as st
import tempfile

from streamlit_mic_recorder import mic_recorder

from src.text.text_model import load_labeled_data, apply_preprocessing, train_model, predict_text
from src.image.image_model import predict_from_image
from src.speech.speech_model import speech_to_text

# 🔥 NEW IMPORTS (only addition)
from src.visuals.news_renderer import generate_news_image
from src.visuals.news_fetcher import get_random_news


# ================================
# 🎨 PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Malayalam Multimodal AI",
    page_icon="🧠",
    layout="wide"
)

# ================================
# 🎨 CUSTOM CSS
# ================================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ================================
# 🧠 HEADER
# ================================
st.markdown('<div class="big-title">🧠 Malayalam Multimodal AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Text • Image • Speech Understanding 🚀</div>', unsafe_allow_html=True)

st.markdown("---")

# ================================
# 🔄 LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    df = load_labeled_data()
    df = apply_preprocessing(df)
    return train_model(df)

model, vectorizer = load_model()

# ================================
# 🎯 SIDEBAR
# ================================
st.sidebar.title("🚀 Choose Mode")

option = st.sidebar.radio(
    "Select Input Type:",
    ["📝 Text", "🖼️ Image", "🎤 Audio"]
)

# ================================
# 📝 TEXT MODE
# ================================
if option == "📝 Text":

    st.markdown("### ✍️ Enter Malayalam Text")

    text_input = st.text_area("Type here...")

    if st.button("🚀 Analyze Text"):

        if text_input:
            result, confidence = predict_text(text_input, model, vectorizer)

            st.markdown("### 🎯 Result")
            st.success(f"📌 Category: **{result.upper()}**")
            st.info(f"📊 Confidence: {confidence:.2f}")

        else:
            st.warning("⚠️ Please enter some text")


# ================================
# 🖼️ IMAGE MODE (NEWS GENERATOR 🔥)
# ================================
elif option == "🖼️ Image":

    st.markdown("### 📰 Generate Malayalam News")

    # 🎯 Category selector
    category = st.selectbox(
        "Select News Category",
        ["sports", "politics", "business", "entertainment", "world"]
    )

    # 🚀 Generate button
    if st.button("📰 Generate News"):

        # 🔥 Get random news text
        news_text = get_random_news(category)

        # 🎨 Generate image
        img_path = generate_news_image(news_text, category)

        if img_path:
            st.image(img_path, caption=f"{category.upper()} NEWS", use_container_width=True)

            # 📄 Show text option
            with st.expander("📄 Show Text"):
                st.write(news_text)
        else:
            st.error("❌ Failed to generate news image")


# ================================
# 🎤 AUDIO MODE
# ================================
elif option == "🎤 Audio":

    st.markdown("### 🎙️ Speak in Malayalam")

    import tempfile
    from pydub import AudioSegment

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True
    )

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(audio["bytes"])
            webm_path = f.name

        wav_path = webm_path.replace(".webm", ".wav")

        sound = AudioSegment.from_file(webm_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_path, format="wav")

        st.session_state.audio_path = wav_path

        st.success("✅ Audio Recorded")

    if "audio_path" in st.session_state:
        st.audio(st.session_state.audio_path)

        if st.button("🚀 Analyze Speech"):

            st.write("🔄 Processing...")

            text = speech_to_text(st.session_state.audio_path)

            st.markdown("### 🧾 Recognized Text")
            st.text_area("", text, height=120)

            if text.strip():
                result, confidence = predict_text(text, model, vectorizer)

                st.markdown("### 🎯 Result")
                st.success(f"📌 Category: **{result.upper()}**")
                st.info(f"📊 Confidence: {confidence:.2f}")
            else:
                st.warning("❌ No speech detected")

# ================================
# 🧾 FOOTER
# ================================
st.markdown("---")
st.markdown("💡 Built with ❤️ using NLP • OCR • Speech AI")