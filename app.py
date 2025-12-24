import io
from typing import Optional

import streamlit as st
from PIL import Image
import google.generativeai as genai

# ------------- Page Config -------------
st.set_page_config(
    page_title="Ollam LLaVA Image Storyteller",
    page_icon="📖",
    layout="wide",
)

# ------------- Basic Styling -------------
# Soft gradient background and centered card-style main panel
page_bg = """
<style>
body {
    background: radial-gradient(circle at top left, #6b8bff 0, #8a5bff 35%, #5a3cff 70%, #2d175f 100%);
}
#root > div:nth-child(1) > div.withScreencast > div {
    padding-top: 1.5rem;
}
.main-card {
    background-color: #f9f7ff;
    border-radius: 24px;
    padding: 2.5rem 3rem;
    max-width: 1100px;
    margin: 0 auto 3rem auto;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
}
.section-card {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    border: 1px solid #e5e0ff;
}
.upload-box {
    border: 2px dashed #c3b7ff !important;
    border-radius: 18px !important;
    padding: 1.75rem 1.5rem !important;
    background: #f5f3ff;
}
.stButton>button {
    width: 100%;
    border-radius: 999px;
    border: none;
    background: linear-gradient(90deg, #7c5cff, #ff6fd8);
    color: white;
    height: 3rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton>button:disabled {
    background: #d2ccff;
    color: #7b739c;
}
.lang-pill {
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 600;
    margin-right: 0.4rem;
    border: 1px solid rgba(148, 163, 184, 0.7);
}
.lang-pill-active {
    background: #4f46e5;
    color: #ffffff;
    border-color: transparent;
}
.app-footer {
    font-size: 0.75rem;
    text-align: center;
    color: #d0cff5;
    margin-top: 1rem;
}
.app-footer a {
    color: #f5f3ff;
    text-decoration: underline;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------- Title Block -------------
st.markdown(
    """
<div class="main-card">
  <div style="text-align:center; margin-bottom: 1.75rem;">
    <h1 style="margin-bottom:0.3rem; font-size:2.3rem;">
      Ollam LLaVA Image Storyteller
    </h1>
    <p style="margin:0; font-size:0.95rem; color:#64748b;">
      Transform images into captivating multilingual stories with AI
    </p>
    <p style="margin-top:0.4rem; font-size:0.9rem; color:#4b5563; font-weight:600;">
      Developed by Atinkut Molla at UCAS, 2025
    </p>
  </div>
""",
    unsafe_allow_html=True,
)

# ------------- Sidebar: API configuration (same deployment) -------------
with st.sidebar:
    st.header("Story Engine Settings")
    st.caption(
        "This demo uses the same Gemini deployment configuration. "
        "If the shared key is unavailable, you may provide your own in "
        "`.streamlit/secrets.toml` as `GEMINI_API_KEY`."
    )

# ------------- Gemini Setup -------------
# Assumes you keep the same deployment style you already use:
# The API key must come from Streamlit secrets or environment.
api_key: Optional[str] = st.secrets.get("GEMINI_API_KEY", None)

if api_key:
    genai.configure(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_story_model():
    # Use the same model you used before; typically "gemini-1.5-flash" or similar.
    return genai.GenerativeModel("gemini-1.5-flash")


def generate_story_from_image(
    image_bytes: bytes,
    language: str,
) -> str:
    """Call Gemini to generate a story given an image and target language."""
    model = get_story_model()

    prompt = (
        "You are an imaginative storyteller. "
        "Look at the image and write a short, vivid narrative (about 3–5 paragraphs). "
        f"Write the story entirely in {language}. "
        "Do not describe the task, only tell the story."
    )

    img = Image.open(io.BytesIO(image_bytes))
    response = model.generate_content(
        [prompt, img],
        request_options={"timeout": 120},
    )
    return response.text.strip()


# ------------- Layout: Left / Right panels -------------
left_col, right_col = st.columns(2, gap="large")

# --- Left: Story Image Upload ---
with left_col:
    st.markdown(
        """
<div class="section-card">
  <h3 style="margin-top:0; margin-bottom:0.75rem;">Story Image</h3>
  <p style="font-size:0.85rem; color:#64748b; margin-bottom:0.75rem;">
    Upload an image to inspire your story. Supports JPG, PNG, and WebP up to 10 MB.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        uploaded_file = st.file_uploader(
            "Upload an Image for Storytelling",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    if uploaded_file is not None:
        st.markdown("#### Preview")
        st.image(uploaded_file, use_column_width=True, caption="Story image")

# --- Right: Generated Story with language tabs ---
with right_col:
    st.markdown(
        """
<div class="section-card">
  <h3 style="margin-top:0; margin-bottom:0.75rem;">Generated Story</h3>
  <p style="font-size:0.85rem; color:#64748b; margin-bottom:0.5rem;">
    Choose a language and let the storyteller craft your narrative.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Language selection styled as pills, matching top mockup
    lang_tabs = ["English", "Amharic", "Chinese"]
    lang_map = {
        "English": "English",
        "Amharic": "Amharic",
        "Chinese": "Chinese",
    }

    selected_lang = st.radio(
        "Language",
        lang_tabs,
        horizontal=True,
        label_visibility="collapsed",
    )

    # Placeholder for the story text
    story_placeholder = st.empty()

    # Generate button
    generate_clicked = st.button(
        "Generate Story",
        type="primary",
        disabled=uploaded_file is None or not api_key,
    )

    if not api_key:
        st.warning(
            "No `GEMINI_API_KEY` found in Streamlit secrets. "
            "Add it before generating stories."
        )

    if generate_clicked and uploaded_file is not None and api_key:
        with st.spinner("Weaving your story..."):
            image_bytes = uploaded_file.getvalue()
            story_text = generate_story_from_image(
                image_bytes=image_bytes,
                language=lang_map[selected_lang],
            )
        story_placeholder.markdown(
            f"##### Your Story in {selected_lang}\n\n{story_text}"
        )
    else:
        story_placeholder.markdown(
            """
<div style="border-radius:18px; border:1px dashed #e5e7eb; padding:1.4rem;
            background:#f9fafb; text-align:center; color:#94a3b8;
            font-size:0.9rem;">
  Your story awaits. Upload an image and select a language, then click
  <strong>Generate Story</strong> to begin.
</div>
""",
            unsafe_allow_html=True,
        )

# ------------- Footer -------------
st.markdown(
    """
<div class="app-footer">
  Ollam LLaVA Storyteller • Developed by Atinkut Molla at UCAS, 2025<br/>
  Note: First story generation may take a little longer as the AI model loads.
</div>
</div>
""",
    unsafe_allow_html=True,
)
