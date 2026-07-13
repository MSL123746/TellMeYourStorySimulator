import os
import importlib
from huggingface_hub import HfApi, InferenceClient  # type: ignore

import streamlit as st

HF_MODEL = os.getenv("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")
FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def get_hf_token_with_source() -> tuple[str, str]:
    def _safe_secret_get(key: str) -> str:
        try:
            return st.secrets.get(key, "")
        except Exception:
            return ""

    token = _safe_secret_get("HF_TOKEN").strip()
    if token:
        return token, "st.secrets.HF_TOKEN"

    return "", "missing"


def get_hf_token() -> str:
    token, _ = get_hf_token_with_source()
    return token


@st.cache_resource
def get_inference_client(token: str) -> InferenceClient:
    return InferenceClient(api_key=token)


def _candidate_models() -> list[str]:
    models = [HF_MODEL] + FALLBACK_MODELS
    deduped = []
    for model in models:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def call_hf_inference(prompt: str, token: str) -> str:
    client = get_inference_client(token)
    last_error = None

    for model_name in _candidate_models():
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                stream=False,
            )

            if completion and completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()

            last_error = ValueError(f"Unexpected response format for model '{model_name}'.")
        except Exception as e:
            msg = str(e)
            last_error = e
            if "model_not_supported" in msg or "not supported by any provider" in msg:
                continue
            if "403" in msg or "Forbidden" in msg or "sufficient permissions" in msg:
                raise ValueError(
                    "Hugging Face token is valid, but it does not have permission to call Inference Providers. "
                    "Create or edit the token in Hugging Face and enable the permission for inference/provider calls, "
                    "then update .streamlit/secrets.toml and restart the app."
                )
            if "401" in msg or "Unauthorized" in msg or "Invalid username or password" in msg:
                raise ValueError(
                    "Hugging Face authentication failed (401). "
                    "Create a new HF access token and update .streamlit/secrets.toml with HF_TOKEN. "
                    "Then restart the app and try again."
                )
            raise

    raise ValueError(
        "No supported model was available for your enabled providers. "
        "Set HF_MODEL in secrets/env to a model available in your HF account/providers. "
        f"Tried: {', '.join(_candidate_models())}. Last error: {last_error}"
    )


def extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext == ".pdf":
        try:
            pdf_module_name = "PyPDF2"
            try:
                pdf_module = importlib.import_module(pdf_module_name)
            except ImportError:
                pdf_module_name = "pypdf"
                pdf_module = importlib.import_module(pdf_module_name)

            PdfReader = getattr(pdf_module, "PdfReader")
            pdf_reader = PdfReader(uploaded_file)
            return " ".join(page.extract_text() or "" for page in pdf_reader.pages).strip()
        except Exception as exc:
            st.error(f"Could not read PDF: {exc}")
            return ""

    if file_ext == ".docx":
        try:
            import docx

            doc = docx.Document(uploaded_file)
            return " ".join(para.text for para in doc.paragraphs).strip()
        except Exception:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore").strip()

    uploaded_file.seek(0)
    return uploaded_file.read().decode("utf-8", errors="ignore").strip()


def build_narrative_prompt(project_description: str, resume_text: str) -> str:
    return f"""
You are an expert interview coach.

Using the job description and resume content below, create a "Tell My Story" narrative for a general professional conversation.

Project Description:
{project_description}

Resume Content:
{resume_text}

Instructions:
1. Provide a speaking script for up to 2 minutes at a medium speaking pace.
2. Keep the total length between 210 and 260 words, and never exceed 260 words.
3. Keep it in first person, polished, confident, and natural for live delivery.
4. Use a professional-conversational tone: warm, clear, and human, but not overly casual.
5. Vary sentence length and avoid repetitive sentence openings.
6. Use plain spoken language and light contractions where natural (for example: "I've", "I've led", "I'm focused").
7. Avoid robotic or overly formal phrases like "I am writing to express", "therefore", "moreover", "in conclusion", "it is imperative", "leverage synergies", and "utilize".
8. Sound like a real person speaking naturally in conversation, not reading a formal essay.
9. Use my resume experience as the foundation and tightly align it to this specific job description.
10. Intertwine my experience with the role requirements so the response sounds tailored, strategic, and role-specific.
11. Highlight concrete impact, measurable outcomes, and transferable strengths that map directly to the job.
12. Use a clear career progression flow: where I started, how I grew, key transitions, and what led me to apply for this role now.
13. Prioritize keywords and responsibilities from the job description when phrasing the narrative.
14. Do not invent experience not present in the resume; if details are missing, stay high-confidence and realistic.
15. Clearly explain what motivated me to apply for this role and why this role is the right next step.
16. End with a concise closing statement that reinforces fit and enthusiasm for this role.
17. Avoid technical jargon, acronyms, and buzzwords; use plain, human language that any interviewer can follow.
18. If a technical term is unavoidable, explain it in simple everyday wording.
19. Do not use section headers, titles, labels, bullet points, or numbered lists.
20. Do not include conversational opening pleasantries (for example: "Hi", "Thanks for having me", "Great to meet you").
21. Return the response as a conversation-like personal story about me in paragraph form only (2-3 cohesive paragraphs).
22. Keep the voice natural and spoken, as if I am talking in a general professional conversation.
23. Start naturally with a spoken opener in this style: "Well, I've been ..." and then explain how my experience built over time.
24. Emphasize progression over time, with examples of what I learned and how that led me to apply for this role.
25. Do not include title-like starters such as "Tell me about yourself:", "Background:", "Who I am:", or similar label text.
26. Do not use words like "excited", "thrilled", "glad", or "motivated".
27. Output only markdown text (no JSON, no HTML).
""".strip()


def build_elevator_pitch_prompt(job_role_details: str) -> str:
    return f"""
You are an expert interview coach.

Create a simple 30-second elevator pitch based on the role details below.

Job Role Details:
{job_role_details}

Instructions:
1. Write in first person as if I am speaking to an executive who asked what I do.
2. Keep it professional-conversational, clear, and natural, not robotic.
3. Keep it short enough for about 30 seconds at a medium pace.
4. Keep the total length between 50 and 75 words, and never exceed 75 words.
5. Focus on: who I am, what I do day-to-day, and the value I create.
6. Use plain language and avoid technical jargon or buzzwords.
7. Follow this structure every time:
    - Sentence 1: My current role context and how I contribute with my team.
    - Sentence 2: The specific business value I drive and the kinds of outcomes I create.
    - Sentence 3: Why this work matters to the business in simple terms.
8. Include a balance of people-oriented strengths and execution strengths (for example: collaboration, creative problem-solving, and data-informed decisions).
9. Keep the tone similar to this style: confident, direct, and grounded in impact, without sounding scripted.
10. Use short, clear sentences that sound like spoken conversation.
11. Prefer wording like "My role is...", "I work on...", "I also help...", and "That has helped...".
12. Avoid formal phrasing like "This enables me" or "propel the organization's success".
13. Keep language simple and natural, like I am speaking in a quick hallway conversation.
14. Avoid generic closing lines such as "has prepared me well for this role" or "my background in ... has prepared me".
15. Do not use titles, section headers, bullets, or numbered lists.
16. Return a single cohesive paragraph only.
17. Speak directly as if I am talking to one executive in front of me.
18. Do not include lead-in phrases like "Here is my narrative", "Here is my elevator pitch", "Sure", or "Absolutely".
19. Start immediately with the pitch content itself.
20. Do not use greeting/opening words such as "Hey", "Hi", or "Hello".
21. Do not use the word "excited".
22. Do not use phrases like "strong fit" or "perfect fit".
23. Output only markdown text (no JSON, no HTML).
""".strip()


def cap_narrative_to_medium_two_minutes(text: str, max_words: int = 260) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    trimmed = " ".join(words[:max_words]).strip()

    # Trim to the end of the last sentence-like boundary when possible.
    last_boundary = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if last_boundary > int(len(trimmed) * 0.6):
        trimmed = trimmed[: last_boundary + 1]
    else:
        trimmed = trimmed.rstrip(" ,;:-") + "."

    return trimmed


def cleanup_narrative_format(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []

    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue

        # Remove markdown-style headers and common section labels.
        if line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith((
            "introduction:",
            "intro:",
            "background:",
            "closing:",
            "why i'm a fit:",
            "tell me about yourself:",
            "who i am:",
            "summary:",
        )):
            line = line.split(":", 1)[1].strip() if ":" in line else ""

        # Drop short heading-like lines ending with ':' to avoid titled output blocks.
        if line.endswith(":") and len(line.split()) <= 8:
            continue

        # Convert bullets/numbered list items into plain lines.
        if line.startswith(("- ", "* ", "• ")):
            line = line[2:].strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in ".)" and line[2] == " ":
            line = line[3:].strip()

        if line:
            cleaned_lines.append(line)

    # Rebuild as paragraph text.
    text_flat = " ".join(part for part in cleaned_lines if part).strip()
    return text_flat


def strip_leading_preface(text: str) -> str:
    lower_text = text.lower().strip()
    preface_starts = [
        "here is my narrative",
        "here's my narrative",
        "here is my elevator pitch",
        "here's my elevator pitch",
        "here is your elevator pitch",
        "here's your elevator pitch",
        "this is my elevator pitch",
        "certainly",
        "absolutely",
        "sure",
    ]

    for phrase in preface_starts:
        if lower_text.startswith(phrase):
            # Drop text up to first sentence boundary or colon, then continue with the pitch.
            for sep in [":", ".", "!", "?"]:
                idx = text.find(sep)
                if idx != -1 and idx < 140:
                    return text[idx + 1 :].strip()
            return ""

    return text


def soften_robotic_tone(text: str) -> str:
    replacements = {
        "I am ": "I'm ",
        "I have ": "I've ",
        "I would ": "I'd ",
        "do not": "don't",
        "cannot": "can't",
        "utilize": "use",
        "leverage": "use",
        "moreover": "also",
        "therefore": "so",
        "in conclusion": "overall",
        "I am confident": "I'm confident",
    }

    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)

    return updated


def enforce_elevator_word_rules(text: str) -> str:
    cleaned = text.strip()

    # Remove common greeting starts.
    for greeting in ["Hey", "Hi", "Hello", "hey", "hi", "hello"]:
        if cleaned.startswith(greeting + " "):
            cleaned = cleaned[len(greeting) + 1 :].strip(" ,.!?-")
            break

    # Replace disallowed enthusiasm wording with a neutral alternative.
    cleaned = cleaned.replace("excited", "motivated")
    cleaned = cleaned.replace("Excited", "Motivated")

    # Remove interview-rating phrasing that the user does not want.
    cleaned = cleaned.replace("strong fit", "good match")
    cleaned = cleaned.replace("Strong fit", "Good match")
    cleaned = cleaned.replace("perfect fit", "good match")
    cleaned = cleaned.replace("Perfect fit", "Good match")

    # Remove overly generic interview filler phrasing.
    generic_phrases = [
        "has prepared me well for this role",
        "have prepared me well for this role",
        "my background in understanding customer behavior and making data-informed decisions",
    ]
    for phrase in generic_phrases:
        cleaned = cleaned.replace(phrase, "my hands-on work and results")
        cleaned = cleaned.replace(phrase.capitalize(), "My hands-on work and results")

    return cleaned


def enforce_story_word_rules(text: str) -> str:
    cleaned = text.strip()

    # Remove disallowed words without replacing with other hype wording.
    disallowed = ["excited", "Excited", "thrilled", "Thrilled", "glad", "Glad", "motivated", "Motivated"]
    for word in disallowed:
        cleaned = cleaned.replace(word, "")

    # Cleanup extra spaces that can appear after removals.
    cleaned = " ".join(cleaned.split())
    return cleaned


def simplify_jargon(text: str) -> str:
    replacements = {
        "cross-functional": "across teams",
        "stakeholders": "the people involved",
        "end-to-end": "from start to finish",
        "strategic": "well-planned",
        "optimized": "improved",
        "optimization": "improvement",
        "synergy": "teamwork",
        "KPI": "key result",
        "KPIs": "key results",
        "scalable": "able to grow",
        "bandwidth": "time and capacity",
        "roadmap": "plan",
    }

    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
        updated = updated.replace(old.title(), new.capitalize())

    return updated


def check_hf_token_status(token: str) -> tuple[bool, str]:
    if not token.strip():
        return False, "HF token is missing. Add HF_TOKEN to .streamlit/secrets.toml."

    try:
        api = HfApi(token=token)
        whoami = api.whoami()
        username = str(whoami.get("name", "unknown"))
        return True, f"Hugging Face token is valid for user: {username}"
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg or "Invalid username or password" in msg:
            return (
                False,
                "HF token is invalid (401 Unauthorized). Create a new token and update .streamlit/secrets.toml.",
            )
        return False, f"Could not validate HF token: {msg}"


# App layout and styling
st.set_page_config(page_title="Tell My Story Narrative Builder", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 10%, #f0f9ff 0%, #eef2ff 35%, #f8fafc 100%);
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    h1, h2, h3 {
        color: #0f172a !important;
    }
    .stButton > button {
        font-weight: 700;
        border: 1px solid #1d4ed8 !important;
    }
    /* Force very large top tab labels */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[role="tab"] {
        min-height: 56px !important;
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[role="tab"] p,
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[role="tab"] span,
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[role="tab"] div {
        font-size: 2rem !important;
        line-height: 1.2 !important;
        font-weight: 800 !important;
        letter-spacing: 0.2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_story, tab_elevator = st.tabs(["Tell My Story", "Create Elevator Pitch"])

# Main narrative workflow
with tab_story:
    st.title("Tell My Story Interview Simulator")
    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        # Project description input
        st.markdown("### Enter Job Description")
        project_description = st.text_area(
            "Enter Job Description",
            key="project_description",
            height=220,
            placeholder="Paste or type the job description here...",
            label_visibility="collapsed",
        )

        if project_description.strip():
            # Resume upload section
            st.markdown("### Upload Resume")
            uploaded_resume = st.file_uploader(
                "Upload your resume (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                key="resume_upload",
            )

            if uploaded_resume is not None:
                st.success(f"Uploaded: {uploaded_resume.name}")
                if st.button("Remove uploaded resume", key="remove_resume"):
                    st.session_state.pop("resume_upload", None)
                    st.session_state.pop("resume_text", None)
                    st.rerun()
        else:
            st.info("Add your project description to unlock resume upload.")
            uploaded_resume = None

    with right_col:
        # Narrative generation and output
        st.markdown("### Narrative Output")

        resume_uploaded = uploaded_resume is not None
        if resume_uploaded:
            generate_narrative = st.button(
                "Generate my Narrative",
                use_container_width=True,
                key="generate_narrative_btn",
            )
        else:
            generate_narrative = False
            st.info("Upload your resume to unlock Generate my Narrative.")

        if generate_narrative:
            if not project_description.strip():
                st.error("Please enter a project description first.")
            elif uploaded_resume is None:
                st.error("Please upload your resume.")
            else:
                resume_text = extract_resume_text(uploaded_resume)
                st.session_state["resume_text"] = resume_text

                if not resume_text.strip():
                    st.error("The resume appears empty or unreadable. Try a different file.")
                else:
                    with st.spinner("Generating your interview narrative..."):
                        try:
                            hf_token = get_hf_token()
                            if not hf_token:
                                st.error(
                                    "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml."
                                )
                                st.stop()

                            prompt = build_narrative_prompt(project_description.strip(), resume_text)
                            narrative_md = call_hf_inference(prompt, hf_token)
                            narrative_md = cleanup_narrative_format(narrative_md)
                            narrative_md = soften_robotic_tone(narrative_md)
                            narrative_md = simplify_jargon(narrative_md)
                            narrative_md = enforce_story_word_rules(narrative_md)
                            narrative_md = cap_narrative_to_medium_two_minutes(narrative_md, max_words=260)
                            st.session_state["narrative_md"] = narrative_md
                        except Exception as exc:
                            st.error(f"Hugging Face API error: {exc}")

        if "narrative_md" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["narrative_md"])

with tab_elevator:
    st.header("Create Elevator Pitch")
    elevator_left_col, elevator_right_col = st.columns([1.15, 1], gap="large")

    with elevator_left_col:
        st.markdown("### Enter Your Job Role with some Detail")
        elevator_role_details = st.text_area(
            "Enter Your Job Role with some Detail",
            key="elevator_role_details",
            height=220,
            placeholder="Describe the role and what the company needs...",
            label_visibility="collapsed",
        )

    with elevator_right_col:
        st.markdown("### 30-Second Elevator Pitch")
        submit_elevator = st.button("Submit", use_container_width=True, key="submit_elevator_btn")

        if submit_elevator:
            if not elevator_role_details.strip():
                st.error("Please enter your job role details first.")
            else:
                with st.spinner("Generating your elevator pitch..."):
                    try:
                        hf_token = get_hf_token()
                        if not hf_token:
                            st.error("Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml.")
                            st.stop()

                        elevator_prompt = build_elevator_pitch_prompt(elevator_role_details.strip())
                        elevator_md = call_hf_inference(elevator_prompt, hf_token)
                        elevator_md = cleanup_narrative_format(elevator_md)
                        elevator_md = strip_leading_preface(elevator_md)
                        elevator_md = soften_robotic_tone(elevator_md)
                        elevator_md = simplify_jargon(elevator_md)
                        elevator_md = enforce_elevator_word_rules(elevator_md)
                        elevator_md = cap_narrative_to_medium_two_minutes(elevator_md, max_words=85)
                        st.session_state["elevator_pitch_md"] = elevator_md
                    except Exception as exc:
                        st.error(f"Hugging Face API error: {exc}")

        if "elevator_pitch_md" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["elevator_pitch_md"])
