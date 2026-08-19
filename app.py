from __future__ import annotations
import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

from workflow import build_graph

app = build_graph()
config = {"configurable": {"thread_id": "user_session_1"}}
logs: List[str] = []

def log(msg: str):
    logs.append(msg)

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates", step_payload) during streaming, and ("final", state) at the end.
    """
    current_state: Dict[str, Any] = dict(inputs)
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
            if isinstance(step, dict):
                for val in step.values():
                    if isinstance(val, dict):
                        current_state.update(val)
        yield ("final", current_state)
        return
    except Exception as e:
        print(f"Streaming failed: {e}, falling back to invoke...")

    out = graph_app.invoke(inputs,config=config)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    base_dir = Path(__file__).resolve().parent
    return (base_dir / src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last: m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), width="stretch")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), width="stretch")
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1

def list_past_blogs() -> List[Path]:
    """
    Returns .md files in output/markdown/ folder and root directory, newest first.
    """
    files: List[Path] = []
    blogs_dir = Path("output/markdown")
    if blogs_dir.exists() and blogs_dir.is_dir():
        files.extend([p for p in blogs_dir.glob("*.md") if p.is_file()])

    cwd = Path(".")
    for p in cwd.glob("*.md"):
        if p.is_file() and p not in files:
            files.append(p)

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "submitted_topic" not in st.session_state:
    st.session_state["submitted_topic"] = None
if "generation_done" not in st.session_state:
    st.session_state["generation_done"] = False

# Sidebar theme toggle & settings first so they affect layout and styling immediately
with st.sidebar:
    st.header("Settings")
    dark_mode = st.toggle("🌙 Dark mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = dark_mode
    as_of = st.date_input("As-of date", value=date.today())

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in output/markdown/ or current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            original_label = title
            label = original_label
            counter = 1
            while label in file_by_label:
                label = f"{original_label} ({counter})"
                counter += 1

            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox(
            "Select a past blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["submitted_topic"] = extract_title_from_md(md_text, selected_md_file.stem)
                st.session_state["generation_done"] = True

# Theme setup (dynamic CSS variables)
if dark_mode:
    bg_color = "#1e1e1e"
    text_color = "#e5e5e5"
    accent_color = "#e07a5f"
    sidebar_bg = "#262626"
    card_bg = "#2d2d2d"
    border_color = "#3a3a3a"
    input_bg = "#2d2d2d"
    button_hover = "#f08b70"
    tab_active = "#e07a5f"
    tab_inactive = "#a0a0a0"
else:
    bg_color = "#faf9f7"
    text_color = "#1a1a1a"
    accent_color = "#cc5a37"
    sidebar_bg = "#f4f3f0"
    card_bg = "#ffffff"
    border_color = "#e6e4e0"
    input_bg = "#ffffff"
    button_hover = "#b84c2a"
    tab_active = "#cc5a37"
    tab_inactive = "#707070"

st.markdown(
    f"""
    <style>
    :root {{
        --bg-color: {bg_color};
        --text-color: {text_color};
        --accent-color: {accent_color};
        --sidebar-bg: {sidebar_bg};
        --card-bg: {card_bg};
        --border-color: {border_color};
        --input-bg: {input_bg};
        --button-hover: {button_hover};
        --tab-active: {tab_active};
        --tab-inactive: {tab_inactive};
    }}

    /* Global backgrounds */
    html, body, .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"],
    .main,
    [data-testid="stMain"],
    [data-testid="stAppViewBlockContainer"] {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}

    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    /* Typography & General Colors — exclude Material Symbols icon spans */
    html, body, .stApp, 
    h1, h2, h3, h4, h5, h6, 
    p, label, li, a, 
    [data-testid="stMarkdownContainer"] p,
    .stMarkdown {{
        color: var(--text-color) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}

    /* Apply font-family to div/span only when NOT a Material Symbols icon */
    div:not([data-testid*="Icon"]):not([class*="e15ve43o"]),
    span:not([data-testid*="Icon"]):not([class*="material-symbols"]):not([style*="Material Symbols"]) {{
        color: var(--text-color) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}

    /* Explicitly restore icon font so sidebar collapse arrow renders correctly */
    span[data-testid*="Icon"],
    [class*="material-symbols"],
    button span[style*="font-family"],
    .stIconMaterial, [class*="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded' !important;
        font-weight: 400 !important;
    }}

    button[data-baseweb="tab"] {{
        color: var(--tab-inactive) !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 16px !important;
        font-weight: 500 !important;
        transition: color 0.2s ease, border-color 0.2s ease !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: var(--accent-color) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--tab-active) !important;
        border-bottom: 2px solid var(--tab-active) !important;
    }}

    /* Inputs, Textareas styling */
    textarea, input {{
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}

    /* Chat input outer and inner redesign */
    div[data-testid="stChatInput"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        padding: 4px 12px !important;
    }}
    div[data-testid="stChatInput"] > div {{
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: var(--text-color) !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        font-size: 1rem !important;
        padding: 8px 12px !important;
    }}
    div[data-testid="stChatInput"] textarea:focus {{
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent-color) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08), 0 0 0 2px rgba(204, 90, 55, 0.2) !important;
    }}
    /* Bottom chat bar — remove the full-width background strip.
       Target: stBottom (the sticky wrapper), its direct div child (emotion yb component
       with target e15ve43o3 which sets backgroundColor:bgColor), stBottomBlockContainer,
       and stChatInputContainer. Use both class-attribute substring for emotion and data-testid. */
    div[data-testid="stChatInputContainer"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    [data-testid="stBottomBlockContainer"] > div > div,
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [class*="e15ve43o2"],
    [class*="e15ve43o3"],
    .stBottomBlockContainer,
    div[style*="position:fixed"][style*="bottom"],
    div[style*="position: fixed"][style*="bottom"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-bottom: 4px !important;
    }}
    div[data-testid="stChatInput"] button {{
        background-color: var(--accent-color) !important;
        border: none !important;
        border-radius: 50% !important;
        color: #ffffff !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.2s ease !important;
        padding: 0 !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
        background-color: var(--button-hover) !important;
    }}
    div[data-testid="stChatInput"] button svg {{
        fill: #ffffff !important;
        color: #ffffff !important;
    }}

    [data-testid="stChatMessage"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }}

    div[data-testid="stExpander"], .element-container {{
        border-radius: 8px !important;
    }}
    .stDataFrame {{
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    div.stButton > button {{
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }}
    div.stButton > button:hover {{
        background-color: var(--sidebar-bg) !important;
        border-color: var(--accent-color) !important;
    }}

    div.stButton > button[kind="primary"] {{
        background-color: var(--accent-color) !important;
        color: white !important;
        border: none !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: var(--button-hover) !important;
    }}

    hr {{
        border: 0 !important;
        border-top: 1px solid var(--border-color) !important;
        margin: 1.5rem 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Blog Writing Agent")

# Show chat bubbles (user topic and assistant confirmation) above the chat input
if st.session_state["submitted_topic"]:
    with st.chat_message("user"):
        st.write(st.session_state["submitted_topic"])

    if st.session_state["generation_done"]:
        with st.chat_message("assistant"):
            st.write("✨ Blog post generated successfully! Check the tabs below for the plan, evidence, preview, images, and logs.")

# Handle chat input at the bottom
prompt = st.chat_input("What should I write about?")
run_generation = False
if prompt:
    st.session_state["submitted_topic"] = prompt
    st.session_state["generation_done"] = False
    st.session_state["last_out"] = None
    run_generation = True

# Logic triggered by chat input
if run_generation:
    topic_text = st.session_state["submitted_topic"].strip()
    if not topic_text:
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic_text,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"),
                                                                              list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(
                    current_state.get("plan"), dict) else None,
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")
            st.session_state["generation_done"] = True
            st.rerun()

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)


out = st.session_state.get("last_out")
if out:

    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, width="stretch", hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)


    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("output/images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = Path("output/images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, width="stretch")

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )


    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem; color: var(--tab-inactive);">
            <h3 style="font-weight: 400; font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--text-color);">What should we write today?</h3>
            <p style="font-size: 0.95rem; margin: 0;">Enter a topic or prompt in the input below to generate a new blog post.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

