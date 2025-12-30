import os
import json
import datetime as dt
from typing import Dict, List, Any
import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv

# Lade Umgebungsvariablen (.env)
load_dotenv()

# Versuche LiteLLM zu importieren
try:
    from litellm import completion
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

# -------------------------
# Konfiguration
# -------------------------
APP_TITLE = "Pflegebericht – MAMA"
# Modellname wie im Proxy (ohne 'o' laut Screenshot)
DEFAULT_MODEL = "azure/gpt-5" 

# Dateinamen
FILES = {
    "tasks": "tasks.json",
    "presets": "presets.json",
    "prompt": "prompt.json"
}

# -------------------------
# Hilfsfunktionen
# -------------------------
def load_json(filepath: str, default: Any = None) -> Any:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Fehler beim Laden von {filepath}: {e}")
    return default if default is not None else {}

def save_json(filepath: str, data: Any):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Fehler beim Speichern in {filepath}: {e}")

def get_sorted_task_ids(task_lib: Dict) -> List[str]:
    def sort_key(tid):
        meta = task_lib.get(tid, {})
        return (meta.get("order", 10000), meta.get("label", "").lower())
    ids = list(task_lib.keys())
    ids.sort(key=sort_key)
    return ids

# -------------------------
# Logik: Bericht erstellen
# -------------------------
def generate_report_with_litellm(raw_text_blocks: List[str], model: str) -> str:
    if not _LITELLM_AVAILABLE:
        return "Fehler: 'litellm' ist nicht installiert."

    full_input_text = "\n".join(raw_text_blocks)
    if not full_input_text.strip():
        return "Keine Aufgaben ausgewählt."

    # Prompt laden
    prompt_data = load_json(FILES["prompt"], {})
    system_prompt = prompt_data.get("system", "Du bist ein hilfreicher Assistent.")
    example_in = prompt_data.get("example_input", "")
    example_out = prompt_data.get("example_output", "")

    messages = [{"role": "system", "content": system_prompt}]

    if example_in and example_out:
        messages.append({"role": "user", "content": f"Hier ist ein Beispiel:\n{example_in}"})
        messages.append({"role": "assistant", "content": example_out})

    user_request = (
        f"Erstelle den Bericht für Heute.\n"
        f"Hier sind die Notizen:\n\n{full_input_text}"
    )
    messages.append({"role": "user", "content": user_request})

    # Proxy Daten aus .env
    proxy_key = os.getenv("LITELLM_API_KEY")
    proxy_url = os.getenv("LITELLM_API_BASE")

    try:
        response = completion(
            model=model,
            messages=messages,
            api_key=proxy_key,
            base_url=proxy_url,
            drop_params=True,  # WICHTIG: Verhindert Fehler bei GPT-5/Reasoning Models
            custom_llm_provider="openai"
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler: {str(e)}\n\n(Bitte Modell-Namen und .env prüfen)"

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

# -------------------------
# App Start
# -------------------------
if "task_library" not in st.session_state:
    tasks = load_json(FILES["tasks"])
    if not tasks:
        st.error(f"⚠️ '{FILES['tasks']}' fehlt!")
        st.stop()
    st.session_state.task_library = tasks

TASK_LIB = st.session_state.task_library

if "presets" not in st.session_state:
    st.session_state.presets = load_json(FILES["presets"], default={})

if "checked" not in st.session_state:
    st.session_state.checked = set()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Einstellungen")
    date_val = st.date_input("Datum", value=dt.date.today())
    # Modell Name editierbar, Default ist azure/gpt-5
    model_name = st.text_input("Modell", value=DEFAULT_MODEL) 

# -------------------------
# Hauptbereich
# -------------------------
st.title(APP_TITLE)

num_sections = st.number_input("Anzahl Tages-Abschnitte", min_value=1, max_value=6, value=4)
default_section_names = ["Morgen", "Mittag", "Nachmittag", "Abend", "Nacht", "Spital"]
sorted_task_ids = get_sorted_task_ids(TASK_LIB)
sections_data = []

for i in range(int(num_sections)):
    with st.expander(f"Abschnitt {i+1}", expanded=(i < 3)):
        c_name, c_note = st.columns([1, 2])
        sec_name = c_name.text_input(f"Name", value=default_section_names[i] if i < len(default_section_names) else f"Abschnitt {i+1}", key=f"sn_{i}", label_visibility="collapsed")
        sec_note = c_note.text_input(f"Notiz", placeholder="Besonderheiten...", key=f"snot_{i}", label_visibility="collapsed")

        order_key = f"order_{i}"
        if order_key not in st.session_state: st.session_state[order_key] = []
        
        # Presets Buttons
        cp1, cp2, cp3 = st.columns([1, 1, 5])
        if cp1.button("💾 Speichern", key=f"sv_{i}"):
            sel = [t for t in st.session_state[order_key] if t in TASK_LIB]
            st.session_state.presets[sec_name] = sel
            save_json(FILES["presets"], st.session_state.presets)
            st.toast("Gespeichert!", icon="✅")
        
        if cp2.button("📂 Laden", key=f"ld_{i}"):
            if sec_name in st.session_state.presets:
                # Reset selection
                for k in list(st.session_state.keys()):
                    if k.startswith(f"sel_{i}_"): del st.session_state[k]
                # Set selection
                items = st.session_state.presets[sec_name]
                st.session_state[order_key] = items
                for tid in items: st.session_state[f"sel_{i}_{tid}"] = True
                st.rerun()
        
        # --- Task Buttons mit NUMMERN ---
        st.markdown("---")
        cols = st.columns(3)
        current_order_list = st.session_state[order_key]

        for idx, tid in enumerate(sorted_task_ids):
            task_def = TASK_LIB[tid]
            raw_label = task_def.get("label", tid)
            
            sel_key = f"sel_{i}_{tid}"
            if sel_key not in st.session_state: st.session_state[sel_key] = False
            is_selected = st.session_state[sel_key]
            
            # --- Hier wird die Nummer berechnet ---
            if is_selected and tid in current_order_list:
                # +1 damit es bei 1 losgeht und nicht bei 0
                pos = current_order_list.index(tid) + 1
                btn_text = f"[{pos}] {raw_label} ✅"
            else:
                btn_text = raw_label
            # --------------------------------------

            if cols[idx % 3].button(btn_text, key=f"btn_{i}_{tid}", type="primary" if is_selected else "secondary", use_container_width=True):
                st.session_state[sel_key] = not is_selected
                
                # Reihenfolge updaten
                if st.session_state[sel_key]:
                    if tid not in current_order_list: current_order_list.append(tid)
                else:
                    if tid in current_order_list: current_order_list.remove(tid)
                st.rerun()

        final_sel = [t for t in st.session_state[order_key] if st.session_state.get(f"sel_{i}_{t}")]
        sections_data.append({"name": sec_name, "note": sec_note, "tasks": final_sel})

st.divider()

col_gen, col_rst = st.columns([1, 1])

if col_gen.button("🚀 Bericht generieren", type="primary"):
    raw_blocks = []
    for sec in sections_data:
        if not sec["tasks"] and not sec["note"].strip(): continue
        
        content = []
        for tid in sec["tasks"]:
            tpl = TASK_LIB.get(tid, {}).get("template", "")
            if tpl: content.append(Template(tpl).render())
        
        if sec["note"].strip(): content.append(f"Notiz: {sec['note']}")
        
        if content:
            raw_blocks.append(f"[{sec['name']}] " + " ".join(content))

    with st.spinner("Bericht wird erstellt.."):
        res = generate_report_with_litellm(raw_blocks, model_name)

    st.subheader("Ergebnis")
    
    # ZURÜCK ZUR ALTEN LÖSUNG: st.text_area
    # Das garantiert korrekten Zeilenumbruch ohne CSS-Hacks.
    st.text_area("Bericht", value=res, height=400)
    
    st.download_button("Download .txt", res, f"Bericht_{date_val}.txt")

if col_rst.button("Alles zurücksetzen"):
    st.session_state.clear()
    st.rerun()