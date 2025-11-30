"""
Minimaler Proof-of-Concept für einen Pflegebericht-Generator
--------------------------------------------------------------------------
- Frontend: Streamlit mit Checkbox-/Multiselect-UI
- Backend: Direkter Aufruf der OpenAI LLM API (optional, mit Fallback ohne Key)
- Logik: Auswahl vordefinierter Tasks -> Templates -> Rohsätze -> (optional) LLM-Polish

Starten:
  1) Python 3.10+ empfohlen
  2) pip install -U streamlit openai jinja2 python-dotenv
  3) (optional) .env anlegen mit: OPENAI_API_KEY=sk-...
  4) streamlit run app.py

Hinweis:
- Wenn kein API-Key vorhanden ist, wird ein Fallback genutzt (nur Templates zusammenfügen).
- Dieses POC speichert nichts dauerhaft und enthält kein Whisper/Audio.
"""
import os
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from jinja2 import Template

try:
    from openai import OpenAI  # offizielles SDK (>= 1.0)
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# -------------------------
# Konfiguration & Konstanten
# -------------------------
APP_TITLE = "Pflegebericht – MAMA"
TIMEZONE = "Europe/Zurich"
DEFAULT_MODEL = "gpt-5.1-mini"  # kann im Sidebar umgestellt werden
TASKS_FILE = "tasks.json"       # optional: externe Speicherung/Laden

# Default-Task-Lexikon (wird beim Start ggf. durch tasks.json überschrieben)
DEFAULT_TASK_LIBRARY: Dict[str, Dict] = {
    "ZAEHNE_PUTZEN": {
        "label": "Zähne geputzt",
        "category": "Pflege",
        "template": "Ich habe Morris die Zähne geputzt.",
    },
    "WINDEN_WECHSELN": {
        "label": "Windeln gewechselt",
        "category": "Pflege",
        "template": "Ich habe Morris die Windeln gewechselt.",
    },
    "KLEIDER_WECHSELN": {
        "label": "Kleider gewechselt",
        "category": "Pflege",
        "template": "Ich habe Morris die Kleider gewechselt.",
    },
    "MEDIKAMENTE_GEGEBEN": {
        "label": "Medikamente gegeben",
        "category": "Medikamente",
        "template": "Ich habe die verordneten Medikamente verabreicht.",
    },
    "SONDE_NAHRUNG": {
        "label": "Nahrung per Sonde",
        "category": "Ernährung",
        "template": "Ich habe die Nahrung über die Sonde verabreicht.",
    },
    "EEG_TERMIN": {
        "label": "EEG im Spital",
        "category": "Arzt/Spital",
        "template": "Wir hatten einen Termin im Spital für ein EEG und ich habe Morris begleitet.",
    },
    "SCHULE_GEBRACHT": {
        "label": "Zur Schule gebracht",
        "category": "Mobilität",
        "template": "Ich habe Morris zur Schule gebracht und später wieder nach Hause begleitet.",
    },
    "TRANSFER_BETT_STUHL": {
        "label": "Transfer Bett ⇄ Rollstuhl",
        "category": "Mobilität",
        "template": "Ich habe Morris zwischen Bett und Rollstuhl transferiert und ihn sicher positioniert.",
    },
    "SPAERZIERGANG": {
        "label": "Spaziergang / Bewegung",
        "category": "Aktivität",
        "template": "Wir sind gemeinsam spazieren gegangen und haben leichte Mobilisationsübungen gemacht.",
    },
}

# Wird beim App-Start geladen
TASK_LIBRARY: Dict[str, Dict] = {}



# -------------------------
# Hilfsfunktionen
# -------------------------
import json

def load_task_library() -> Dict[str, Dict]:
    """Lädt Tasks aus tasks.json, sonst Default."""
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # einfache Validierung
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                return data
        except Exception:
            pass
    return DEFAULT_TASK_LIBRARY.copy()


def get_task_order(task_lib: Dict[str, Dict]) -> List[str]:
    """Ermittelt eine stabile Reihenfolge aller Tasks.
    - Wenn Tasks ein Feld 'order' (int) besitzen, danach sortieren
    - sonst alphabetisch nach Label
    """
    def sort_key(tid: str):
        meta = task_lib.get(tid, {})
        order = meta.get("order")
        label = meta.get("label", "")
        # order zuerst (None => sehr groß), dann label
        return (order if isinstance(order, int) else 10_000, label.lower())

    tids = list(task_lib.keys())
    tids.sort(key=sort_key)
    return tids

PRESETS_FILE = "presets.json"

def load_presets() -> Dict[str, List[str]]:
    """Lädt Abschnitt-Standards aus presets.json (Mapping: abschnittsname -> [task_ids])."""
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # nur Listen von Strings akzeptieren
                clean = {str(k): [str(x) for x in v if isinstance(x, str)] for k, v in data.items()}
                return clean
    except Exception:
        pass
    return {}

def save_presets(presets: Dict[str, List[str]]) -> None:
    """Schreibt die Abschnitt-Standards nach presets.json."""
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Preset konnte nicht gespeichert werden: {e}")


@dataclass
class ComposeInput:
    date: dt.date
    selected_task_ids: List[str]
    free_notes: str
    sections: List[str]


def compose_sentences(task_ids: List[str]) -> List[str]:
    sentences: List[str] = []
    for tid in task_ids:
        tpl = TASK_LIBRARY.get(tid, {}).get("template")
        if tpl:
            # Jinja2-Template für spätere Erweiterbarkeit
            sentences.append(Template(tpl).render())
    return sentences


def polish_with_llm(sentences: List[str], date_str: str, model: str, api_key: str | None) -> str:
    """Formt Sätze zu einem kurzen Pflegebericht. Fallback ohne Key: Rohtext.
    """
    base_text = " ".join(sentences).strip()
    if not base_text:
        return ""

    if not api_key or not _OPENAI_AVAILABLE:
        # Fallback: Nur zusammenfügen, leicht glätten
        return f"Pflegebericht vom {date_str}: {base_text}"

    try:
        client = OpenAI(api_key=api_key)
        system_msg = (
            "Du schreibst präzise, sachliche Pflegeberichte auf Deutsch. "
            "Nutze Vergangenheitsform, klare Zeiten, keine Spekulation, keine Du-Anreden."
        )
        user_msg = (
            f"Bitte formuliere aus den folgenden Sätzen einen kurzen, zusammenhängenden Pflegebericht "
            f"für den {date_str}. Sätze:\n- " + "\n- ".join(sentences)
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Sicherheitsnetz: niemals crashen
        return f"Pflegebericht: {base_text}\n\n(Hinweis: LLM-Formulierung nicht verfügbar: {e})"


# -------------------------
# UI – Streamlit App
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

# Tasks initial laden und in Session State halten
if "task_library" not in st.session_state:
    st.session_state.task_library = load_task_library()
if "checked" not in st.session_state:
    # Merkt sich, welche Task-IDs heute angehakt sind
    st.session_state.checked = set()

TASK_LIBRARY = st.session_state.task_library

# Presets (Standard-Abläufe) einmalig laden
if "presets" not in st.session_state:
    st.session_state.presets = load_presets()


st.title("📝 Pflegebericht – MAMA")

with st.sidebar:
    st.header("Einstellungen")
    today = dt.date.today()
    date_value = st.date_input("Datum", value=today)

    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        value=env_key,
        help=(
            "Wenn leer: Fallback ohne LLM (nur Vorlagensätze). "
            "Du kannst den Key hier eingeben oder per .env setzen."
        ),
    )

    model = st.selectbox(
        "LLM-Modell",
        options=[DEFAULT_MODEL, "gpt-4o-mini", "gpt-4.1-mini"],
        index=0,
        help="Kleines, schnelles Modell reicht für Feinschliff im POC.",
    )

st.markdown("Wähle unten die heutigen Aufgaben pro **Abschnitt** (Morgen/Mittag/Abend …). **Ein Klick auf den Task-Button markiert/entmarkiert** ihn. Optional pro Abschnitt Kommentar hinzufügen.")

# ------------------------- Abschnitte konfigurieren -------------------------
# Anzahl der Abschnitte
num_sections = st.number_input("Wie viele Abschnitte heute?", min_value=1, max_value=6, value=4, step=1)

# Standardnamen vorschlagen
default_names = ["Morgen", "Mittag", "Nachmittag","Abend", "Nacht", "Spital", "Sonstiges"]

# Flache, einheitliche Taskliste (keine Kategorien) in stabiler Reihenfolge
ALL_TIDS_ORDERED = get_task_order(TASK_LIBRARY)

section_defs = []
for i in range(int(num_sections)):
    with st.expander(f"Abschnitt {i+1}", expanded=True if i < 3 else False):
        name = st.text_input(
            f"Name Abschnitt {i+1}",
            value=default_names[i] if i < len(default_names) else f"Abschnitt {i+1}",
            key=f"sec_name_{i}"
        )
        # optionaler Kommentar pro Abschnitt
        note = st.text_area(
            f"Kommentar zu {name} (optional)",
            key=f"sec_note_{i}",
            placeholder="z. B. Zeiten, Besonderheiten …"
        )

        # Suchfeld pro Abschnitt
        search = st.text_input(
            f"Aufgaben filtern ({name})",
            key=f"search_{i}",
            placeholder="z. B. Medikamente, Zähne, Sonde …"
        )
        # Reihenfolge-Liste für diesen Abschnitt initialisieren
        order_key = f"order_{i}"
        if order_key not in st.session_state:
            st.session_state[order_key] = []

        # --- Abschnitt-Standard speichern / laden (links) + Abschnitt zurücksetzen (rechts) ---
        preset_name = name.strip() or f"Abschnitt {i+1}"

        ctrl_save, ctrl_load, spacer, ctrl_reset = st.columns([1, 1, 6, 1])

        with ctrl_save:
            if st.button("Standard speichern", key=f"save_preset_{i}", use_container_width=True):
                seq = [tid for tid in st.session_state[order_key] if tid in TASK_LIBRARY]
                st.session_state.presets[preset_name] = seq
                save_presets(st.session_state.presets)
                st.success("Standard gespeichert", icon="✅")
                st.rerun()

        with ctrl_load:
            if st.button("Standard laden", key=f"load_preset_{i}", use_container_width=True):
                presets = st.session_state.presets
                if preset_name in presets:
                    seq = [tid for tid in presets[preset_name] if tid in TASK_LIBRARY]
                    # Auswahl dieses Abschnitts leeren
                    for k in list(st.session_state.keys()):
                        if k.startswith(f"sel_{i}_"):
                            del st.session_state[k]
                    # Reihenfolge + Auswahl setzen
                    st.session_state[order_key] = seq[:]
                    for tid in seq:
                        st.session_state[f"sel_{i}_{tid}"] = True
                    st.rerun()
                else:
                    st.info("Kein Standard für diesen Abschnittsnamen gefunden.")

        # spacer bleibt leer (nur zum Schieben)

        with ctrl_reset:
            if st.button("Abschnitt zurücksetzen", key=f"reset_section_{i}", use_container_width=True):
                # alle sel_-Keys dieses Abschnitts löschen
                for k in list(st.session_state.keys()):
                    if k.startswith(f"sel_{i}_"):
                        del st.session_state[k]
                st.session_state[order_key] = []
                st.rerun()

        st.write("Klicke die **Task-Namen** um sie zu markieren. Markierte Buttons werden hervorgehoben.")

        cols = st.columns(3)
        # Wir bauen selected_ordered am Ende aus der gespeicherten Tap-Reihenfolge
        for idx, tid in enumerate(ALL_TIDS_ORDERED):
            label = TASK_LIBRARY[tid]["label"]
            if search.strip() and search.strip().lower() not in label.lower():
                continue

            sel_key = f"sel_{i}_{tid}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = False

            active = bool(st.session_state[sel_key])
            # Positionsnummer, falls ausgewählt und in der Reihenfolge vorhanden
            ord_list = st.session_state[order_key]
            pos = (ord_list.index(tid) + 1) if (active and tid in ord_list) else None

            # Button-Label mit Nummer (z. B. "1. Zähne geputzt")
            num_prefix = f"{pos}. " if pos is not None else ""
            btn_label = (f"[{pos}] " if pos is not None else "") + label + (" ✅" if active else "")
            btn_type = "primary" if active else "secondary"

            # Eindeutiger Widget-Key
            btn_key = f"btn_{i}_{tid}_{idx}"

            col = cols[idx % 3]
            with col:
                pressed = st.button(
                    btn_label,
                    key=btn_key,
                    type=btn_type,
                    use_container_width=True,
                )

            if pressed:
                # Toggle Auswahl
                st.session_state[sel_key] = not active

                # Reihenfolge aktualisieren
                ord_list = st.session_state[order_key]
                if st.session_state[sel_key]:        # jetzt aktiv
                    if tid not in ord_list:
                        ord_list.append(tid)         # ans Ende = Tap-Reihenfolge
                else:                                # jetzt inaktiv
                    if tid in ord_list:
                        ord_list.remove(tid)

                st.rerun()

        # Aus der gespeicherten Reihenfolge die aktuell aktiven Tasks ableiten (in genau dieser Reihenfolge)
        selected_ordered: List[str] = [
            tid for tid in st.session_state[order_key]
            if st.session_state.get(f"sel_{i}_{tid}", False)
        ]

        # Kleine Vorschau unter der Liste
        if selected_ordered:
            preview = " → ".join(TASK_LIBRARY[t]["label"] for t in selected_ordered)
            st.caption(f"Reihenfolge: {preview}")

        section_defs.append({"name": name, "note": note, "selected": selected_ordered})



st.divider()

col1, col2 = st.columns([1,1])
with col1:
    generate_btn = st.button("Bericht generieren", type="primary")
with col2:
    # Reset: setzt Auswahl und Eingaben zurück
    if st.button("Alles zurücksetzen"):
        keys_to_del = [
            k for k in st.session_state.keys()
            if k.startswith("sel_")
            or k.startswith("sec_note_")
            or k.startswith("search_")
            or k.startswith("sec_name_")
            or k.startswith("order_")   # ← Reihenfolgen auch zurücksetzen
        ]
        for k in keys_to_del:
            del st.session_state[k]
        st.rerun()


# ------------------------- Generierung -------------------------

def compose_from_sections(sections: List[dict]) -> List[str]:
    sentences: List[str] = []
    for sec in sections:
        name = sec["name"].strip()
        selected: List[str] = sec.get("selected", [])
        note = sec.get("note", "").strip()
        block: List[str] = []
        for tid in selected:
            tpl = TASK_LIBRARY.get(tid, {}).get("template")
            if not tpl:
                continue
            sent = Template(tpl).render()
            block.append(sent)
        if note:
            block.append(f"Hinweis: {note}")
        if block:
            if name:
                sentences.append(f"[{name}]")
            sentences.extend(block)
    return sentences

# === Bericht auf Knopfdruck erzeugen & anzeigen ===
if generate_btn:
    sentences = compose_sentences([])  # nur für Robustheit; wird unten überschrieben
    try:
        sentences = compose_from_sections(section_defs)
    except Exception as e:
        st.error(f"Fehler beim Zusammenstellen der Sätze: {e}")
        sentences = []

    if not sentences:
        st.info("Bitte markiere mindestens einen Task in einem Abschnitt oder füge einen Kommentar hinzu.")
    else:
        date_str = date_value.strftime("%d.%m.%Y")
        result_text = polish_with_llm(sentences, date_str, model=DEFAULT_MODEL if not 'model' in locals() else model, api_key=api_key)

        st.subheader("Vorschau")
        st.text_area("Ergebnis", value=result_text, height=320)
        st.download_button(
            "Als TXT speichern",
            data=result_text,
            file_name=f"pflegebericht_{date_value.isoformat()}.txt",
            mime="text/plain",
        )

st.caption("Dieses POC unterstützt **mehrfache Durchführungen pro Task** und **Abschnitte** (z. B. Morgen/Mittag/Abend). Tasks werden optional aus 'tasks.json' geladen.")
