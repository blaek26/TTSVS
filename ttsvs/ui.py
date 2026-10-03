"""Local Gradio web UI for TTSVS (bound to 127.0.0.1 only)."""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

DEFAULT_VOICE = "Default voice"
CONSENT_TEXT = "I have permission to use this voice and will not use it to deceive or impersonate."


def _friendly(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "VoiceExistsError":
        return f"A voice with that name already exists. {exc}"
    if name == "VoiceNotFoundError":
        return f"Voice not found. {exc}"
    if isinstance(exc, ImportError):
        return f"A required package is missing: {exc}. See the README install steps."
    return f"{exc}" if str(exc) else name


def build_ui(engine: str = "chatterbox", device: str = "auto") -> gr.Blocks:
    from .engines import get_engine
    from .synth import Synthesizer
    from .voices import VoiceLibrary

    try:
        from .voices import check_duration
    except ImportError:  # pragma: no cover
        def check_duration(seconds):  # type: ignore[misc]
            return None

    library = VoiceLibrary()
    state: dict = {"synth": None}
    params_spec = get_engine(engine, device).supported_params()

    def get_synth() -> Synthesizer:
        # Created lazily: the model itself only loads on first generation.
        if state["synth"] is None:
            state["synth"] = Synthesizer(engine, device, library)
        return state["synth"]

    def voice_choices() -> list[str]:
        return [DEFAULT_VOICE] + [v.name for v in library.list()]

    def find_voice(name: str):
        for v in library.list():
            if v.name == name or v.id == name:
                return v
        raise gr.Error(f"Voice '{name}' not found.")

    def voice_rows() -> list[list]:
        return [[v.name, round(v.duration_s, 1), v.notes, v.created_at[:19]]
                for v in library.list()]

    def refreshed(selected=None):
        """Update every voice picker + table after a library change."""
        choices = voice_choices()
        manage = [c for c in choices if c != DEFAULT_VOICE]
        sel = selected if selected in manage else (manage[0] if manage else None)
        return (
            gr.update(choices=choices, value=selected if selected in choices else DEFAULT_VOICE),
            gr.update(choices=manage, value=sel),
            voice_rows(),
        )

    with gr.Blocks(title="TTSVS") as demo:
        gr.Markdown("# TTSVS\nLocal text-to-speech with voice cloning. "
                    "Everything runs on this computer.")
        with gr.Tabs():
            with gr.Tab("Speak"):
                text = gr.Textbox(label="Text", lines=6,
                                  placeholder="Type or paste what you want spoken...")
                voice_dd = gr.Dropdown(choices=voice_choices(), value=DEFAULT_VOICE,
                                       label="Voice")
                sliders = []
                with gr.Accordion("Settings", open=True):
                    for pname, spec in params_spec.items():
                        sliders.append(gr.Slider(
                            minimum=spec.get("min", 0.0), maximum=spec.get("max", 1.0),
                            value=spec.get("default", 0.5),
                            step=spec.get("step", 0.05), label=pname,
                            info=spec.get("help", "")))
                gen_btn = gr.Button("Generate", variant="primary")
                audio_out = gr.Audio(label="Result", type="numpy", interactive=False,
                                     **_download_kwargs())

            with gr.Tab("Voices"):
                gr.Markdown("Add a voice from a 10 to 30 second clip: clean, one speaker, "
                            "no music. Only clone voices you have permission to use.")
                new_name = gr.Textbox(label="Voice name")
                new_audio = gr.Audio(label="Voice sample (upload or record)",
                                     sources=["upload", "microphone"], type="filepath")
                new_notes = gr.Textbox(label="Notes (optional)")
                consent = gr.Checkbox(label=CONSENT_TEXT, value=False)
                add_btn = gr.Button("Add voice", variant="primary")
                add_msg = gr.Markdown()
                gr.Markdown("### Your voices")
                table = gr.Dataframe(headers=["Name", "Seconds", "Notes", "Added"],
                                     value=voice_rows(), interactive=False)
                manage_dd = gr.Dropdown(
                    choices=[c for c in voice_choices() if c != DEFAULT_VOICE],
                    label="Selected voice")
                preview = gr.Audio(label="Preview", interactive=False)
                with gr.Row():
                    rename_to = gr.Textbox(label="New name", scale=3)
                    rename_btn = gr.Button("Rename")
                    delete_btn = gr.Button("Delete", variant="stop")

        # ------------------------------------------------------ handlers
        def generate(txt, voice_name, *vals, progress=gr.Progress()):
            if not txt or not txt.strip():
                raise gr.Error("Please type some text first.")
            try:
                voice_id = None if voice_name in (None, "", DEFAULT_VOICE) \
                    else find_voice(voice_name).id
                params = dict(zip(params_spec.keys(), vals))
                progress(0, desc="Loading model (first time can take a while)...")
                synth = get_synth()

                def cb(done, total):
                    progress(done / max(total, 1), desc=f"Synthesizing {done}/{total}")

                sr, audio = synth.speak(txt, voice_id, progress=cb, **params)
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                raise gr.Error(_friendly(e))
            return sr, audio

        gen_btn.click(generate, [text, voice_dd, *sliders], audio_out)

        def add_voice(name, audio_path, notes, ok):
            if not ok:
                raise gr.Error("Please confirm you have permission to use this voice.")
            if not name or not name.strip():
                raise gr.Error("Please give the voice a name.")
            if not audio_path:
                raise gr.Error("Please upload or record a voice sample.")
            try:
                voice = library.add(name.strip(), audio_path, notes or "")
            except Exception as e:  # noqa: BLE001
                raise gr.Error(_friendly(e))
            msg = f"Added **{voice.name}** ({voice.duration_s:.1f} s)."
            warn = check_duration(voice.duration_s)
            if warn:
                msg += f"\n\nWarning: {warn}"
            return (msg, "", None, "", False, *refreshed(voice.name))

        add_btn.click(add_voice, [new_name, new_audio, new_notes, consent],
                      [add_msg, new_name, new_audio, new_notes, consent,
                       voice_dd, manage_dd, table])

        def do_preview(name):
            if not name:
                return None
            return str(find_voice(name).reference_path)

        manage_dd.change(do_preview, manage_dd, preview)

        def do_rename(name, new):
            if not name:
                raise gr.Error("Select a voice first.")
            if not new or not new.strip():
                raise gr.Error("Enter a new name.")
            try:
                v = library.rename(find_voice(name).id, new.strip())
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                raise gr.Error(_friendly(e))
            return (*refreshed(v.name), "")

        rename_btn.click(do_rename, [manage_dd, rename_to],
                         [voice_dd, manage_dd, table, rename_to])

        def do_delete(name):
            if not name:
                raise gr.Error("Select a voice first.")
            try:
                library.delete(find_voice(name).id)
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                raise gr.Error(_friendly(e))
            return (*refreshed(), None)

        delete_btn.click(do_delete, manage_dd, [voice_dd, manage_dd, table, preview])

    return demo


def _download_kwargs() -> dict:
    """Enable the download button across gradio versions."""
    import inspect
    params = inspect.signature(gr.Audio.__init__).parameters
    if "buttons" in params:  # gradio 6+
        return {"buttons": ["download"]}
    if "show_download_button" in params:
        return {"show_download_button": True}
    return {}


def launch(port: int = 7860, engine: str = "chatterbox", device: str = "auto") -> None:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo = build_ui(engine, device)
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=port, share=False,
                show_api=False, inbrowser=False)
