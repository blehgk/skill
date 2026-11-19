"""Gradio front-end for the Dead by Daylight auto skill check bot.

The UI is a thin shell: :class:`dbd.pilot.SkillCheckPilot` owns the capture,
classification and key injection, and this module only translates between that
loop and the widgets.
"""

import argparse
import base64
import io
import json
import os
import tempfile
from time import sleep

# Gradio caches every image it serves and never prunes them. Left at its default
# that cache lands in the working directory: streaming frames during a RUN
# session produced ~1500 files in the project root. Point it at the system temp
# directory instead. Must happen before gradio is imported.
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "dbd-skillcheck"))

from gradio import (  # noqa: E402
    Accordion, Blocks, Button, Column, Dropdown, Error, HTML, Image, Info,
    Markdown, Number, Row, Slider, skip,
)

from dbd.capture import BACKENDS, dxcam_available
from dbd.pilot import PilotError, SkillCheckPilot, describe_provider
from dbd.utils.directkeys import VK_XBUTTON2
from dbd.utils.monitor import get_frame, get_monitor_attributes, get_monitors

MODELS_FOLDER = "models"
DEFAULT_MODEL = f"{MODELS_FOLDER}/model.onnx"

#: Square edge of the crop sent to the model (~224 px at 1080p).
CROP_SIZE = 224
#: Debug preview is a wider crop so the whole skill check ring is visible.
PREVIEW_CROP_SIZE = 520

#: Design tokens and layout CSS.
#:
#: Tokens come from the ui-ux-pro-max design system for this product shape
#: ("game automation control panel", dark, density 8/10, variance 3/10):
#: Minimalism/Swiss style, dark-tech palette with a status green, and
#: ``--muted-foreground`` at #94A3B8 so secondary text keeps ~7:1 contrast on
#: the card rather than the near-invisible grey it had before.
#:
#: Gradio rewrites selectors passed as ``css`` by prefixing them with its scope,
#: which inserts an extra ``.contain`` descendant step. Plain descendant rules
#: (``.gradio-container .block``) and ``:has()`` rules survive that intact, so
#: everything below works through ``css``; nothing needs a JS workaround.
COMPACT_CSS = """
:root {
  --dbd-primary: #1E293B;
  --dbd-accent: #22C55E;
  --dbd-bg: #0F172A;
  --dbd-card: #1B2336;
  --dbd-fg: #F8FAFC;
  --dbd-muted: #272F42;
  --dbd-muted-fg: #94A3B8;
  --dbd-border: #475569;
  --dbd-ring: #FFFFFF;
  --dbd-destructive: #EF4444;
  --dbd-warn: #F59E0B;
}

footer {display: none !important}
.gradio-container {
  max-width: 100% !important;
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

/* --- Accessibility (ui-ux-pro-max: focus must stay visible on every operable
   control; hover feedback with a 150-300 ms transition; reduced motion
   respected). --- */
.gradio-container *:focus-visible {
  outline: 2px solid var(--dbd-ring) !important;
  outline-offset: 2px !important;
}
.gradio-container button {
  cursor: pointer !important;
  transition: background-color 200ms ease, border-color 200ms ease, color 200ms ease;
}
.gradio-container button:hover:not(:disabled) {filter: brightness(1.12)}
.gradio-container button:disabled {cursor: not-allowed !important; opacity: .55}
@media (prefers-reduced-motion: reduce) {
  .gradio-container * {transition: none !important; animation: none !important}
}

/* Secondary text at the muted token instead of Gradio's dimmer grey. */
.gradio-container label, .gradio-container .label-wrap span {
  color: var(--dbd-muted-fg) !important;
}

/* --- Status badge. Colour is never the only signal: the state word is the
   content and it is exposed as a live region (see status_badge()). ON/Pausa are
   bright fills with dark text; OFF keeps the destructive red at white text. --- */
.dbd-badge {
  text-align: center; font-size: 34px; font-weight: 800; letter-spacing: .06em;
  border-radius: 10px; padding: 14px 0; line-height: 1.15;
  font-variant-numeric: tabular-nums;
}
.dbd-badge--on    {background: var(--dbd-accent); color: #0F172A}
.dbd-badge--pausa {background: var(--dbd-warn); color: #0F172A}
.dbd-badge--off   {background: var(--dbd-destructive); color: #FFFFFF}

/* --- Action buttons. Gradio's stock primary is orange, which fights the
   dark-tech palette; the accent green is this product's "go" colour. --- */
.gradio-container button.primary {
  background: var(--dbd-accent) !important;
  border-color: var(--dbd-accent) !important;
  color: #0F172A !important; font-weight: 800 !important;
  letter-spacing: .04em;
}
.gradio-container button.stop {
  background: transparent !important;
  border: 1px solid var(--dbd-destructive) !important;
  color: #FCA5A5 !important; font-weight: 800 !important;
  letter-spacing: .04em;
}
.gradio-container button.stop:hover:not(:disabled) {
  background: var(--dbd-destructive) !important; color: #FFFFFF !important;
}

/* --- Preview. It is a live readout, not a file to download, so Gradio's
   fullscreen/download overlay goes away. --- */
.gradio-container .icon-button-wrapper,
.gradio-container .download-link {display: none !important}
.gradio-container .image-container img {border-radius: 8px}
"""


#: State word -> (css modifier, screen-reader sentence).
#: The badge is a live region so its changes are announced without stealing
#: focus, and it announces a meaningful phrase rather than a bare word.
_BADGE_STATES = {
    "ON": ("on", "Bot activo, analizando la pantalla"),
    "PAUSA": ("pausa", "Bot en pausa"),
    "OFF": ("off", "Bot detenido"),
}


def status_badge(state):
    """Render the status badge.

    ``role="status"`` + ``aria-atomic`` make this one polite live region, so a
    state change is announced once with its full meaning instead of the screen
    reader reading a bare "OFF". The visible word is the primary signal; colour
    only reinforces it.
    """
    modifier, spoken = _BADGE_STATES.get(state, _BADGE_STATES["OFF"])
    return (
        '<div class="dbd-badge dbd-badge--{mod}" role="status" aria-atomic="true">'
        '<span aria-hidden="true">{word}</span>'
        '<span class="sr-only" style="position:absolute;width:1px;height:1px;'
        'overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap">{spoken}</span>'
        "</div>".format(mod=modifier, word=state, spoken=spoken)
    )


class MonitorBot:
    """Owns the running :class:`SkillCheckPilot` and drives the UI loop."""

    def __init__(self):
        self.pilot = None
        self.paused = False

    # -- lifecycle ---------------------------------------------------------
    def start(self, monitor_id, hit_ante_ms, static_gate, confirm_frames, backend):
        """Build a pilot, then yield the UI updates for each processed frame."""
        self.stop()
        if monitor_id is None:
            raise Error("Invalid monitor option")

        try:
            self.pilot = SkillCheckPilot(
                model_path=DEFAULT_MODEL,
                monitor_id=int(monitor_id),
                crop_size=CROP_SIZE,
                hit_ante_ms=hit_ante_ms,
                # cpu_threads=None lets AI_model pick its measured optimum.
                static_gate_threshold=1.0 if static_gate else None,
                confirm_frames=int(confirm_frames),
                capture_backend=backend,
            )
        except PilotError as exc:
            raise Error(f"Error when loading AI model: {exc}", duration=0)

        Info(f"{describe_provider(self.pilot.provider)} | capture: {self.pilot.capture_backend}")

        self.paused = False
        yield skip(), skip(), status_badge("ON")

        try:
            while not self.pilot.stopped():
                sample = self.pilot.step(paused=self.paused)

                if sample["toggle_pause"]:
                    self.paused = not self.paused
                    Info("Auto skill check paused (mouse button)" if self.paused
                         else "Auto skill check resumed (mouse button)")

                paused = self.paused
                status = status_badge("PAUSA" if paused else "ON")
                frame = sample["frame"]

                if paused:
                    # Report the pause as soon as it happens, then idle.
                    yield 0.0, skip(), status
                    while self.paused and not self.pilot.stopped():
                        sample = self.pilot.step(paused=True)
                        if sample["toggle_pause"]:
                            self.paused = False
                            Info("Auto skill check resumed (mouse button)")
                        sleep(0.05)
                    continue

                yield sample["fps"], _preview_html(frame), status
        except Exception as exc:  # keep the UI alive and surface the reason
            raise Error(f"Monitoring stopped: {exc}")
        finally:
            self.stop()

    def stop(self):
        """Stop the loop and release the model. Safe to call when idle."""
        if self.pilot is not None:
            pilot, self.pilot = self.pilot, None
            pilot.stop()
        self.paused = False
        return 0.0, status_badge("OFF")

    def on_finished(self):
        """Called when the generator ends for any reason."""
        return self.stop()


def _preview_html(frame):
    """Raw ``<img>`` markup for the debug preview.

    Returned as an HTML string rather than a Gradio ``Image`` component: a
    component inside a closed accordion reports a 0x0 box, so any visibility
    check on it is meaningless, and the string lets the panel own the styling.
    """
    if frame is None:
        return "<p style='color:var(--dbd-muted-fg);font-size:.8rem;margin:0'>" \
               "Sin captura todavía.</p>"

    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<img alt="Último skill check capturado, recorte central de la pantalla" '
        'src="data:image/png;base64,{data}" '
        "style='width:100%;aspect-ratio:1/1;object-fit:contain;"
        "border:1px solid var(--dbd-border);border-radius:10px;"
        "background:#0B1020;display:block'>".format(data=encoded)
    )

def preview_frame(monitor_id):
    """Centre crop of ``monitor_id``, used for the debug preview."""
    if monitor_id is None:
        return skip()
    frame = get_frame(get_monitor_attributes(monitor_id, crop_size=PREVIEW_CROP_SIZE))
    return _preview_html(frame)


def build_ui():
    """Assemble the Blocks app. Separated so it can be built without launching."""
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    if not os.path.isfile(DEFAULT_MODEL):
        raise Error(f"No AI model found in {MODELS_FOLDER}/", duration=0)

    monitor_choices = get_monitors()
    if not monitor_choices:
        raise Error("No monitors detected", duration=0)

    bot = MonitorBot()

    # The panel must stay small enough to sit beside a portrait monitor, so the
    # explanatory prose that used to be per-control `info` text lives in the
    # collapsed settings accordion instead of padding every field.
    with Blocks(
        title="Auto skill check",
        css=COMPACT_CSS,
        # Typography from the design system. Falls back to system-ui, so the
        # panel still looks right with no network access.
        head='<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap">',
    ) as webui:
        Markdown("<h1 style='text-align:center;margin:0 0 8px 0'>"
                 "DBD Auto skill check</h1>")

        # Layout follows the original (D:) front-end: two `variant="panel"`
        # columns with Gradio's own grouping, which is what made that version
        # read as a finished panel. Rebuilding it as ungrouped full-width
        # controls with ~200 lines of CSS overrides looked worse and fought the
        # framework for no gain; this keeps Gradio's native layout and only
        # layers the design-system tokens on top of it.
        with Row(equal_height=True):
            with Column(variant="panel", scale=2):
                status = HTML(value=status_badge("OFF"), label=None)
                with Row():
                    run_button = Button("RUN", variant="primary", scale=1)
                    stop_button = Button("STOP", variant="stop", scale=1)
                fps = Number(label="FPS", interactive=False)

                with Accordion("Ajustes", open=False):
                    monitor_id = Dropdown(
                        choices=monitor_choices,
                        value=monitor_choices[0][1],
                        label="Monitor",
                    )
                    backend = Dropdown(
                        choices=[("Auto (dxcam si puede)", "auto"),
                                 ("Desktop Duplication (dxcam)", "dxcam"),
                                 ("GDI (mss)", "mss")],
                        value="auto",
                        label="Captura",
                    )
                    confirm_frames = Slider(
                        minimum=1, maximum=4, step=1, value=1,
                        label="Frames a confirmar",
                    )
                    hit_ante = Slider(
                        minimum=0, maximum=50, step=5, value=3,
                        label="Retardo ante-frontier (ms)",
                    )
                    static_gate = Dropdown(
                        choices=[("Activada", True), ("Desactivada", False)],
                        value=True,
                        label="Puerta de frames estáticos",
                    )

            with Column(variant="panel", scale=1):
                image_pil = HTML(value=_preview_html(None), label=None)

        # concurrency_limit=None so STOP is not queued behind the endless RUN
        # generator; it must be able to interrupt the loop immediately.
        run_button.click(
            fn=bot.start,
            inputs=[monitor_id, hit_ante, static_gate, confirm_frames, backend],
            outputs=[fps, image_pil, status],
            show_progress="hidden",
            concurrency_limit=None,
        ).then(fn=bot.on_finished, outputs=[fps, status])

        stop_button.click(fn=bot.stop, outputs=[fps, status])

        # Refresh the debug preview when the monitor changes and on page load.
        monitor_id.change(fn=preview_frame, inputs=monitor_id, outputs=image_pil)
        webui.load(fn=preview_frame, inputs=monitor_id, outputs=image_pil)

    return webui, bot


def main():
    parser = argparse.ArgumentParser(description="DBD auto skill check web UI")
    # A fixed port avoids the "it moved to 7861" surprise when another instance
    # is already running: the bind simply fails and says so.
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    webui, bot = build_ui()
    try:
        webui.launch(server_name=args.host, server_port=args.port,
                     share=args.share, show_api=False)
    except KeyboardInterrupt:
        print("\nWeb UI stopped by user.")
    finally:
        bot.stop()
        print("Resources released.")


if __name__ == "__main__":
    main()
