#!/usr/bin/env python3
"""Side-by-side listening comparison for Larynx acceptance gates.

Serves a single local web page that plays two WAV files next to each other so
you can A/B them. Generic by design: pass any two WAV files and their real
filenames are what get shown on the page — nothing about the labels is
hardcoded, so this same script is meant to be reused across phases (Flow-only,
all-GGML, HiFT-only, whatever pair you're comparing) without editing it.

The server is stdlib-only (``http.server``): no pip dependencies, no framework.
It binds to ``0.0.0.0`` only, serves the two files plus the comparison page,
and exits when you Ctrl-C.

Usage:
    python3 scripts/listen_compare.py <file_a> <file_b> [--port 8765]
                                      [--label-a TEXT] [--label-b TEXT]
                                      [--no-browser]

Defaults:
    file_a  = wavs/wav_ggml_full.wav
    file_b  = wavs/wav_reference.wav
    labels  = each file's basename, unless overridden with --label-a/--label-b
    port    = 8765
"""

import argparse
import html
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # doesn't actually send anything; just asks
                                    # the OS which interface would be used
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_A = os.path.join(ROOT, "wavs", "wav_ggml_full.wav")
DEFAULT_B = os.path.join(ROOT, "wavs", "wav_reference.wav")

# Internal route names are intentionally decoupled from the real filenames —
# the browser doesn't care what the URL path is called, only the labels
# shown on the page need to reflect the actual files being compared.
_PATH_A = DEFAULT_A
_PATH_B = DEFAULT_B
_LABEL_A = os.path.basename(DEFAULT_A)
_LABEL_B = os.path.basename(DEFAULT_B)


def render_page():
    """Build the HTML fresh from whatever files/labels are currently set,
    instead of a frozen template with filenames baked in."""
    label_a = html.escape(_LABEL_A)
    label_b = html.escape(_LABEL_B)
    path_a = html.escape(_PATH_A)
    path_b = html.escape(_PATH_B)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Larynx — A/B listening</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 2rem; background: #14161a; color: #e6e8eb;
    max-width: 1100px; margin-inline: auto;
  }}
  h1 {{ font-size: 1.3rem; font-weight: 650; margin: 0 0 .25rem; }}
  .sub {{ color: #9aa0a6; font-size: .9rem; margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 720px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: #1d2025; border: 1px solid #2a2e35; border-radius: 10px;
    padding: 1.1rem 1.25rem 1.25rem;
  }}
  .card h2 {{
    font-size: .95rem; margin: 0 0 .15rem; font-weight: 600;
    word-break: break-all;
  }}
  .tag {{
    display: inline-block; font-size: .72rem; letter-spacing: .04em;
    padding: .1rem .5rem; border-radius: 999px; margin-bottom: .8rem;
  }}
  .tag.a {{ background: #123a2a; color: #7ee2b0; }}
  .tag.b {{ background: #1d2b4a; color: #8fb8ff; }}
  audio {{ width: 100%; margin-top: .4rem; }}
  .missing {{
    color: #e5484d; font-size: .85rem; margin-top: .6rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .btn-row {{ margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }}
  button {{
    background: #2a2e35; color: #e6e8eb; border: 1px solid #3a3f47;
    padding: .35rem .7rem; border-radius: 7px; cursor: pointer; font-size: .82rem;
  }}
  button:hover {{ background: #333842; }}
  .foot {{ margin-top: 1.75rem; color: #7c828a; font-size: .82rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
</style>
</head>
<body>
  <h1>Larynx — A/B listening</h1>
  <div class="sub">Same input, two renders. Whatever difference you hear is
    attributable to whatever differs between these two files — check the
    paths below to know exactly what's being compared.</div>

  <div class="grid">
    <div class="card">
      <span class="tag a">A</span>
      <h2>{label_a}</h2>
      <audio id="a" controls loop preload="auto" src="/file_a.wav"></audio>
      <div class="btn-row">
        <button onclick="play('a')">Play</button>
        <button onclick="stop('a')">Stop</button>
      </div>
      <div class="missing" id="m-a" style="display:none">
        missing: <span id="p-a"></span>
      </div>
    </div>

    <div class="card">
      <span class="tag b">B</span>
      <h2>{label_b}</h2>
      <audio id="b" controls loop preload="auto" src="/file_b.wav"></audio>
      <div class="btn-row">
        <button onclick="play('b')">Play</button>
        <button onclick="stop('b')">Stop</button>
      </div>
      <div class="missing" id="m-b" style="display:none">
        missing: <span id="p-b"></span>
      </div>
    </div>
  </div>

  <div class="foot">
    A: <code>{path_a}</code><br>
    B: <code>{path_b}</code>
  </div>

  <script>
    const play = (id) => {{
      const el = document.getElementById(id);
      el.play().catch((err) => {{
        alert("播放失败: " + err.name + " — " + err.message);
        console.error("play() failed for #" + id, err);
      }});
    }};
    const stop = (id) => {{
      const a = document.getElementById(id);
      a.pause(); a.currentTime = 0;
    }};
    for (const [id, mId, pId] of [["a", "m-a", "p-a"], ["b", "m-b", "p-b"]]) {{
      const el = document.getElementById(id);
      el.addEventListener("error", () => {{
        document.getElementById(mId).style.display = "block";
        document.getElementById(pId).textContent = el.getAttribute("src");
        console.error("media error for #" + id, el.error);
      }});
    }}
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 + keep-alive is what lets browsers do Range-based streaming
    # against this server instead of falling back to "not streamable".
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            body = render_page().encode()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/file_a.wav":
            self._serve_file(_PATH_A)
            return
        if path == "/file_b.wav":
            self._serve_file(_PATH_B)
            return
        self._send(404, b"not found\n", "text/plain")

    def do_HEAD(self):
        path = self.path.split("?", 1)[0]
        target = None
        if path == "/file_a.wav":
            target = _PATH_A
        elif path == "/file_b.wav":
            target = _PATH_B
        if target is None or not os.path.isfile(target):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(os.path.getsize(target)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _serve_file(self, filepath):
        if not os.path.isfile(filepath):
            self._send(404, b"file not generated yet\n", "text/plain")
            return

        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            start, end = self._parse_range(range_header, file_size)
            length = end - start + 1
            with open(filepath, "rb") as f:
                f.seek(start)
                chunk = f.read(length)

            self.send_response(206)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(chunk)
            return

        with open(filepath, "rb") as f:
            body = f.read()
        self._send(200, body, "audio/wav", extra_headers={"Accept-Ranges": "bytes"})

    @staticmethod
    def _parse_range(range_header, file_size):
        # Expected form: "bytes=START-END", either side may be omitted.
        try:
            _, rng = range_header.split("=", 1)
            start_s, end_s = rng.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            if start > end:
                start, end = 0, file_size - 1
        except (ValueError, IndexError):
            start, end = 0, file_size - 1
        return start, end

    def log_message(self, fmt, *args):
        # Keep the console readable; skip favicon noise.
        if args and "favicon" in str(args[0]):
            return
        super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description="Serve a local A/B page for any two WAV files.")
    ap.add_argument("file_a", nargs="?", default=DEFAULT_A, help="first WAV (default: %(default)s)")
    ap.add_argument("file_b", nargs="?", default=DEFAULT_B, help="second WAV (default: %(default)s)")
    ap.add_argument("--label-a", default=None, help="override the displayed label for file_a (default: its basename)")
    ap.add_argument("--label-b", default=None, help="override the displayed label for file_b (default: its basename)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    global _PATH_A, _PATH_B, _LABEL_A, _LABEL_B
    _PATH_A = os.path.abspath(args.file_a)
    _PATH_B = os.path.abspath(args.file_b)
    _LABEL_A = args.label_a or os.path.basename(_PATH_A)
    _LABEL_B = args.label_b or os.path.basename(_PATH_B)

    for label, p in ((_LABEL_A, _PATH_A), (_LABEL_B, _PATH_B)):
        if not os.path.isfile(p):
            print(f"[warn] not found: {label} -> {p}", file=sys.stderr)
            print("       the page will show it as missing until the file exists.", file=sys.stderr)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://{get_local_ip()}:{args.port}/"
    print(f"Serving A/B comparison at {url}")
    print(f"  A ({_LABEL_A}) : {_PATH_A}")
    print(f"  B ({_LABEL_B}) : {_PATH_B}")
    print("  Ctrl-C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
    
