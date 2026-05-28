#!/usr/bin/env python3
"""Bundle one reveal.js slide module into index.bundled.html."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_attrs(text: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', text))


def trim_blank_lines(text: str) -> str:
    return re.sub(r"(?:\n[ \t]*)+$", "", re.sub(r"^(?:[ \t]*\n)+", "", text))


def split_parts(body: str) -> list[str]:
    return [trim_blank_lines(part) for part in re.split(r"^\+\+\+[ \t]*$", body, flags=re.MULTILINE)]


def slide_comment(attrs: dict[str, str]) -> str:
    parts = []
    if attrs.get("id"):
        parts.append(f'id="{attrs["id"]}"')
    if attrs.get("cls"):
        parts.append(f'class="{attrs["cls"]}"')
    if attrs.get("state"):
        parts.append(f'data-state="{attrs["state"]}"')
    return f"<!-- .slide: {' '.join(parts)} -->"


def content_slide(attrs: dict[str, str], title: str | None, body_md: str) -> str:
    out = []
    if attrs.get("id") or attrs.get("cls") or attrs.get("state"):
        out.extend([slide_comment(attrs), ""])
    if title:
        out.extend([f"## {title}", ""])
    out.append(trim_blank_lines(body_md))
    return "\n".join(out)


def need(attrs: dict[str, str], keys: list[str], name: str) -> None:
    for key in keys:
        if key not in attrs:
            raise ValueError(f'ERROR: :::{name} is missing required attribute "{key}"')


def expand_component(name: str, attrs: dict[str, str], body: str) -> str:
    if name == "figure":
        need(attrs, ["img", "name", "kicker"], name)
        alt = attrs.get("alt", attrs["name"])
        return "\n".join([
            '<div class="figure-card">', "",
            f"![{alt}]({attrs['img']})", "",
            '<div class="figure-info">', "",
            f'## {attrs["name"]} <!-- .element: class="fragment fade-in" -->', "",
            '<div class="fragment fade-in">', "",
            f"### {attrs['kicker']}", "",
            trim_blank_lines(body), "",
            "</div>", "", "</div>", "", "</div>",
        ])

    if name == "terminal":
        need(attrs, ["id", "title", "cmd"], name)
        maxw = f' style="max-width: {attrs["maxw"]};"' if attrs.get("maxw") else ""
        pre_body = trim_blank_lines(body).replace("\n", "&#10;")
        lines = [
            f'<section id="{attrs["id"]}">',
            f'  <h2>{attrs["title"]}</h2>',
            '  <div class="content" style="justify-content: center;">',
            f'    <div class="terminal-output"{maxw}>',
            '      <div class="terminal-bar">',
            '        <span class="terminal-dot red"></span>',
            '        <span class="terminal-dot yellow"></span>',
            '        <span class="terminal-dot green"></span>',
            f'        <span class="terminal-title">{attrs["cmd"]}</span>',
            "      </div>",
            '      <div class="terminal-body">',
            f'        <pre class="terminal-pre">{pre_body}</pre>',
            "      </div>",
            "    </div>",
        ]
        if "caption" in attrs:
            lines.append(f'    <p class="text-lg terminal-caption">{attrs["caption"]}</p>')
        lines.extend(["  </div>", "</section>"])
        return "\n".join(lines)

    if name == "manim":
        need(attrs, ["id", "scene"], name)
        lines = [f'<section id="{attrs["id"]}">']
        if "title" in attrs:
            lines.append(f'  <h2>{attrs["title"]}</h2>')
        lines.extend([
            '  <div class="video-container">',
            f'    <video class="manim-stepper" data-manim-scene="{attrs["scene"]}" preload="auto"></video>',
            "  </div>",
            "</section>",
        ])
        return "\n".join(lines)

    if name == "video":
        need(attrs, ["id", "src"], name)
        return "\n".join([
            f'<!-- .slide: id="{attrs["id"]}" -->', "",
            '<div class="video-container">',
            f'  <video data-autoplay src="{attrs["src"]}"></video>',
            "</div>",
        ])

    if name == "divider":
        need(attrs, ["id", "title"], name)
        lines = [
            slide_comment({"id": attrs["id"], "cls": "section-divider", "state": "is-section-divider"}),
            "",
            f'# {attrs["title"]}',
            "",
        ]
        if "sub" in attrs:
            lines.extend([attrs["sub"], ""])
        trimmed = trim_blank_lines(body)
        if trimmed:
            lines.append(trimmed)
        return "\n".join(lines)

    if name in {"columns", "cols"}:
        template = attrs.get("grid", f'repeat({attrs.get("cols", "2")}, 1fr)')
        gap = attrs.get("gap", "30px")
        align = f' align-items: {attrs["valign"]};' if attrs.get("valign") else ""
        out = [
            f'<div class="slide-columns" style="grid-template-columns: {template}; gap: {gap};{align}">',
            "",
        ]
        for col in split_parts(body):
            out.extend(["<div>", "", col, "", "</div>", ""])
        out.append("</div>")
        return "\n".join(out)

    if name == "note":
        classes = ["note"]
        if attrs.get("variant") == "hint":
            classes.append("hint-box")
        if attrs.get("reveal") == "fragment":
            classes.insert(0, "fragment")
        return "\n".join([
            f'<div class="{" ".join(classes)}">', "",
            trim_blank_lines(body), "",
            "</div>",
        ])

    if name == "quiz":
        need(attrs, ["id", "title"], name)
        parts = split_parts(body)
        prompt = parts[0] if len(parts) > 0 else ""
        answer = parts[1] if len(parts) > 1 else ""
        md = "\n".join([
            '<div class="quiz-prompt">', "",
            prompt, "",
            "</div>", "",
            '<div class="fragment note quiz-answer">', "",
            answer, "",
            "</div>",
        ])
        return content_slide({"id": attrs["id"]}, attrs["title"], md)

    if name == "step":
        need(attrs, ["id", "title"], name)
        parts = split_parts(body)
        code = parts[0] if len(parts) > 0 else ""
        hint = parts[1] if len(parts) > 1 else ""
        answer = parts[2] if len(parts) > 2 else ""
        md = "\n".join([
            code, "",
            '<div class="fragment hint-box">', "",
            hint, "",
            "</div>", "",
            '<div class="fragment note">', "",
            answer, "",
            "</div>",
        ])
        return content_slide({"id": attrs["id"]}, attrs["title"], md)

    if name == "interactive":
        need(attrs, ["id", "widget"], name)
        mode_attr = f' data-mode="{attrs["mode"]}"' if "mode" in attrs else ""
        lines = [f'<section id="{attrs["id"]}">']
        if "title" in attrs:
            lines.append(f'  <h2>{attrs["title"]}</h2>')
        lines.extend([
            f'  <div class="interactive-host" data-widget="{attrs["widget"]}"{mode_attr}></div>',
            "</section>",
        ])
        return "\n".join(lines)

    raise ValueError(f"ERROR: unknown component :::{name} in slides.md")


def expand_components(src: str) -> str:
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        match = re.match(r"^:::([a-z]+)[ \t]*(.*)$", lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue

        name = match.group(1)
        attrs = parse_attrs(match.group(2))
        body_lines = []
        j = i + 1
        while j < len(lines) and not re.match(r"^:::[ \t]*$", lines[j]):
            body_lines.append(lines[j])
            j += 1
        if j >= len(lines):
            raise ValueError(f'ERROR: unterminated :::{name} fence in slides.md (no closing ":::")')
        out.append(expand_component(name, attrs, "\n".join(body_lines)))
        i = j + 1
    return "\n".join(out)


def fill(template: str, token: str, value: str) -> str:
    if token not in template:
        raise ValueError(f"ERROR: base.html is missing the {token} placeholder")
    return template.replace(token, value, 1)


def build_module(module_dir: Path) -> Path:
    shared_dir = module_dir.parent / "source"
    base_html = (shared_dir / "base.html").read_text(encoding="utf-8")
    shared_js = (shared_dir / "slides.js").read_text(encoding="utf-8")
    shared_css = (shared_dir / "styles.css").read_text(encoding="utf-8")

    source_dir = module_dir / "source"
    raw_md = (source_dir / "slides.md").read_text(encoding="utf-8")
    module_config = (source_dir / "config.js").read_text(encoding="utf-8")
    module_css = read_optional(source_dir / "styles.css")
    head_extra = read_optional(source_dir / "head.html")

    md = expand_components(raw_md)
    if "</textarea>" in md:
        raise ValueError("ERROR: slides.md contains a literal </textarea>, which would terminate the inline template early.")

    bundled = base_html
    bundled = fill(bundled, "{{HEAD_EXTRA}}", head_extra)
    bundled = fill(bundled, "{{STYLES}}", f"{shared_css}\n{module_css}")
    bundled = fill(bundled, "{{MODULE_CONFIG}}", module_config)
    bundled = fill(bundled, "{{SHARED_JS}}", shared_js)
    bundled = fill(bundled, "{{SLIDES_MD}}", md)

    out_path = module_dir / "index.bundled.html"
    out_path.write_text(bundled, encoding="utf-8")
    print(f"wrote {out_path} ({len(bundled.encode('utf-8')) / 1024:.1f} KB)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_dir", type=Path, help="Slide module directory, for example slides/module_01_introduction")
    args = parser.parse_args()

    try:
        build_module(args.module_dir)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
