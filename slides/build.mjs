#!/usr/bin/env node
// Bundle a reveal.js slide deck into a single self-contained index.bundled.html
// that can be opened directly with file:// (no server fetch of slides.md).
//
// A shared master template drives every module. The repeated HTML/JS/CSS
// lives once under slides/source/ and is composed with each module's content
// at build time:
//
//   slides/source/base.html          <- shared HTML shell with {{placeholders}}
//   slides/source/slides.js          <- shared deck logic (reads MODULE_CONFIG)
//   slides/source/styles.css         <- shared base styles
//
//   <module-dir>/source/slides.md    <- module content (authored from :::fences)
//   <module-dir>/source/config.js    <- sets window.MODULE_CONFIG (title, manim, hooks)
//   <module-dir>/source/styles.css   <- OPTIONAL per-module style overrides
//   <module-dir>/source/head.html    <- OPTIONAL per-module <head> includes
//   <module-dir>/index.bundled.html  <- generated; the only html at the module root
//
// Usage: node slides/build.mjs <module-dir>
//   e.g. node slides/build.mjs slides/module_01_introduction

import fs from 'node:fs';
import path from 'node:path';

const moduleDir = process.argv[2];
if (!moduleDir) {
  console.error('Usage: node slides/build.mjs <module-dir>');
  process.exit(1);
}

// Shared template lives alongside the module dirs (e.g. slides/source/).
const sharedDir = path.join(path.dirname(moduleDir), 'source');
const baseHtml = fs.readFileSync(path.join(sharedDir, 'base.html'), 'utf8');
const sharedJs = fs.readFileSync(path.join(sharedDir, 'slides.js'), 'utf8');
const sharedCss = fs.readFileSync(path.join(sharedDir, 'styles.css'), 'utf8');

// Per-module pieces. slides.md and config.js are required; styles.css and
// head.html are optional overrides.
const readOptional = (p) => (fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '');
const rawMd = fs.readFileSync(path.join(moduleDir, 'source', 'slides.md'), 'utf8');
const moduleConfig = fs.readFileSync(path.join(moduleDir, 'source', 'config.js'), 'utf8');
const moduleCss = readOptional(path.join(moduleDir, 'source', 'styles.css'));
const headExtra = readOptional(path.join(moduleDir, 'source', 'head.html'));
const outPath = path.join(moduleDir, 'index.bundled.html');

// ---------------------------------------------------------------------------
// Component preprocessor.
//
// slides.md is authored entirely from a limited, canonical set of `:::name
// key="val"` ... `:::` fences. Each fence expands to the exact reveal.js
// Markdown/HTML it replaces, with the blank-line structure preserved so
// Markdown inside still parses. Authoring stays free of inline layout
// styles; the rendered deck is unchanged.
//
// Canonical types:
//   :::divider id="" title="" [sub=""]        - section title slide
//   :::columns cols="2"|grid="" [gap=""] [valign=""]
//                                             - N-column grid (+++ splits)
//   :::note [reveal="fragment"] [variant="hint"]
//                                             - callout box
//   :::quiz id="" title=""                    - prompt +++ reveal answer
//   :::step id="" title=""                    - code +++ hint +++ answer
//   :::interactive id="" widget="" [title=""] - JS widget host slide
//   :::figure img="" name="" kicker=""        - notable-figure card
//   :::terminal id="" title="" cmd="" [...]   - terminal-output window
//   :::manim id="" scene="" [title=""]        - manim step-through video
//   :::video id="" src=""                     - autoplay video slide
//
// Attributes are `key="value"` pairs on the opening line. The body is every
// line up to a line that is exactly `:::`. Multi-part types split their body
// on lines that are exactly `+++`.
// ---------------------------------------------------------------------------

function parseAttrs(s) {
  const attrs = {};
  const re = /(\w+)="([^"]*)"/g;
  let m;
  while ((m = re.exec(s)) !== null) attrs[m[1]] = m[2];
  return attrs;
}

// Strip leading/trailing blank lines but keep interior whitespace intact.
function trimBlankLines(text) {
  return text.replace(/^(?:[ \t]*\n)+/, '').replace(/(?:\n[ \t]*)+$/, '');
}

// Split a component body into parts on lines that are exactly `+++`.
// Used by multi-part canonical types (columns, quiz, step).
function splitParts(body) {
  return body.split(/^\+\+\+[ \t]*$/m).map(trimBlankLines);
}

// Build the reveal-markdown per-slide attribute comment. This must be the
// first line of a slide so reveal sets the attributes on the slide <section>
// itself (needed for ids, section-divider class, and data-state).
function slideComment(attrs) {
  const parts = [];
  if (attrs.id) parts.push(`id="${attrs.id}"`);
  if (attrs.cls) parts.push(`class="${attrs.cls}"`);
  if (attrs.state) parts.push(`data-state="${attrs.state}"`);
  return `<!-- .slide: ${parts.join(' ')} -->`;
}

// A standard content slide: optional per-slide attrs comment, an `## h2`
// title, then the markdown body. Blank lines are preserved so Markdown
// inside still parses.
function contentSlide(attrs, title, bodyMd) {
  const out = [];
  if (attrs.id || attrs.cls || attrs.state) out.push(slideComment(attrs), '');
  if (title) out.push(`## ${title}`, '');
  out.push(trimBlankLines(bodyMd));
  return out.join('\n');
}

function need(attrs, keys, name) {
  for (const k of keys) {
    if (attrs[k] === undefined) {
      console.error(`ERROR: :::${name} is missing required attribute "${k}"`);
      process.exit(1);
    }
  }
}

function expandComponent(name, attrs, body) {
  switch (name) {
    case 'figure': {
      need(attrs, ['img', 'name', 'kicker'], name);
      const alt = attrs.alt !== undefined ? attrs.alt : attrs.name;
      return [
        '<div class="figure-card">', '',
        `![${alt}](${attrs.img})`, '',
        '<div class="figure-info">', '',
        `## ${attrs.name} <!-- .element: class="fragment fade-in" -->`, '',
        '<div class="fragment fade-in">', '',
        `### ${attrs.kicker}`, '',
        trimBlankLines(body), '',
        '</div>', '', '</div>', '', '</div>',
      ].join('\n');
    }
    case 'terminal': {
      need(attrs, ['id', 'title', 'cmd'], name);
      const maxw = attrs.maxw ? ` style="max-width: ${attrs.maxw};"` : '';
      // Collapse the <pre> body to a single physical line, encoding real
      // newlines as &#10;. A blank line inside the body would otherwise end
      // the surrounding HTML block in the Markdown parser, which then wraps
      // the following text in <p> tags at the default (larger) paragraph
      // size -- the "PART 2 is bigger than PART 1, overflows the slide" bug.
      // <pre> preserves the &#10; line feeds, so the rendering is identical.
      const preBody = trimBlankLines(body).replace(/\n/g, '&#10;');
      const lines = [
        `<section id="${attrs.id}">`,
        `  <h2>${attrs.title}</h2>`,
        '  <div class="content" style="justify-content: center;">',
        `    <div class="terminal-output"${maxw}>`,
        '      <div class="terminal-bar">',
        '        <span class="terminal-dot red"></span>',
        '        <span class="terminal-dot yellow"></span>',
        '        <span class="terminal-dot green"></span>',
        `        <span class="terminal-title">${attrs.cmd}</span>`,
        '      </div>',
        '      <div class="terminal-body">',
        `        <pre class="terminal-pre">${preBody}</pre>`,
        '      </div>',
        '    </div>',
      ];
      if (attrs.caption !== undefined) {
        lines.push(`    <p class="text-lg terminal-caption">${attrs.caption}</p>`);
      }
      lines.push('  </div>', '</section>');
      return lines.join('\n');
    }
    case 'manim': {
      need(attrs, ['id', 'scene'], name);
      const lines = [`<section id="${attrs.id}">`];
      if (attrs.title !== undefined) lines.push(`  <h2>${attrs.title}</h2>`);
      lines.push(
        '  <div class="video-container">',
        `    <video class="manim-stepper" data-manim-scene="${attrs.scene}" preload="auto"></video>`,
        '  </div>',
        '</section>',
      );
      return lines.join('\n');
    }
    case 'video': {
      need(attrs, ['id', 'src'], name);
      return [
        `<!-- .slide: id="${attrs.id}" -->`, '',
        '<div class="video-container">',
        `  <video data-autoplay src="${attrs.src}"></video>`,
        '</div>',
      ].join('\n');
    }
    // --- Canonical slide types -------------------------------------------
    // The limited set of structural patterns the decks are built from.
    // Authoring stays terse; each fence expands to the exact reveal.js
    // markup it replaces so slides.md carries no inline layout styles.

    case 'divider': {
      // Centered section title slide.
      need(attrs, ['id', 'title'], name);
      const a = { id: attrs.id, cls: 'section-divider', state: 'is-section-divider' };
      const lines = [slideComment(a), '', `# ${attrs.title}`, ''];
      if (attrs.sub !== undefined) lines.push(attrs.sub, '');
      const b = trimBlankLines(body);
      if (b) lines.push(b);
      return lines.join('\n');
    }

    case 'columns':
    case 'cols': {
      // N-column grid. `cols="3"` -> three equal columns; `grid="1.1fr 1fr"`
      // for an explicit template. Body columns are separated by `+++`.
      const tmpl = attrs.grid
        ? attrs.grid
        : `repeat(${attrs.cols || '2'}, 1fr)`;
      const gap = attrs.gap || '30px';
      const align = attrs.valign ? ` align-items: ${attrs.valign};` : '';
      const cols = splitParts(body);
      const out = [
        `<div class="slide-columns" style="grid-template-columns: ${tmpl}; gap: ${gap};${align}">`, '',
      ];
      for (const col of cols) out.push('<div>', '', col, '', '</div>', '');
      out.push('</div>');
      return out.join('\n');
    }

    case 'note': {
      // Callout box. `reveal="fragment"` makes it click-to-reveal;
      // `variant="hint"` uses the lighter hint styling.
      const classes = ['note'];
      if (attrs.variant === 'hint') classes.push('hint-box');
      if (attrs.reveal === 'fragment') classes.unshift('fragment');
      return [
        `<div class="${classes.join(' ')}">`, '',
        trimBlankLines(body), '',
        '</div>',
      ].join('\n');
    }

    case 'quiz': {
      // Quiz slide: prompt, then `+++`, then a click-to-reveal answer.
      need(attrs, ['id', 'title'], name);
      const [prompt = '', answer = ''] = splitParts(body);
      const md = [
        '<div class="quiz-prompt">', '',
        prompt, '',
        '</div>', '',
        '<div class="fragment note quiz-answer">', '',
        answer, '',
        '</div>',
      ].join('\n');
      return contentSlide({ id: attrs.id }, attrs.title, md);
    }

    case 'step': {
      // Exercise walkthrough slide: code, then `+++` hint,
      // then `+++` answer. Hint and answer reveal on click.
      need(attrs, ['id', 'title'], name);
      const [code = '', hint = '', answer = ''] = splitParts(body);
      const md = [
        code, '',
        '<div class="fragment hint-box">', '',
        hint, '',
        '</div>', '',
        '<div class="fragment note">', '',
        answer, '',
        '</div>',
      ].join('\n');
      return contentSlide({ id: attrs.id }, attrs.title, md);
    }

    case 'interactive': {
      // Host slide for a vanilla-JS widget hydrated from source/index.html
      // by its `widget` key. No video; the widget owns the slide.
      need(attrs, ['id', 'widget'], name);
      const modeAttr = attrs.mode !== undefined ? ` data-mode="${attrs.mode}"` : '';
      const lines = [`<section id="${attrs.id}">`];
      if (attrs.title !== undefined) lines.push(`  <h2>${attrs.title}</h2>`);
      lines.push(
        `  <div class="interactive-host" data-widget="${attrs.widget}"${modeAttr}></div>`,
        '</section>',
      );
      return lines.join('\n');
    }

    default:
      console.error(`ERROR: unknown component :::${name} in slides.md`);
      process.exit(1);
  }
}

function expandComponents(src) {
  const lines = src.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const open = lines[i].match(/^:::([a-z]+)[ \t]*(.*)$/);
    if (!open) { out.push(lines[i]); continue; }
    const name = open[1];
    const attrs = parseAttrs(open[2]);
    const bodyLines = [];
    let j = i + 1;
    for (; j < lines.length; j++) {
      if (/^:::[ \t]*$/.test(lines[j])) break;
      bodyLines.push(lines[j]);
    }
    if (j >= lines.length) {
      console.error(`ERROR: unterminated :::${name} fence in slides.md (no closing ":::")`);
      process.exit(1);
    }
    out.push(expandComponent(name, attrs, bodyLines.join('\n')));
    i = j; // skip past the closing :::
  }
  return out.join('\n');
}

const md = expandComponents(rawMd);

if (md.includes('</textarea>')) {
  console.error('ERROR: slides.md contains a literal </textarea>, which would terminate the inline template early.');
  process.exit(1);
}

// Compose base.html by substituting each {{placeholder}}. The replacement is
// passed as a FUNCTION so String.replace does not interpret `$` sequences in
// the content (KaTeX `$...$`, JS regex, etc.) as replacement patterns.
function fill(tpl, token, value) {
  if (!tpl.includes(token)) {
    console.error(`ERROR: base.html is missing the ${token} placeholder`);
    process.exit(1);
  }
  return tpl.replace(token, () => value);
}

let bundled = baseHtml;
bundled = fill(bundled, '{{HEAD_EXTRA}}', headExtra);
bundled = fill(bundled, '{{STYLES}}', sharedCss + '\n' + moduleCss);
bundled = fill(bundled, '{{MODULE_CONFIG}}', moduleConfig);
bundled = fill(bundled, '{{SHARED_JS}}', sharedJs);
bundled = fill(bundled, '{{SLIDES_MD}}', md);

fs.writeFileSync(outPath, bundled);

console.log(`wrote ${outPath} (${(bundled.length / 1024).toFixed(1)} KB)`);
