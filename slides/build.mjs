#!/usr/bin/env node
// Bundle a reveal.js slide deck into a single self-contained index.bundled.html
// that can be opened directly with file:// (no server fetch of slides.md).
//
// Layout (per module):
//   <module-dir>/source/index.html   <- editable source (inline JS)
//   <module-dir>/source/slides.md    <- editable content
//   <module-dir>/source/styles.css   <- editable styles
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

const indexPath = path.join(moduleDir, 'source', 'index.html');
const mdPath = path.join(moduleDir, 'source', 'slides.md');
const outPath = path.join(moduleDir, 'index.bundled.html');

const html = fs.readFileSync(indexPath, 'utf8');
const rawMd = fs.readFileSync(mdPath, 'utf8');

// ---------------------------------------------------------------------------
// Component preprocessor.
//
// slides.md may use terse `:::name key="val"` ... `:::` fences for the bulky,
// repeated slide patterns (notable-figure cards, terminal-output windows,
// manim/video slides, multi-column grids). Each fence expands to the exact
// reveal.js-Markdown HTML it replaces, with the blank-line structure preserved
// so Markdown inside still parses. Authoring stays readable; the rendered deck
// is unchanged.
//
// Fence syntax:
//   :::figure img="..." name="..." kicker="..."
//   <markdown body>
//   :::
// Attributes are `key="value"` pairs on the opening line. The body is every
// line up to a line that is exactly `:::`. `:::cols` splits its body into
// columns on lines that are exactly `+++`.
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
        `        <pre class="terminal-pre">${trimBlankLines(body)}</pre>`,
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
    case 'cols': {
      const grid = attrs.grid || '1fr 1fr';
      const gap = attrs.gap || '30px';
      const columns = body.split(/^\+\+\+[ \t]*$/m).map(trimBlankLines);
      const out = [`<div style="display: grid; grid-template-columns: ${grid}; gap: ${gap};">`, ''];
      for (const col of columns) {
        out.push('<div>', '', col, '', '</div>', '');
      }
      out.push('</div>');
      return out.join('\n');
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

const sectionRegex = /<section\s+data-markdown="slides\.md"([\s\S]*?)>\s*<\/section>/;
if (!sectionRegex.test(html)) {
  console.error('ERROR: could not find <section data-markdown="slides.md" ...></section> in source/index.html');
  process.exit(1);
}

const replacement = `<section data-markdown$1>
        <textarea data-template>
${md}
        </textarea>
      </section>`;

let bundled = html.replace(sectionRegex, replacement);

// The bundle sits at the module root but styles.css lives in source/.
// Repoint the stylesheet link so the root bundle resolves it.
if (!bundled.includes('href="styles.css"')) {
  console.error('ERROR: could not find <link ... href="styles.css"> to repoint to source/styles.css');
  process.exit(1);
}
bundled = bundled.replace('href="styles.css"', 'href="source/styles.css"');

fs.writeFileSync(outPath, bundled);

console.log(`wrote ${outPath} (${(bundled.length / 1024).toFixed(1)} KB)`);
