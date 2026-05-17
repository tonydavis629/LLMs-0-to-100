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
const md = fs.readFileSync(mdPath, 'utf8');

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
