:::divider id="divider-cyber" title="Cybersecurity"
:::

---

<!-- .slide: id="cyber-why" -->

## Why Agents Are Good at Finding Exploits

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 960 300" width="100%" style="max-height: 440px;">
  <defs>
    <marker id="cy-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
    <marker id="cy-arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a9eff"/>
    </marker>
  </defs>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="40" y="110" width="180" height="64" rx="8"/>
    <rect x="270" y="110" width="180" height="64" rx="8"/>
    <rect x="500" y="110" width="180" height="64" rx="8"/>
    <path d="M 800 92 L 880 142 L 800 192 L 720 142 Z" stroke="#f5a623"/>
    <rect x="720" y="230" width="160" height="50" rx="8" stroke="#50c878"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600" fill="#4a9eff">
    <text x="130" y="147">Read the code</text>
    <text x="360" y="147">Guess a flaw</text>
    <text x="590" y="147">Try an input</text>
    <text x="800" y="147" fill="#f5a623">Did it work?</text>
    <text x="800" y="261" fill="#50c878">Exploit</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#cy-arrow)">
    <line x1="220" y1="142" x2="266" y2="142"/>
    <line x1="450" y1="142" x2="496" y2="142"/>
    <line x1="680" y1="142" x2="716" y2="142"/>
    <line x1="800" y1="192" x2="800" y2="226"/>
  </g>
  <path d="M 800 90 L 800 58 Q 800 46 788 46 L 372 46 Q 360 46 360 58 L 360 106" stroke="#4a9eff" stroke-width="1.5" fill="none" marker-end="url(#cy-arrow-blue)"/>
  <text x="580" y="36" text-anchor="middle" font-size="13" fill="#8892a4">no: guess again (cheap, so try thousands)</text>
  <text x="812" y="214" font-size="13" fill="#8892a4">yes</text>
  <text x="900" y="134" font-size="12" fill="#8892a4">crash?</text>
  <text x="900" y="152" font-size="12" fill="#8892a4">flag?</text>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.04)">
    <rect x="40" y="230" width="200" height="50" rx="25"/>
    <rect x="255" y="230" width="200" height="50" rx="25"/>
    <rect x="470" y="230" width="200" height="50" rx="25"/>
  </g>
  <g text-anchor="middle">
    <text x="140" y="251" font-size="13.5" font-weight="600" fill="#e8eaf0">Seen in training</text>
    <text x="140" y="269" font-size="12" fill="#8892a4">code, patches, exploit write-ups</text>
    <text x="355" y="251" font-size="13.5" font-weight="600" fill="#e8eaf0">Text-only tools</text>
    <text x="355" y="269" font-size="12" fill="#8892a4">debuggers, scanners, fuzzers</text>
    <text x="570" y="251" font-size="13.5" font-weight="600" fill="#e8eaf0">Runs in parallel</text>
    <text x="570" y="269" font-size="12" fill="#8892a4">many agents, no fatigue</text>
  </g>
</svg>
</div>

The check at the end is a verifier, so Module 7's RL with verifiable rewards can train this skill directly. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="cyber-trace" -->

## An Exploit Search, Step by Step

:::columns cols="2" gap="30px"
**The code**

```python
@app.get("/manual")
def manual(name):
    # user input is joined to the path unchecked
    path = "/srv/manuals/" + name
    return open(path).read()
```
+++
**The agent's trace**

```text
Thought: name goes straight into a path.
         Try ../ to leave the folder.
Action:  GET /manual?name=../../etc/passwd
Observation: root:x:0:0:root:/root:/bin/bash
Thought: Any file is readable. Look for
         the app's config.
Action:  GET /manual?name=../app/config.py
Observation: DB_PASSWORD = "..."
```
:::

A file read leads to a database password. Small bugs chain into serious ones. (Illustrative example.) <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="cyber-dual" -->

## The Same Skill on Both Sides

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 960 290" width="100%" style="max-height: 430px;">
  <defs>
    <marker id="cy2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
  </defs>
  <g stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="370" y="14" width="220" height="56" rx="8" stroke="#f5a623"/>
    <rect x="80" y="96" width="250" height="56" rx="8" stroke="#50c878"/>
    <rect x="630" y="96" width="250" height="56" rx="8" stroke="#e06c75"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600">
    <text x="480" y="48" fill="#f5a623">An agent finds a flaw</text>
    <text x="205" y="122" fill="#50c878">Defender</text>
    <text x="755" y="122" fill="#e06c75">Attacker</text>
  </g>
  <g text-anchor="middle" font-size="12.5" fill="#8892a4">
    <text x="205" y="142">patches it before release</text>
    <text x="755" y="142">exploits it in production</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#cy2-arrow)">
    <line x1="420" y1="70" x2="300" y2="94"/>
    <line x1="540" y1="70" x2="660" y2="94"/>
  </g>
  <g font-size="15" fill="#e8eaf0" text-anchor="end">
    <text x="280" y="212">Attacker needs</text>
    <text x="280" y="262">Defender must fix</text>
  </g>
  <g stroke="#2a3450" stroke-width="1.5">
    <rect x="300" y="190" width="30" height="30" rx="4" fill="#e06c75"/>
    <rect x="338" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="376" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="414" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="452" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="490" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="528" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="566" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="604" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="642" y="190" width="30" height="30" rx="4" fill="rgba(136,146,164,0.15)"/>
    <rect x="300" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="338" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="376" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="414" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="452" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="490" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="528" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="566" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="604" y="240" width="30" height="30" rx="4" fill="#50c878"/>
    <rect x="642" y="240" width="30" height="30" rx="4" fill="#50c878"/>
  </g>
  <g font-size="15" font-weight="600">
    <text x="690" y="212" fill="#e06c75">1 flaw</text>
    <text x="690" y="262" fill="#50c878">all 10</text>
  </g>
</svg>
</div>

Agents make finding flaws cheap for both sides. Defenders gain the most by scanning their own code first. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="cyber-contain" -->

## Testing a Model That Attacks

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 960 290" width="100%" style="max-height: 430px;">
  <defs>
    <marker id="cy3-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
    <marker id="cy3-arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#e06c75"/>
    </marker>
  </defs>
  <rect x="30" y="30" width="560" height="240" rx="12" fill="rgba(74,158,255,0.03)" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="50" y="58" font-size="14" font-weight="600" fill="#8892a4">Sandbox: safeguards switched off</text>
  <text x="700" y="58" font-size="14" font-weight="600" fill="#8892a4">The real internet</text>
  <g stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="70" y="100" width="190" height="64" rx="8" stroke="#f5a623"/>
    <rect x="370" y="100" width="190" height="64" rx="8" stroke="#2a3450"/>
    <rect x="690" y="180" width="230" height="64" rx="8" stroke="#e06c75"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600">
    <text x="165" y="128" fill="#f5a623">Model</text>
    <text x="465" y="128" fill="#4a9eff">Practice target</text>
    <text x="805" y="208" fill="#e06c75">Real company</text>
  </g>
  <g text-anchor="middle" font-size="12.5" fill="#8892a4">
    <text x="165" y="150">task: get the flag</text>
    <text x="465" y="150">the intended route</text>
    <text x="805" y="230">attacked, no one directed it</text>
  </g>
  <line x1="260" y1="132" x2="366" y2="132" stroke="#8892a4" stroke-width="1.5" marker-end="url(#cy3-arrow)"/>
  <path d="M 165 166 L 165 200 Q 165 212 177 212 L 686 212" stroke="#e06c75" stroke-width="1.8" fill="none" marker-end="url(#cy3-arrow-red)"/>
  <rect x="582" y="196" width="16" height="32" fill="#0a0e1a"/>
  <text x="400" y="234" text-anchor="middle" font-size="13" fill="#e06c75">a network leak</text>
</svg>
</div>

The task never says which routes are allowed, so isolation is the only control. Models at two labs reached real systems this way in 2026. <!-- .element: class="text-lg" -->
