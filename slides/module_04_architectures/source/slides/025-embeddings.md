:::divider id="divider-embeddings" title="Embeddings" sub="From token IDs to vectors"
:::

---

<!-- .slide: id="embedding-layer" -->

## Step 1: Words to Vectors

<div class="decoder-flow active-embedding">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

<div class="embedding-visual">
  <div class="word-card">The capital of France</div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="token-strip"><span>The</span><span>&nbsp;capital</span><span>&nbsp;of</span><span>&nbsp;France</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="id-strip"><span>464</span><span>3139</span><span>286</span><span>4881</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="embedding-table">
    <div class="et-title">embedding matrix<br><span>one learned vector per row</span></div>
    <div class="et-row"><span class="et-id">row 464</span><span class="et-vec">[ 0.14 -0.22 0.05 0.61 &#8230; ]</span></div>
    <div class="et-row"><span class="et-id">row 3139</span><span class="et-vec">[ -0.31 0.47 0.18 -0.09 &#8230; ]</span></div>
    <div class="et-row"><span class="et-id">row 286</span><span class="et-vec">[ 0.02 0.33 -0.27 0.40 &#8230; ]</span></div>
    <div class="et-row accent"><span class="et-id">row 4881</span><span class="et-vec">[ 0.55 -0.12 0.29 -0.63 &#8230; ]</span></div>
  </div>
</div>

Tokenize, then look up each ID in the embedding matrix. From here on, the transformer sees vectors, not text.

:::note
Row $k$ of the **embedding matrix** is the learned vector for token $k$. Lookup is pure indexing: ID 4881 ("France") selects row 4881, and that row **is** the token's vector.
:::

---

:::manim id="embedding-anim" scene="embedding-lookup"
:::

---

<!-- .slide: id="embedding-matrix" -->

## Where the Embedding Matrix Comes From

<div class="emb-svg">
<svg viewBox="0 0 1030 330" role="img" aria-label="Step 0: every entry of the embedding matrix is drawn from a Gaussian with mean 0 and standard deviation 0.02. Training by gradient descent then changes the entries into learned values.">
<defs><marker id="emb" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#8892a4"></path></marker></defs>
<text x="130" y="32" text-anchor="middle" font-size="17" fill="#e8eaf0">1. Initialize</text>
<text x="130" y="54" text-anchor="middle" font-size="13" fill="#8892a4">Gaussian: mean 0, std 0.02</text>
<polygon points="10,210 10.0,208.7 12.0,208.5 14.0,208.2 16.0,207.9 18.0,207.6 20.0,207.3 22.0,206.9 24.0,206.4 26.0,205.9 28.0,205.4 30.0,204.7 32.0,204.0 34.0,203.3 36.0,202.4 38.0,201.5 40.0,200.5 42.0,199.3 44.0,198.1 46.0,196.8 48.0,195.3 50.0,193.8 52.0,192.1 54.0,190.3 56.0,188.3 58.0,186.3 60.0,184.0 62.0,181.7 64.0,179.2 66.0,176.6 68.0,173.9 70.0,171.0 72.0,168.1 74.0,165.0 76.0,161.8 78.0,158.5 80.0,155.1 82.0,151.6 84.0,148.1 86.0,144.5 88.0,140.9 90.0,137.2 92.0,133.6 94.0,130.0 96.0,126.4 98.0,122.9 100.0,119.4 102.0,116.1 104.0,112.9 106.0,109.8 108.0,106.8 110.0,104.1 112.0,101.6 114.0,99.2 116.0,97.1 118.0,95.3 120.0,93.7 122.0,92.4 124.0,91.3 126.0,90.6 128.0,90.1 130.0,90.0 132.0,90.1 134.0,90.6 136.0,91.3 138.0,92.4 140.0,93.7 142.0,95.3 144.0,97.1 146.0,99.2 148.0,101.6 150.0,104.1 152.0,106.8 154.0,109.8 156.0,112.9 158.0,116.1 160.0,119.4 162.0,122.9 164.0,126.4 166.0,130.0 168.0,133.6 170.0,137.2 172.0,140.9 174.0,144.5 176.0,148.1 178.0,151.6 180.0,155.1 182.0,158.5 184.0,161.8 186.0,165.0 188.0,168.1 190.0,171.0 192.0,173.9 194.0,176.6 196.0,179.2 198.0,181.7 200.0,184.0 202.0,186.3 204.0,188.3 206.0,190.3 208.0,192.1 210.0,193.8 212.0,195.3 214.0,196.8 216.0,198.1 218.0,199.3 220.0,200.5 222.0,201.5 224.0,202.4 226.0,203.3 228.0,204.0 230.0,204.7 232.0,205.4 234.0,205.9 236.0,206.4 238.0,206.9 240.0,207.3 242.0,207.6 244.0,207.9 246.0,208.2 248.0,208.5 250.0,208.7 250,210" fill="rgba(74,158,255,0.15)" stroke="none"></polygon>
<polyline points="10.0,208.7 12.0,208.5 14.0,208.2 16.0,207.9 18.0,207.6 20.0,207.3 22.0,206.9 24.0,206.4 26.0,205.9 28.0,205.4 30.0,204.7 32.0,204.0 34.0,203.3 36.0,202.4 38.0,201.5 40.0,200.5 42.0,199.3 44.0,198.1 46.0,196.8 48.0,195.3 50.0,193.8 52.0,192.1 54.0,190.3 56.0,188.3 58.0,186.3 60.0,184.0 62.0,181.7 64.0,179.2 66.0,176.6 68.0,173.9 70.0,171.0 72.0,168.1 74.0,165.0 76.0,161.8 78.0,158.5 80.0,155.1 82.0,151.6 84.0,148.1 86.0,144.5 88.0,140.9 90.0,137.2 92.0,133.6 94.0,130.0 96.0,126.4 98.0,122.9 100.0,119.4 102.0,116.1 104.0,112.9 106.0,109.8 108.0,106.8 110.0,104.1 112.0,101.6 114.0,99.2 116.0,97.1 118.0,95.3 120.0,93.7 122.0,92.4 124.0,91.3 126.0,90.6 128.0,90.1 130.0,90.0 132.0,90.1 134.0,90.6 136.0,91.3 138.0,92.4 140.0,93.7 142.0,95.3 144.0,97.1 146.0,99.2 148.0,101.6 150.0,104.1 152.0,106.8 154.0,109.8 156.0,112.9 158.0,116.1 160.0,119.4 162.0,122.9 164.0,126.4 166.0,130.0 168.0,133.6 170.0,137.2 172.0,140.9 174.0,144.5 176.0,148.1 178.0,151.6 180.0,155.1 182.0,158.5 184.0,161.8 186.0,165.0 188.0,168.1 190.0,171.0 192.0,173.9 194.0,176.6 196.0,179.2 198.0,181.7 200.0,184.0 202.0,186.3 204.0,188.3 206.0,190.3 208.0,192.1 210.0,193.8 212.0,195.3 214.0,196.8 216.0,198.1 218.0,199.3 220.0,200.5 222.0,201.5 224.0,202.4 226.0,203.3 228.0,204.0 230.0,204.7 232.0,205.4 234.0,205.9 236.0,206.4 238.0,206.9 240.0,207.3 242.0,207.6 244.0,207.9 246.0,208.2 248.0,208.5 250.0,208.7" fill="none" stroke="#4a9eff" stroke-width="2"></polyline>
<line x1="10" y1="210" x2="250" y2="210" stroke="#8892a4" stroke-width="1.3"></line>
<line x1="50.0" y1="210" x2="50.0" y2="215" stroke="#8892a4" stroke-width="1.2"></line>
<text x="50.0" y="230" text-anchor="middle" font-size="12" fill="#8892a4">&#8722;0.04</text>
<line x1="90.0" y1="210" x2="90.0" y2="215" stroke="#8892a4" stroke-width="1.2"></line>
<text x="90.0" y="230" text-anchor="middle" font-size="12" fill="#8892a4">&#8722;0.02</text>
<line x1="130.0" y1="210" x2="130.0" y2="215" stroke="#8892a4" stroke-width="1.2"></line>
<text x="130.0" y="230" text-anchor="middle" font-size="12" fill="#8892a4">0</text>
<line x1="170.0" y1="210" x2="170.0" y2="215" stroke="#8892a4" stroke-width="1.2"></line>
<text x="170.0" y="230" text-anchor="middle" font-size="12" fill="#8892a4">0.02</text>
<line x1="210.0" y1="210" x2="210.0" y2="215" stroke="#8892a4" stroke-width="1.2"></line>
<text x="210.0" y="230" text-anchor="middle" font-size="12" fill="#8892a4">0.04</text>
<text x="130" y="252" text-anchor="middle" font-size="13" fill="#8892a4">value of one entry</text>
<text x="130" y="276" text-anchor="middle" font-size="13" fill="#e8eaf0">most draws land between &#8722;0.04 and 0.04</text>
<line x1="258" y1="150" x2="318" y2="150" stroke="#8892a4" stroke-width="2" marker-end="url(#emb)"></line>
<text x="288" y="138" text-anchor="middle" font-size="13" fill="#8892a4">fill</text>
<text x="288" y="170" text-anchor="middle" font-size="13" fill="#8892a4">every entry</text>
<text x="455" y="72" text-anchor="middle" font-size="17" fill="#e8eaf0">2. E at step 0</text>
<text x="455" y="94" text-anchor="middle" font-size="13" fill="#8892a4">small random numbers, no meaning yet</text>
<text x="377" y="132" text-anchor="end" font-size="14" fill="#e8eaf0">The</text>
<rect x="385" y="110" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="417.0" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.023</text>
<rect x="453" y="110" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="485.0" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.023</text>
<rect x="521" y="110" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="553.0" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.005</text>
<text x="599" y="132" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="377" y="171" text-anchor="end" font-size="14" fill="#e8eaf0">capital</text>
<rect x="385" y="149" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="417.0" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">0.017</text>
<rect x="453" y="149" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="485.0" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">0.014</text>
<rect x="521" y="149" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="553.0" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.006</text>
<text x="599" y="171" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="377" y="210" text-anchor="end" font-size="14" fill="#e8eaf0">of</text>
<rect x="385" y="188" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="417.0" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">0.006</text>
<rect x="453" y="188" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="485.0" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.025</text>
<rect x="521" y="188" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="553.0" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">0.007</text>
<text x="599" y="210" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="377" y="249" text-anchor="end" font-size="14" fill="#e8eaf0">France</text>
<rect x="385" y="227" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="417.0" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">0.002</text>
<rect x="453" y="227" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="485.0" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">0.025</text>
<rect x="521" y="227" width="64" height="34" rx="4" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.2"></rect>
<text x="553.0" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">0.022</text>
<text x="599" y="249" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="377" y="284" text-anchor="end" font-size="16" fill="#8892a4">&#8942;</text>
<text x="485" y="284" text-anchor="middle" font-size="16" fill="#8892a4">&#8942;</text>
<line x1="640" y1="150" x2="714" y2="150" stroke="#8892a4" stroke-width="2" marker-end="url(#emb)"></line>
<text x="677" y="138" text-anchor="middle" font-size="13" fill="#8892a4">train</text>
<text x="677" y="170" text-anchor="middle" font-size="13" fill="#8892a4">(backprop)</text>
<text x="855" y="72" text-anchor="middle" font-size="17" fill="#e8eaf0">3. E after training</text>
<text x="855" y="94" text-anchor="middle" font-size="13" fill="#8892a4">real GPT-2 small values</text>
<text x="777" y="132" text-anchor="end" font-size="14" fill="#e8eaf0">The</text>
<rect x="785" y="110" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="817" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.069</text>
<rect x="853" y="110" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="885" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.020</text>
<rect x="921" y="110" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="953" y="132" text-anchor="middle" font-size="13" fill="#e8eaf0">0.064</text>
<text x="999" y="132" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="777" y="171" text-anchor="end" font-size="14" fill="#e8eaf0">capital</text>
<rect x="785" y="149" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="817" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">0.057</text>
<rect x="853" y="149" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="885" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.007</text>
<rect x="921" y="149" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="953" y="171" text-anchor="middle" font-size="13" fill="#e8eaf0">0.091</text>
<text x="999" y="171" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="777" y="210" text-anchor="end" font-size="14" fill="#e8eaf0">of</text>
<rect x="785" y="188" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="817" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.057</text>
<rect x="853" y="188" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="885" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">0.018</text>
<rect x="921" y="188" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="953" y="210" text-anchor="middle" font-size="13" fill="#e8eaf0">0.033</text>
<text x="999" y="210" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="777" y="249" text-anchor="end" font-size="14" fill="#e8eaf0">France</text>
<rect x="785" y="227" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="817" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">&#8722;0.069</text>
<rect x="853" y="227" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="885" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">0.049</text>
<rect x="921" y="227" width="64" height="34" rx="4" fill="rgba(63,185,80,0.10)" stroke="#3fb950" stroke-width="1.2"></rect>
<text x="953" y="249" text-anchor="middle" font-size="13" fill="#e8eaf0">0.162</text>
<text x="999" y="249" text-anchor="middle" font-size="15" fill="#8892a4">&#8943;</text>
<text x="777" y="284" text-anchor="end" font-size="16" fill="#8892a4">&#8942;</text>
<text x="885" y="284" text-anchor="middle" font-size="16" fill="#8892a4">&#8942;</text>
</svg>
</div>

$E$ has one row per token and one column per dimension: $50{,}257 \times 768 \approx$ 38.6M numbers in GPT-2 small. It starts random and is learned like every other weight.

---

<!-- .slide: id="embedding-coordinates" -->

## Vectors Are Coordinates

A vector of $d$ numbers is a point in $d$-dimensional space.

<div class="emb-svg">
<svg viewBox="40 0 960 325" role="img" aria-label="Left: cat and dog vectors point in nearly the same direction with a small angle between them, while Paris points elsewhere. Right: the man to woman arrow is parallel to the king to queen arrow.">
<defs><marker id="eca" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#4a9eff"></path></marker><marker id="ecb" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#f5a623"></path></marker><marker id="ecc" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#3fb950"></path></marker></defs>
<text x="240" y="24" text-anchor="middle" font-size="18" fill="#e8eaf0">similar tokens point the same way</text>
<line x1="70" y1="270" x2="450" y2="270" stroke="#2a3450" stroke-width="1.5"></line><line x1="70" y1="270" x2="70" y2="50" stroke="#2a3450" stroke-width="1.5"></line>
<line x1="70" y1="270" x2="390" y2="150" stroke="#3fb950" stroke-width="2.6" marker-end="url(#ecc)"></line><text x="398" y="150" text-anchor="start" font-size="16" fill="#3fb950">cat</text>
<line x1="70" y1="270" x2="380" y2="110" stroke="#3fb950" stroke-width="2.6" marker-end="url(#ecc)"></line><text x="388" y="106" text-anchor="start" font-size="16" fill="#3fb950">dog</text>
<line x1="70" y1="270" x2="150" y2="60" stroke="#4a9eff" stroke-width="2.6" marker-end="url(#eca)"></line><text x="158" y="60" text-anchor="start" font-size="16" fill="#4a9eff">Paris</text>
<path d="M182.4,227.9 A120,120 0 0 0 176.6,215.0" fill="none" stroke="#f5a623" stroke-width="2"></path>
<text x="196.35950130828536" y="227.41401831848134" text-anchor="start" font-size="15" fill="#f5a623">&#952; small</text>
<text x="240" y="312" text-anchor="middle" font-size="15" fill="#8892a4">small angle between cat and dog</text>
<text x="760" y="24" text-anchor="middle" font-size="18" fill="#e8eaf0">directions carry meaning</text>
<line x1="580" y1="250" x2="830" y2="250" stroke="#2a3450" stroke-width="1.6" stroke-dasharray="5 4"></line>
<line x1="650" y1="120" x2="900" y2="120" stroke="#2a3450" stroke-width="1.6" stroke-dasharray="5 4"></line>
<line x1="584.3" y1="242.1" x2="644.3" y2="130.6" stroke="#f5a623" stroke-width="2.6" marker-end="url(#ecb)"></line>
<line x1="834.3" y1="242.1" x2="894.3" y2="130.6" stroke="#f5a623" stroke-width="2.6" marker-end="url(#ecb)"></line>
<circle cx="580" cy="250" r="6" fill="#4a9eff"></circle>
<text x="580" y="276" text-anchor="middle" font-size="16" fill="#e8eaf0">man</text>
<circle cx="650" cy="120" r="6" fill="#4a9eff"></circle>
<text x="650" y="106" text-anchor="middle" font-size="16" fill="#e8eaf0">woman</text>
<circle cx="830" cy="250" r="6" fill="#4a9eff"></circle>
<text x="830" y="276" text-anchor="middle" font-size="16" fill="#e8eaf0">king</text>
<circle cx="900" cy="120" r="6" fill="#4a9eff"></circle>
<text x="900" y="106" text-anchor="middle" font-size="16" fill="#e8eaf0">queen</text>
<text x="775" y="190" text-anchor="middle" font-size="15" fill="#f5a623">same offset</text>
<text x="760" y="312" text-anchor="middle" font-size="15" fill="#8892a4">king &#8722; man + woman &#8776; queen (word2vec, Mikolov et al., 2013)</text>
</svg>
</div>

---

<!-- .slide: id="cosine-similarity" -->

## Cosine Similarity

A dot product that ignores magnitude: it compares only direction.

<div class="cos-formula"><span class="cos-eq">$\textcolor{#f5a623}{\cos\theta} = \dfrac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}$</span><span class="cos-note">ranges from &#8722;1 (opposite) to 1 (same direction)</span></div>

<div class="interactive-host cos-host" data-widget="cosineDial"></div>

---

:::manim id="embedding-space-anim" scene="embedding-space"
:::
