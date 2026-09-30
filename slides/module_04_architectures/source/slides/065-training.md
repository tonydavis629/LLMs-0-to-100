<!-- .slide: id="training-ce" -->

## Training: Match the Target Token

<div class="emb-svg">
<svg viewBox="0 0 1000 320" role="img" aria-label="Cross-entropy training: the model gives Paris probability 0.174, the target puts all probability on Paris, and the loss is minus log 0.174, which is 1.75.">
<text x="200" y="28" text-anchor="middle" font-size="17" fill="#e8eaf0">model output p</text>
<text x="200" y="50" text-anchor="middle" font-size="13" fill="#8892a4">softmax of the last row (GPT-2 medium)</text>
<line x1="20" y1="260" x2="380" y2="260" stroke="#2a3450" stroke-width="1.5"></line>
<rect x="40" y="89.8" width="40" height="170.2" rx="3" fill="#f5a623" fill-opacity="0.9"></rect>
<text x="60" y="82.0" text-anchor="middle" font-size="13" fill="#e8eaf0">0.174</text>
<text x="60" y="280" text-anchor="middle" font-size="14" fill="#e8eaf0">Paris</text>
<rect x="102" y="207.4" width="40" height="52.6" rx="3" fill="#f5a623" fill-opacity="0.4"></rect>
<text x="122" y="199.2" text-anchor="middle" font-size="13" fill="#8892a4">0.054</text>
<text x="122" y="280" text-anchor="middle" font-size="14" fill="#8892a4">the</text>
<rect x="164" y="219.0" width="40" height="41.0" rx="3" fill="#f5a623" fill-opacity="0.4"></rect>
<text x="184" y="211.0" text-anchor="middle" font-size="13" fill="#8892a4">0.042</text>
<text x="184" y="280" text-anchor="middle" font-size="14" fill="#8892a4">Lyon</text>
<rect x="226" y="229.8" width="40" height="30.2" rx="3" fill="#f5a623" fill-opacity="0.4"></rect>
<text x="246" y="221.7" text-anchor="middle" font-size="13" fill="#8892a4">0.031</text>
<text x="246" y="280" text-anchor="middle" font-size="14" fill="#8892a4">not</text>
<rect x="288" y="234.5" width="40" height="25.5" rx="3" fill="#f5a623" fill-opacity="0.4"></rect>
<text x="308" y="226.6" text-anchor="middle" font-size="13" fill="#8892a4">0.026</text>
<text x="308" y="280" text-anchor="middle" font-size="14" fill="#8892a4">a</text>
<text x="370" y="280" text-anchor="middle" font-size="14" fill="#8892a4">&#8230;</text>
<text x="660" y="28" text-anchor="middle" font-size="17" fill="#e8eaf0">target y</text>
<text x="660" y="50" text-anchor="middle" font-size="13" fill="#8892a4">the real next token gets all the probability</text>
<line x1="480" y1="260" x2="840" y2="260" stroke="#2a3450" stroke-width="1.5"></line>
<rect x="500" y="90.0" width="40" height="170.0" rx="3" fill="#3fb950" fill-opacity="0.9"></rect>
<text x="520" y="82.0" text-anchor="middle" font-size="13" fill="#e8eaf0">1</text>
<text x="520" y="280" text-anchor="middle" font-size="14" fill="#e8eaf0">Paris</text>
<rect x="562" y="258.5" width="40" height="1.5" rx="3" fill="#3fb950" fill-opacity="0.4"></rect>
<text x="582" y="250.5" text-anchor="middle" font-size="13" fill="#8892a4">0</text>
<text x="582" y="280" text-anchor="middle" font-size="14" fill="#8892a4">the</text>
<rect x="624" y="258.5" width="40" height="1.5" rx="3" fill="#3fb950" fill-opacity="0.4"></rect>
<text x="644" y="250.5" text-anchor="middle" font-size="13" fill="#8892a4">0</text>
<text x="644" y="280" text-anchor="middle" font-size="14" fill="#8892a4">Lyon</text>
<rect x="686" y="258.5" width="40" height="1.5" rx="3" fill="#3fb950" fill-opacity="0.4"></rect>
<text x="706" y="250.5" text-anchor="middle" font-size="13" fill="#8892a4">0</text>
<text x="706" y="280" text-anchor="middle" font-size="14" fill="#8892a4">not</text>
<rect x="748" y="258.5" width="40" height="1.5" rx="3" fill="#3fb950" fill-opacity="0.4"></rect>
<text x="768" y="250.5" text-anchor="middle" font-size="13" fill="#8892a4">0</text>
<text x="768" y="280" text-anchor="middle" font-size="14" fill="#8892a4">a</text>
<rect x="810" y="258.5" width="40" height="1.5" rx="3" fill="#3fb950" fill-opacity="0.4"></rect>
<text x="830" y="250.5" text-anchor="middle" font-size="13" fill="#8892a4">0</text>
<text x="830" y="280" text-anchor="middle" font-size="14" fill="#8892a4">&#8230;</text>
<text x="200" y="300" text-anchor="middle" font-size="13" fill="#8892a4">the other 50,252 tokens share the remaining 0.673</text>
<text x="660" y="300" text-anchor="middle" font-size="13" fill="#8892a4">one-hot: 1 for Paris, 0 for every other token</text>
<line x1="440" y1="40" x2="440" y2="290" stroke="#2a3450" stroke-width="1.2" stroke-dasharray="4 4"></line>
</svg>
</div>

$$\mathcal L = -\sum_w y_w \log p_w = -\log p(\text{Paris}) = -\log 0.174 = 1.75$$

Backpropagation nudges every weight to raise $p(\text{Paris})$. In training every row of the parallel pass has its own target, and the loss is averaged over all of them. Module 5 covers this in depth.
