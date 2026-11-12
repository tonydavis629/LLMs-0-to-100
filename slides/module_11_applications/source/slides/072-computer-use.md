:::divider id="divider-computer-use" title="Computer Use"
:::

---

<!-- .slide: id="cu-loop" -->

## Agents That Use a Screen

Most software has no API. A computer-use agent works through the same screen, mouse, and keyboard a person does.

:::columns cols="2" gap="34px"
**Observation**

- A screenshot, encoded into image tokens (Module 8)
- The model has to find buttons and text fields in the pixels
- Browser agents can also read the page's HTML
+++
**Action**

- Tool calls for the mouse and keyboard: `click(x, y)`, `type(text)`, `key(name)`, `scroll(dy)`
- The runtime performs the action, then takes a new screenshot
:::

This is the agent loop from the last section. Every observation is a screenshot. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="cu-demo" -->

## A Computer Use Trace

Task: "My printer shows E-341. Find the fix, and order the part if I need one."

<div class="shot-strip">
<figure>
<p class="shot-step">Screenshot 1</p>
<img src="images/computer_use_1.png" alt="Support site home page with an empty search box, marked where the agent clicks">
<pre class="shot-action">click(324, 103)
type("E-341")
key("Enter")</pre>
</figure>
<figure>
<p class="shot-step">Screenshot 2</p>
<img src="images/computer_use_2.png" alt="Article for error E-341 with an order button, marked where the agent clicks">
<pre class="shot-action">click(200, 332)</pre>
</figure>
<figure>
<p class="shot-step">Screenshot 3</p>
<img src="images/computer_use_3.png" alt="Confirmation dialog asking whether to place the order">
<pre class="shot-action">ask_user("The fuser assembly FA-7700
is in the cart. Place the order?")</pre>
</figure>
</div>

The red ring marks where each click lands. Before the irreversible step, the agent stops and asks. (A mock site built for this slide.) <!-- .element: class="text-lg" -->

---

<!-- .slide: id="cu-hard" -->

## Why Computer Use Is Hard

<div class="card-grid cols-4">
<div class="card"><h4>Many steps</h4><p>A task a person finishes in a minute can take dozens of actions, and errors compound across them.</p></div>
<div class="card"><h4>Grounding</h4><p>The model has to turn "the Search button" into pixel coordinates. A click a few pixels off hits a different control.</p></div>
<div class="card"><h4>Cost and speed</h4><p>Every step sends a new screenshot, which is a large block of image tokens (Module 8), and waits for a model call.</p></div>
<div class="card warn"><h4>Untrusted screens</h4><p>Any text on a web page can be an injected instruction, and this agent can click "Place order".</p></div>
</div>

OSWorld (2024) scores 369 desktop tasks by their end state. At release, humans completed 72% and the best model 12%. Models passed the human baseline in early 2026 (Module 9). <!-- .element: class="text-lg" style="margin-top: 12px;" -->
