# Matrix-Qwen architecture

This document is the mathematical specification of how MatrixChat wraps
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
into a multi-agent, column-synchronous conversational model. It describes the
architecture that the current ranking leaders all share
(`h100p3_cand_00106` / `qkv_gated` + simplex), not every optional module that
exists in the codebase.

Authoritative implementation:

- wrapper, flatten, embeddings, heads, loss: `model/matrix_qwen.py`
- gated Q/K/V identity injection: `model/agent_attention.py`
- matrix-causal mask: `model/masks.py`
- turn-taking reward: `model/turn_reward.py`
- autoregressive generation: `model/generation.py`
- LoRA / freeze / unfreeze: `model/lora_utils.py`
- row-shifted labels: `data/convert.py`

Optional modules that exist but are **off** on the current leaders
(`agent_dynamic_state_mode=none`, `agent_relation_bias_mode=none`,
`agent_same_attention_bias=false`, `use_channel_embeddings=false`) are
summarized in §12 so they are not confused with the deployed model.

---

## 1. What is left unchanged in Qwen3

Let \(L=36\) be the number of decoder layers, \(H=2560\) the hidden size,
\(n_h=32\) query heads, \(n_{\mathrm{kv}}=8\) key/value heads (grouped-query
attention), \(d_h=128\) the per-head dimension, and \(V=151936\) the vocabulary
size. Qwen3 uses RMSNorm, rotary position embeddings (RoPE) with
\(\theta=5\cdot 10^6\), SwiGLU MLPs, and Q/K RMSNorm inside each attention
block. Tied input/output embeddings.

MatrixChat does **not** replace those blocks. It never edits `q_proj`,
`k_proj`, `v_proj`, or `o_proj` weights in place. The pretrained transformer
still sees a 1-D token sequence of length \(S=A T\) and still runs ordinary
causal self-attention on that sequence. Everything multi-agent is either:

1. a reindexing of the conversation into that 1-D sequence,
2. a replacement of the additive attention mask and of RoPE position ids,
3. extra embeddings / a second output head around the frozen (or LoRA'd)
   backbone,
4. a residual added to Q/K/V **after** the pretrained projections and **before**
   Qwen's Q/K RMSNorm and RoPE.

That last residual is installed with `register_forward_hook` so PEFT LoRA can
wrap the projections without a fork of the Hugging Face attention class.

---

## 2. The conversation as a matrix

A training example is a tensor of token ids

\[
X \in \{0,\ldots,V-1\}^{B \times A \times T},
\]

together with a binary activity mask

\[
M \in \{0,1\}^{B \times A \times T},
\]

where \(B\) is batch size, \(A\) is the number of agent rows in this example
(\(A\le A_{\max}=8\); leaders train at \(A=4\)), and \(T\) is the number of
**time columns**. Column \(t\) is one simultaneous slice of the conversation:
agent \(a\) may speak a single token there, or yield.

There is **no silence token** in the vocabulary. Yielding is represented by
\(M_{b,a,t}=0\). The corresponding \(X_{b,a,t}\) is an arbitrary placeholder
id (conventionally \(0\)); its embedding is never used (§4).

Turns occupy contiguous columns on the speaker's row. After each turn a
random run of all-inactive columns is inserted. Turn ends are therefore a
change in \(M\), not an EOS token (`data/convert.py`).

Private cells (Werewolf role cards) additionally carry a boolean private mask
and a visibility matrix \(\mathrm{Vis}\in\{0,1\}^{B\times A\times A}\) with
\(\mathrm{Vis}_{b,\mathrm{owner},\mathrm{viewer}}=1\) iff the viewer may attend
to that owner's private keys. Public cells ignore \(\mathrm{Vis}\).

---

## 3. Column-major flattening

Qwen consumes rank-2 sequences. The wrapper maps the matrix to a flat index
\(i\in\{0,\ldots,S-1\}\) with \(S=AT\) by **column-major** order:

\[
i = t A + a, \qquad t=\bigl\lfloor i/A\bigr\rfloor, \qquad a = i \bmod A.
\]

In code this is `input_ids.permute(0, 2, 1).reshape(B, T*A)`, so the flat
order is

\[
(t{=}0,a{=}0),\;(t{=}0,a{=}1),\;\ldots,\;(t{=}0,a{=}A{-}1),\;
(t{=}1,a{=}0),\;\ldots
\]

The inverse reshape is applied to logits and to the activity head so every
output tensor is again \([B,A,T,\ldots]\).

This flattening is **not** the same as concatenating each agent's utterance
as a 1-D chat. Adjacent flat positions in the same column belong to
**different agents at the same time**. Adjacent columns of the same agent are
\(A\) steps apart on the 1-D tape. That is why the stock causal LM shift
("logit at flat index \(i\) predicts token \(i+1\)") is the wrong content
objective, and why position ids and the attention mask must be rebuilt.

---

## 4. Input embeddings

Let \(E\in\mathbb{R}^{V\times H}\) be Qwen's tied token embedding table, and
write \(e(v)=E_v\). For each flat position \(i\) with agent id \(a(i)\) and
activity bit \(m_i\):

\[
\tilde{e}_i
=
\begin{cases}
e(x_i) & \text{if } m_i=1,\\
u & \text{if } m_i=0,
\end{cases}
\qquad
u \in \mathbb{R}^{H}
\text{ a learned inactive vector (one row, } \texttt{inactive\_embedding}\text{).}
\]

A learned row embedding \(r_a\in\mathbb{R}^{H}\) is then added (leaders:
`use_agent_embeddings=true`):

\[
h_i^{(0)} = \tilde{e}_i + r_{a(i)}.
\]

\(r_a\) is an ordinary `nn.Embedding(A_max, H)` initialized
\(\mathcal{N}(0,0.02^2)\). It is a **static** identity cue in residual
stream space. The gated-QKV path in §7 is a second, attention-space identity
cue; both are on for the leaders.

Channel embeddings exist in the wrapper but are disabled
(`use_channel_embeddings=false`).

The tensor \(h^{(0)}\) is passed to Qwen as `inputs_embeds`. Token ids are
**not** re-looked-up inside the backbone, so inactive cells never leak the
placeholder id's pretrained meaning.

---

## 5. Column RoPE (`position_mode=column`)

Stock Qwen would assign RoPE index \(i\) to flat position \(i\), which would
treat co-temporal agents as if they occurred at different times, and would
stretch one agent's own timeline by a factor of \(A\).

Instead the wrapper builds

\[
p_i = \bigl\lfloor i/A\bigr\rfloor = t(i) \in \{0,\ldots,T-1\}
\]

and passes `position_ids` of shape \([B,S]\). Every agent in column \(t\)
shares the same rotary phase. An agent's successive spoken tokens, even with
intervening inactive columns, still advance \(p\) by the true column clock.

The unused alternative `position_mode=flat` would set \(p_i=i\). Leaders do
not use it.

Qwen's RoPE itself is unmodified: if \(q,k\in\mathbb{R}^{d_h}\) are a head's
query/key after Q/K-norm, they are rotated by the usual complex
multiplication with \(e^{i p \theta_j}\) at even/odd pairs, using the
supplied \(p\).

---

## 6. Matrix-causal attention mask

Qwen attention at layer \(\ell\) is grouped-query attention. After the
hooks of §7, queries, keys, and values are reshaped to heads, Q and K are
RMSNormed per head, RoPE is applied, and

\[
\mathrm{Attn}(Q,K,V)
=
\mathrm{softmax}\!\left(
\frac{Q K^\top}{\sqrt{d_h}} + \mathcal{M}
\right) V.
\]

\(\mathcal{M}\in\mathbb{R}^{B\times 1\times S\times S}\) is an **additive**
mask (broadcast over heads). Allowed pairs contribute \(0\); blocked pairs
contribute a large negative (\(-10^4\) in fp16/bf16, otherwise
\(\texttt{finfo.min}\)).

For a query cell \(q=(t_q,a_q)\) and key cell \(k=(t_k,a_k)\), leaders use
`allow_same_column=false`:

\[
\mathrm{allowed}(q,k)
=
\bigl(t_k < t_q\bigr)
\;\lor\;
\bigl(t_k=t_q \land a_k=a_q\bigr).
\]

That is: attend to **all previous columns** (every agent), plus the query's
**own** slot. Other agents in the *current* column are blocked. This prevents
same-step teacher-forcing leakage (agent \(a\) cannot see agent \(a'\)'s
ground-truth token at time \(t\) while predicting at time \(t\)).

The diagonal is always allowed even for inactive queries. Without it, column
\(t=0\) would have an empty key set, softmax would be uniform over the
whole row (including the future), and causality would collapse.

Inactive keys are additionally hidden. Let \(I_j=[m_j=0]\). Then a key \(k\)
with \(I_k=1\) is blocked for every query except \(k\) itself. An inactive
slot therefore cannot be read later, including by the same agent: it carries
no token.

Private keys are blocked unless \(\mathrm{Vis}_{b,a_k,a_q}=1\), again except
the diagonal. Newly generated columns are public.

Stock Qwen causal masking (lower-triangular on the flat tape) is **not**
used. A lower-triangular mask on column-major order would let agent \(a\) at
time \(t\) see agents \(0,\ldots,a-1\) in the same column and would hide
agents \(a+1,\ldots,A-1\) in previous columns — both wrong.

---

## 7. Gated simplex Q/K/V identity (`qkv_gated` + `simplex`)

This is the only attention-level architectural change on the leaders.

### 7.1 Regular-simplex agent codes

Let \(A_{\max}=8\). Define centered one-hot codes in \(\mathbb{R}^{A_{\max}}\):

\[
c_a
=
\frac{
e_a - \tfrac{1}{A_{\max}}\mathbf{1}
}{
\bigl\|e_a - \tfrac{1}{A_{\max}}\mathbf{1}\bigr\|_2
}
\qquad
a=0,\ldots,A_{\max}-1,
\]

where \(e_a\) is the \(a\)-th standard basis vector. These \(A_{\max}\)
points are the vertices of a regular simplex centered at the origin: they
are permutation-equivariant, equally spaced, and have \(\langle c_a,c_{a'}\rangle\)
constant for \(a\neq a'\). They are a registered buffer, not parameters.

**Note.** The hyperparameter `agent_attention_dim=32` is ignored in simplex
mode. The code sets the representation dimension to \(A_{\max}\), not 32
(`AgentQKVConditioner` in `model/agent_attention.py`). Learned-vector mode
(`representation=learned`) would instead use an embedding
\(\mathbb{R}^{A_{\max}\times d}\) with \(d=32\); leaders do not use that.

### 7.2 Residual on the pretrained projections

Write Qwen's (possibly LoRA-wrapped) projections at layer \(\ell\) as

\[
Q_0 = h W_Q,\qquad
K_0 = h W_K,\qquad
V_0 = h W_V,
\]

with \(Q_0\in\mathbb{R}^{B\times S\times n_h d_h}\)
(\(n_h d_h=4096\)) and
\(K_0,V_0\in\mathbb{R}^{B\times S\times n_{\mathrm{kv}} d_h}\)
(\(n_{\mathrm{kv}} d_h=1024\)). Biases are absent (`attention_bias=false`).

A per-layer adapter maps the simplex code of the token's agent into each of
those three spaces:

\[
\Delta Q = c_{a(\cdot)}\, W_Q^{(\ell)},\quad
\Delta K = c_{a(\cdot)}\, W_K^{(\ell)},\quad
\Delta V = c_{a(\cdot)}\, W_V^{(\ell)},
\]

with \(W_Q^{(\ell)}\in\mathbb{R}^{A_{\max}\times 4096}\),
\(W_K^{(\ell)},W_V^{(\ell)}\in\mathbb{R}^{A_{\max}\times 1024}\),
initialized \(\mathcal{N}(0, \sigma^2)\) at \(\sigma=10^{-3}\). No adapter
bias.

### 7.3 Content-dependent gate

`mode=qkv_gated` additionally learns, at every layer and every token,

\[
g_i^{(\ell)}
=
\sigma\!\bigl(
\langle w_g^{(\ell)}, h_i\rangle + b_g^{(\ell)}
\bigr)
\in (0,1),
\]

where \(w_g^{(\ell)}\in\mathbb{R}^{H}\), \(b_g^{(\ell)}\in\mathbb{R}\), and
\(\sigma\) is the logistic sigmoid. Both \(w_g\) and \(b_g\) are initialized
at **0**, so at the start of training \(g_i^{(\ell)}=\sigma(0)=1/2\)
everywhere. Combined with small \(\Delta\), the initial perturbation of
pretrained Q/K/V is \(\tfrac12\sigma\)-scale.

The hook then returns

\[
Q = Q_0 + g\,\Delta Q,\qquad
K = K_0 + g\,\Delta K,\qquad
V = V_0 + g\,\Delta V
\]

(broadcast \(g\) over the last dimension; \(g\) is computed in fp32 from the
pre-projection hidden state, then the residual is cast back to the
projection dtype). Ungated `mode=qkv` would omit \(g\) (fixed strength 1).
Leaders use the gated form.

After this residual, Qwen proceeds exactly as usual: reshape to heads, Q/K
RMSNorm, RoPE with column positions, GQA repeat of K/V, masked softmax,
`o_proj`.

Mechanistically the simplex code is a persistent, permutation-equivariant
**row identity** available inside every attention map, while the gate lets
the model shrink that identity when the residual stream already knows who
is speaking (or when identity would interfere with content).

---

## 8. Two heads on the same hidden state

Let \(h^{(L)}_i\in\mathbb{R}^{H}\) be Qwen's final hidden state at flat
position \(i\) (post-final RMSNorm, the same vector that feeds the tied
lm_head). Unflatten to \(h_{b,a,t}\).

### 8.1 Content head (unmodified lm_head)

\[
z_{b,a,t}
=
h_{b,a,t}\, E^\top
\in \mathbb{R}^{V},
\qquad
P_{\mathrm{tok}}(v\mid b,a,t)
=
\mathrm{softmax}(z_{b,a,t})_v.
\]

This is Qwen's ordinary next-token distribution, but it is **only meaningful
conditional on speaking**. It is never trained to emit a silence symbol.

### 8.2 Activity head (new)

A linear map \(w_{\mathrm{act}}\in\mathbb{R}^{H}\), \(b_{\mathrm{act}}\in\mathbb{R}\)
(the `activity_head`) produces

\[
\alpha_{b,a,t}
=
\langle w_{\mathrm{act}}, h_{b,a,t}\rangle + b_{\mathrm{act}},
\qquad
q_{b,a,t}
=
\sigma(\alpha_{b,a,t})
=
P(\text{agent }a\text{ speaks at column }t{+}1\mid \text{context through }t).
\]

Independence across agents at the next column is the modeling assumption used
by the reward (§10): given the \(q_{\cdot,a,t}\), the next column's occupancy
is a product of Bernoullis.

The normalized per-cell generation policy is therefore

\[
P(\text{yield at }t{+}1)=1-q_{a,t},
\qquad
P(\text{token }v\text{ at }t{+}1)
=
q_{a,t}\, P_{\mathrm{tok}}(v\mid a,t).
\]

---

## 9. Supervision is row-shifted, not tape-shifted

Hugging Face `Qwen3ForCausalLM` loss would shift along the flat tape:
logit at \(i\) versus token at \(i+1\). Under column-major flattening,
\(i+1\) is usually **another agent in the same column**, not this agent's
next token. The wrapper therefore **ignores** the backbone's built-in loss
and applies an unshifted cross-entropy to already-shifted matrix labels
(`data/convert.py`).

### 9.1 Activity labels

For every agent and every column except the last,

\[
Y^{\mathrm{act}}_{a,t} = M_{a,t+1}\in\{0,1\}.
\]

This one target covers floor-taking (\(0\to 1\)), continuation (\(1\to 1\)),
and stopping (\(1\to 0\)). Column \(T-1\) is ignore-index.

### 9.2 Content labels

Let \(\mathcal{R}\) be the set of rows that receive content supervision
(leaders: all speakers on content-bearing sources). Then

\[
Y^{\mathrm{tok}}_{a,t}
=
\begin{cases}
X_{a,t+1} & \text{if }a\in\mathcal{R}\text{ and }M_{a,t}=M_{a,t+1}=1,\\
-100 & \text{otherwise.}
\end{cases}
\]

The last token of a turn has no content target: whether to stop is an
activity decision. Gaps and other agents' speech are context, not content
loss.

### 9.3 Weighted activity BCE

Let \(\mathcal{P}=\{(b,a,t): Y^{\mathrm{act}}=1\}\) and
\(\mathcal{N}\) the yield cells. With class weight
\(w=0.75\) (`activity_pos_weight`) on the leaders,

\[
\mathcal{L}_{\mathrm{act}}
=
w\,\overline{\mathrm{BCE}}(\alpha;1)_{\mathcal{P}}
+
(1-w)\,\overline{\mathrm{BCE}}(\alpha;0)_{\mathcal{N}}.
\]

\(w>1/2\) penalizes missed-speak more than extra-speak, so the (much more
common) yield class cannot dominate.

### 9.4 Content CE

\[
\mathcal{L}_{\mathrm{tok}}
=
\mathrm{CE}\bigl(z,\, Y^{\mathrm{tok}}\bigr)
\quad\text{with ignore-index }-100.
\]

If a batch has no valid content targets, \(\mathcal{L}_{\mathrm{tok}}:=0\)
(plain `F.cross_entropy` on an all-ignored batch is NaN).

---

## 10. Differentiable turn-taking reward (Stage A)

The activity logits at column \(t-1\) are treated as independent Bernoulli
parameters for who speaks at column \(t\). For \(t=1,\ldots,T-1\):

\[
q_a := \sigma(\alpha_{a,t-1}),
\qquad
p_0 = \prod_{a=1}^{A}(1-q_a)
\quad\text{(all silent)},
\]

\[
p^{\mathrm{only}}_a
=
q_a \prod_{a'\neq a}(1-q_{a'})
\quad\text{(exactly agent }a\text{ speaks)},
\]

\[
p_{\mathrm{one}}=\sum_a p^{\mathrm{only}}_a,
\qquad
p_{\mathrm{overlap}}=1-p_0-p_{\mathrm{one}}.
\]

Let \(o_{t-1}\) be the unique speaker at column \(t-1\), or \(\bot\) if that
column was silent or overlapped (computed from **ground-truth** \(M\), not
from the model — this is Stage A teacher forcing). Then

\[
p_{\mathrm{same}}
=
\begin{cases}
p^{\mathrm{only}}_{o_{t-1}} & o_{t-1}\neq\bot,\\
0 & \text{otherwise,}
\end{cases}
\qquad
p_{\mathrm{handoff}}
=
\begin{cases}
p_{\mathrm{one}}-p_{\mathrm{same}} & o_{t-1}\neq\bot,\\
p_{\mathrm{one}} & \text{otherwise (floor acquisition).}
\end{cases}
\]

Run lengths \(x_{t-1}\) (consecutive sole-speaker columns of the same owner),
\(z_{t-1}\) (consecutive all-silent columns), and \(\omega_{t-1}\)
(consecutive overlap columns) are also read from ground-truth \(M\). The
shaped reward at column \(t\) is

\begin{align*}
s_x &= \exp\!\bigl(-\max(x_{t-1}-g_{\mathrm{spk}},0)/\tau_{\mathrm{spk}}\bigr),\\
w_{\mathrm{sil}} &= 1-\exp\!\bigl(-\max(z_{t-1}-g_{\mathrm{sil}},0)/\tau_{\mathrm{sil}}\bigr),\\
\rho &= 1-\exp\!\bigl(-\max(\omega_{t-1}-g_{\mathrm{ov}},0)/\tau_{\mathrm{ov}}\bigr),\\
w_{\mathrm{ov}} &= w_{\mathrm{ov}}^{\mathrm{base}}
+(w_{\mathrm{ov}}^{\mathrm{max}}-w_{\mathrm{ov}}^{\mathrm{base}})\,\rho,\\
R_t &=
s_x\, p_{\mathrm{same}}
+(1+\beta)\, p_{\mathrm{handoff}}
- w_{\mathrm{sil}}\, p_0
- w_{\mathrm{ov}}\, p_{\mathrm{overlap}}.
\end{align*}

Leader hyperparameters:

| symbol | config field | value |
|---|---|---|
| \(g_{\mathrm{spk}},\tau_{\mathrm{spk}}\) | `speak_grace`, `speak_tau` | \(5\), \(200\) |
| \(g_{\mathrm{sil}},\tau_{\mathrm{sil}}\) | `silence_grace`, `silence_tau` | \(0\), \(5\) |
| \(g_{\mathrm{ov}},\tau_{\mathrm{ov}}\) | `overlap_grace`, `overlap_tau` | \(0\), \(2\) |
| \(w_{\mathrm{ov}}^{\mathrm{base}}, w_{\mathrm{ov}}^{\mathrm{max}}\) | `overlap_base_weight`, `overlap_max_weight` | \(1.674\), \(5.567\) |
| \(\beta\) | `handoff_bonus_weight` | \(1.1\) |

\(\bar R\) is the mean of \(R_t\) over valid columns. Gradients flow only
through the \(q_a\) (hence through \(\alpha\)), not through the ground-truth
counters.

A separate same/handoff proper-scoring loss exists
(`same_handoff_cross_entropy`) that scores only genuine new-arrival columns.
Leaders set `lambda_same_handoff=0`, so it does not enter the deployed
objective.

---

## 11. Full training objective

The wrapper loss is

\[
\mathcal{L}_{\mathrm{wrap}}
=
\lambda_{\mathrm{tok}}\,\mathcal{L}_{\mathrm{tok}}
+
\lambda_{\mathrm{act}}\,\mathcal{L}_{\mathrm{act}}
-
\lambda_{R}\,\bar R,
\]

with leaders \(\lambda_{\mathrm{tok}}=1\), \(\lambda_{\mathrm{act}}=0.5\),
\(\lambda_{R}=0.05\).

On top of that, `main.py` adds L2-SP on every **trainable non-LoRA**
parameter of the base model (the unfrozen first two and last two decoder
layers), anchored at the load-time snapshot:

\[
\mathcal{L}_{\mathrm{L2SP}}
=
\lambda_{\mathrm{L2SP}}
\cdot
\frac{1}{N}
\sum_{\theta\in\Theta_{\mathrm{unfrozen}}}
\|\theta-\theta_0\|_2^2,
\qquad
\lambda_{\mathrm{L2SP}}=0.02,
\]

where \(N\) is the total number of scalar elements in \(\Theta_{\mathrm{unfrozen}}\).
LoRA weights and the matrix interface (`agent_embeddings`, QKV adapters,
`inactive_embedding`, `activity_head`) are not in this sum.

The optimized loss is \(\mathcal{L}_{\mathrm{wrap}}+\mathcal{L}_{\mathrm{L2SP}}\).

### 11.1 Which Qwen parameters move

1. Freeze the entire backbone.
2. Inject LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` with rank
   \(r=128\), \(\alpha=256\), dropout \(0.05\). For a weight
   \(W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\),

   \[
   W_{\mathrm{eff}} = W + \frac{\alpha}{r} B A,
   \qquad
   A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\;
   B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
   \]

3. Unfreeze decoder layers \(\{0,1\}\) and \(\{34,35\}\) (first two and last
   two of 36). Those full weights are the L2-SP-anchored set.
4. Keep the matrix-interface modules fully trainable (they were never frozen).

After PEFT replaces the projection modules, `install_agent_attention_hooks`
is called again so the §7 residuals still sit on the LoRA-wrapped `q/k/v_proj`.

---

## 12. Autoregressive generation

Generation is column-synchronous, not left-to-right on the flat tape
(`model/generation.py`). Given a prompt matrix of length \(T_0\), for each
new column \(s=1,\ldots,S_{\mathrm{new}}\):

1. Run the full wrapper on the current \([B,A,T_0+s-1]\) matrix.
2. Read \(\alpha_{\cdot,a,T_{\mathrm{last}}}\) and \(z_{\cdot,a,T_{\mathrm{last}},:}\).
3. Speak iff \(q_a > \tau\) with \(\tau=0.5\).
4. If speaking, sample \(v\sim\mathrm{softmax}(z_a/\tau_{\mathrm{temp}})\)
   (evaluation contract: \(\tau_{\mathrm{temp}}=1\)) or argmax if
   \(\tau_{\mathrm{temp}}=0\). If yielding, emit the placeholder id and mark
   the new cell inactive.
5. Append the new column and repeat. New columns are public.

There is no beam search. KV cache is disabled (`use_cache=False`) because
the mask, position ids, and QKV hooks are rebuilt over the whole matrix each
step.

---

## 13. Optional modules that are **not** part of the leaders

These are implemented and tested, but the current top models set them to
identity / off. They are listed so a reader of the source does not assume
they are active.

**Dynamic per-agent GRU** (`model/dynamic_agent_state.py`). A causal GRU
along each agent's row of token embeddings, added into \(h^{(0)}\). Off:
`agent_dynamic_state_mode=none`.

**Same-agent attention-logit bias.** A learned scalar \(\beta^{(\ell)}\)
added to \(\mathcal{M}\) on pairs with \(a_q=a_k\). Off:
`agent_same_attention_bias=false`.

**Generalized relation bias.** Either per-layer same/diff scalars or a
bilinear identity bias \(q_{\mathrm{id}}^\top k_{\mathrm{id}}/\sqrt{d}\)
added to \(\mathcal{M}\) (AgentFormer-style, still via the additive mask so
LoRA and gradient checkpointing stay intact). Off:
`agent_relation_bias_mode=none`.

**Channel embeddings.** A second additive table over a channel id. Off.

**Same/handoff CE.** Proper scoring of who takes the floor at genuine
new-arrival columns. Present in the forward pass but weighted by
\(\lambda_{\mathrm{same/handoff}}=0\).

---

## 14. End-to-end data flow (leaders)

\[
\begin{aligned}
&(X,M)
\;\xrightarrow{\text{column-major flatten}}\;
(x_i, m_i, a(i), p_i=t(i)) \\
&\;\xrightarrow{\tilde e_i + r_{a(i)}}\;
h^{(0)} \\
&\;\xrightarrow[\text{mask }\S6]{\text{Qwen3-4B, }L=36}
\;
\text{each layer: }
Q,K,V \leftarrow \mathrm{proj}(h)+g\cdot W c_a
\;\to\;
\text{QK-norm, column-RoPE, GQA}\\
&\;\xrightarrow{\text{lm\_head }E^\top,\; w_{\mathrm{act}}}\;
(z_{a,t},\;\alpha_{a,t}) \\
&\;\xrightarrow{\S9\text{--}11}\;
\mathcal{L}.
\end{aligned}
\]

What Qwen3 was: a 1-D instruction-tuned causal LM. What it is here: the same
weights, consuming a column-major multi-agent tape, with (i) a learned
inactive input, (ii) additive row embeddings, (iii) column-indexed RoPE,
(iv) a matrix-causal + inactive + private mask, (v) a gated simplex residual
on every Q/K/V, (vi) a second head that decides speak vs yield, and (vii) a
loss that supervises next-token **along each agent's row**, next-column
activity for every agent, and a soft occupancy reward — not next-token along
the flattened tape.
