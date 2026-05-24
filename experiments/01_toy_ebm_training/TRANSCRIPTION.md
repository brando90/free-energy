# Transcription of Toy EBM Notes

Source images:

- `assets/toy_ebm_notes_photo_1.jpg`
- `assets/toy_ebm_notes_photo_2.jpg`

These are handwritten notes photographed sideways. The transcription below is
best-effort and preserves the mathematical intent rather than pretending every
symbol is certain. Unclear text is marked explicitly.

## Photo 1

Top note, main update:

```text
Note:
theta^{t+1} := theta^t
  - eta { grad_theta E_theta(x)
          - E_{x_tilde ~ p_theta}[grad_theta E_theta(x_tilde)] }
```

Equivalent line, partially rewritten:

```text
:= H(theta^t)
   - eta H( + grad_theta E_theta(w)
            - E_{x_tilde ~ p_theta}[grad_theta E_theta(x_tilde)] )
   + e[J]     unclear
```

Right/top margin:

```text
temp an ebm (machine learning?)
def train(beta, steps)
theta_s := p*
```

Middle notes:

```text
# Goal: max likelihood / MLE:
E_{x ~ p*}[log p_theta(x)]

# ... max log p_theta(x)

# p_theta log p_theta
# ... but p_theta log Z_theta and d/dtheta Z_theta = ...
# ... but since there is a sum, grad_theta Z_theta = E_{...}[...]

# update needs info if model?

# do:
theta^{t+1} := theta^t - eta grad_theta loss(theta)
```

Lower notes:

```text
# for more general sample:
x ~ p*(x)        # real data batch

theta_0 := theta_0 - E_{x ~ p_0}[E(x)]      unclear
let g := -D_theta E_theta(x)                # real data batch

g to produce a maybe x_tilde ~ p_theta      unclear

# currently call hmm? / hmc? let x_tilde ~ p_theta via x ...
x_tilde = train(p_theta^s, steps=20)
b_j = - grad_theta E(x_tilde^s)

eta_t g = D_theta E(x)       unclear
```

Bottom/left formula:

```text
mathcal{J} = D_theta E(x_tilde)
```

Bottom/right note:

```text
g_maybe = left - right
theta_{t+1} := theta_t - eta g_maybe if isn't ...
```

Interpretation:

- The update is the standard EBM MLE gradient:
  positive/data phase minus model/negative phase.
- `p*` is the data distribution.
- `p_theta` is the model distribution.
- `x_tilde ~ p_theta` is the model sample used for the negative phase.
- The note is trying to turn the derivation into a `train(...)` procedure.

## Photo 2

Left/top margin:

```text
temp an ebm (machine learning?)

def train(beta, steps)
theta_s := p*
```

Central algorithm sketch:

```text
... + e[J]

# Goal: max likelihood / MLE:
E_{x ~ p*}[log p_theta(x)]

# ... max grad log p_theta(x)

# p_theta log p_theta = -D_theta E_theta(x) - D_theta log Z_theta

# D_theta log Z_theta and d/dtheta Z_theta = ...
# ... but since there is a sum, D_theta Z_theta = E_{...}[...]

# update needs info if model?

# do:
theta^{t+1} := theta^t - eta grad_theta loss(theta)
```

Lower algorithm sketch:

```text
# for more general sample:
x ~ p*(x)        # real data batch

theta_0 := theta_0 - E_{x ~ p_theta}[E(x)]      unclear
let g := -D_theta E_theta(x)                    # real data batch

g to produce a maybe x_tilde ~ p_theta          unclear

# currently call hmm? / hmc? let x_tilde ~ p_theta via x ...
x_tilde = train(p_theta^s, steps=20)
b_j = -D_theta E(x_tilde^s)
```

Bottom note:

```text
mathcal{J} = D_theta E(x_tilde)
g_maybe = left - right
theta_{t+1} := theta_t - eta g_maybe if isn't ...
```

Interpretation:

- The same update appears twice, once as a derivation and once as a rough
  training loop.
- `train(beta, steps)` looks like a placeholder for drawing or refining model
  samples, possibly via MCMC/HMC.
- The toy experiment implemented here uses exact finite enumeration first.
  That is the cleanest executable version of the note because it computes the
  negative phase without sampler confounds.

