# RaLU - A New Activation Function for Deep Neural Network

**RaLU** (not ReLU!), stands for **Rational Linear Unit**, is a simple, parametric, and gradient-stable activation function designed for Deep Neural Network.

* **Gradient stability** -- Resistant to loss-of-gradient problems, and not prone to deterioration even when layered.
* **Learnable** -- Can form the best shape for each unit.
* **Smooth** -- Infinitely differentiable at all points.
* **Zero-centered** -- Beneficial for training.
* **Unbouded output range** -- No "dead neuron".

## Definition

```math
\text{RaLU}_{a}(x) = x \frac{x^{2} + a}{x^{2} + 1}
```

$`a (\in \mathbb{R})`$ is a learnable parameter.

![plot 1](./assets/ralu-1.svg)
![plot 2](./assets/ralu-2.svg)

* Gradient is $a$ at $`x=0`$.
* It asymptotes to the identity function at $`x \to \pm \infty`$, regardless of $`a`$.
* It is an indentity function when $`a=1`$.
* It is a monotonic increasing function when $`0 \le a \le 9`$.
    * Gradient is $0$ at $`x=0`$ when $`a=0`$.
    * Gradient is $0$ at $`x= \pm \sqrt{3}`$ when $`a=9`$.
    * It loses its monotonically increasing property when $`a<0`$ or $`a>9`$.

## Why for DNN?

### Resistant to vanishing/exploding gradient problems

It asymptotes to the identity function at $`x \to \pm \infty`$ and the gradient is $1$.

In other words, it is unlikely to cause a vanishing/exploding gradient and is well suited for regression problems, CV (CNN), NLP (RNN, Transformer), etc.

### Beneficial for training

It outputs zero mean values because it is a zero-centred odd function.
This prevents systematic bias in the activations.

### No "dead neuron problems"

The output range of RaLU is unlimited (unsaturated).
This could avoid the "Dying ReLU Problem".

### Learnable

It has a parameter so that it can form the best possible shape for each unit or layer.

### Fast and lightweight

It uses only basic arithmetic operations, no exponents or trigonometry.

Therefore it is fast enough, although not as fast as ReLU.
