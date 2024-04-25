# Adam-Optimizer-Algorithm-Implementation-
ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION


```
**Algorithm 1:** Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efficient (but less clear) order of computation. g2
t indicates the elementwise
square gt gt. Good default settings for the tested machine learning problems are α = 0.001,
β1 = 0.9, β2 = 0.999 and  = 10−8. All operations on vectors are element-wise. With βt
1 and βt
2
we denote β1 and β2 to the power t.
Require: α: Stepsize
Require: β1, β2 ∈ [0, 1): Exponential decay rates for the moment estimates
Require: f (θ): Stochastic objective function with parameters θ
Require: θ0: Initial parameter vector
m0 ← 0 (Initialize 1st moment vector)
v0 ← 0 (Initialize 2nd moment vector)
t ← 0 (Initialize timestep)
while θt not converged do
t ← t + 1
gt ← ∇θ ft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
vt ← β2 · vt−1 + (1 − β2) · g2
t (Update biased second raw moment estimate)̂
mt ← mt/(1 − βt
1) (Compute bias-corrected first moment estimate)̂
vt ← vt/(1 − βt
2) (Compute bias-corrected second raw moment estimate)
θt ← θt−1 − α ·̂ mt/(√̂ vt + ) (Update parameters)
end while
return θt (Resulting parameters)
```
