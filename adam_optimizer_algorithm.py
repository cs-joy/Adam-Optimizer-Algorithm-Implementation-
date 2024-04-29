```
!pip install labml-nn
```

import math
from typing import Dict, Any, Tuple, Optional

import torch
from labml import tracker
from torch import nn

from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay

"""
* `params` is the list of parameters
* `lr` is the learning rate α
* `betas` is a tuple of (β1​, β2​)
* `eps` is ϵ^ or ϵ based on optimized_update
* `weight_decay` is an instance of class WeightDecay defined in `__init__()`
* `optimized_update` is a flag whether to optimize the bias correction of the second moment by doing it after adding ϵ
* `defaults` is a dictionary of default for group values. This is useful when you want to extend the class ``Adam`` .
"""

class Adam(GenericAdaptiveOptimizer):
  # initialize the optimizer
  def __init__(self, params,
               lr: float = 1e-3, betas: Tuple[float, float] = (0.0, 0.999),
               eps: float = 1e-16,
               weight_decay: WeightDecay = WeightDecay(),
               optimized_update: bool = True,
               defaults: Optional[Dict[str, Any]] = None):
    defaults = {} if defaults is None else defaults
    defaults.update(weight_decay.defaults())
    super().__init__(params, defaults, lr, betas, eps)

    self.weight_decay = weight_decay
    self.optimized_update = optimized_update


  # initialize the parameter state
                       # `state` is the optimizer state of the parameter (tensor)
                                              # `group` stores optimizer attributes of the parameter group
                                                                     # `param` is the parameter tensor θ_t−1
  def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
    state['step'] = 0
    state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    state['éxp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


  # calculate m_t abd v_t
                    # `state` is the optimizer state of the parameter (tensor)
                                           # `group` stores optimizer attributes of the parameter group
                                                                  # `grad` is the current gradient tensor `g_t` for the parameter `θ_t−1`
  def get_mv(self, state: Dict [str, Any], group: Dict[str, Any], grad: torch.Tensor):
    beta1, beta2 = group['betas'] # get beta1 and beta2
    m, v = state['exp_avg'], state['exp_avg_sq'] # get m_t-1 and v_t-1
    m.mul_(beta1).add_(grad, alpha = 1 - beta1) # in place calculation of m_t, where m_t << beta1 * m_t-1 + (1 - beta1) * g_t
    v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2) # in place calculation of v_t, where v_t << beta2 * v_t-1 + (1 - beta2) * g_t^2

    return m, v

  # get learning-rate
  # This returns the modified learning rate based on the state. For Adam this is just specified learning rate for the parameter group, alpha
  def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
    return group['lr']

  # Do the Adam parameter update
  def adam_update(self,
      state: Dict[str, any], # `state` is the optimizer state of the parameter (tensor)
      group: Dict[str, any], # `group` stores optimizer attributes of the parameter group
      param: nn.Parameter, # `param` is the parameter tensor θ_t-1
      m: torch.Tensor, # `m` is the uncorrected first moments m_t
      v: torch.Tensor # `v` is the uncorrected scond moments v_t
  ):
    beta1, beta2 = group['betas'] # get β1 and β2
    bias_correction1 = 1 - beta1 ** state['step'] # bias correction term for m_hat_t, (1 - β_t_1 (where `t` is superscript))
    bias_correction2 = 1 - beta2 ** state['step'] # bias correction term for v_hat_t, (1 - β_t_2 (where `t` is superscript))
    lr = self.get_lr(state, group) # get learning rate
    if self.optimized_update: # whether to optimize the computation
      denominator = v.sqrt().add_(group['eps']) # sqrt(v_t) + ϵ_hat
      step_size = lr * math.sqrt(bias_correction2) / bias_correction1 # α * (sqrt(1 - β2_t (where `t` is superscript) / β1- 1_t (where `t` is superscript))
      param.data.addcdiv_(m, denominator, value=-step_size) # θ_t << θ_t−1 - α * (sqrt(1 - β2_t (where `t` is superscript) / 1 - β1_t (where `t` is superscript)) * (m_t / sqrt(v_t) + ϵ_hat)       m_hat_t / sqrt(v_hat_t) + ϵ
    else: # computation without optimization
      denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps']) # sqrt(v_t) / sqrt(1-β2_t (where `t` is superscript)) + ϵ
      step_size = lr / bias_correction1 # α / 1 - β1_t (where `t` is superscript)
      param.data.addcdiv_(m, denominator, value=-step_size) # θ_t << θ_t−1 - α * m_hat_t / sqrt(v_hat_t) + ϵ

  #
  def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
    grad = self.weight_decay(param, grad, group) # calculate weight decay
    m, v = self.get_mv(state, group, grad) # get m_t and v_t
    state['step'] += 1 # Increment t the number of optimizer steps
    self.adam_update(state, group, param, m, v) # Perform Adam update

