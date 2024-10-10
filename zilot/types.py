from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor, TensorType

Batch = TensorDict

Obs = TensorType
Goal = Obs
Action = Float[Tensor, "*B dim_a"]
Latent = Float[Tensor, "*B dim_z"]
GLatent = Latent
Value = Float[Tensor, " *B"]
