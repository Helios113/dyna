List of all configs available:
NORMS√
  low_precision_layernorm = "low_precision_layernorm"
  low_precision_rmsnorm = "low_precision_rmsnorm"
  dynamic_thanh = "dynamic_thanh"
Rescaling
  none = "none"
  cum_avg_prot_emb = "cum_avg_prot_emb"
  cum_avg_no_prot_emb = "cum_avg_no_prot_emb"
  sqrt_prot_emb = "sqrt_prot_emb"
  sqrt_no_prot_emb = "sqrt_no_prot_emb"
  sqrt_scale_prot_emb = "sqrt_scale_prot_emb"
  avg_prot_emb = "avg_prot_emb"
early exiting
  True
  False
MODELS layers √
  MoEUT
  Transformer
Bias vs reg loss √
  True
  False
norm_structure:√
  peri = "peri"
  pre = "pre"
  post = "post"
  moeut = "moeut"
Norm after embedding√


Rescaling

Early exit is the big one -- maybe last one?


Callbacks .... ughgggg
Basically I need to compare how callbacks affect gradients -- this is a big bummer imo.
