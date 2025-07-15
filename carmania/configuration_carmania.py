from transformers import PretrainedConfig


class CarmaniaConfig(PretrainedConfig):
    model_type = "carmania"

    def __init__(
        self,
        vocab_size=5,
        hidden_size=1024,
        intermediate_size=4608,
        num_layers=5,
        num_attention_heads=16,
        num_key_value_heads=4,
        attention_window_size=128,
        mlp_hidden_act="silu",
        seq_length=10000,
        use_positional_embedding=True,
        rope_base=10000,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_window_size = attention_window_size
        self.mlp_hidden_act = mlp_hidden_act
        self.seq_length = seq_length
        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
