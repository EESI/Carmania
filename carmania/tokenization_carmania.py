from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple, Dict
import numpy as np
import os
import json

# --- Custom DNA Vocabulary for Carmania ---
VOCAB = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '[PAD]': 4}
ID2CHAR = {v: k for k, v in VOCAB.items()}

class CarmaniaTokenizer(PreTrainedTokenizer):
    tokenizer_class = "CarmaniaTokenizer"
    model_input_names = ["input_ids"]

    def __init__(
        self,
        model_max_length: int = 512,
        pad_token='[PAD]',
        unk_token='[PAD]',
        bos_token=None,
        eos_token=None,
        calculate_bigram: bool = False,
        **kwargs
    ):
        self.vocab = VOCAB
        self.inv_vocab = ID2CHAR
        self.model_max_length = model_max_length
        self.calculate_bigram = calculate_bigram

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            model_max_length=model_max_length,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab['[PAD]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self.inv_vocab.get(index, '[PAD]')

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    def encode_with_bigram(self, text: str) -> Tuple[List[int], Optional[np.ndarray]]:
        text = text.upper()
        token_ids = []
        bigram_matrix = np.zeros((4, 4), dtype=np.int16) if self.calculate_bigram else None

        prev = None
        for char in text[:self.model_max_length]:
            idx = self.vocab.get(char, self.vocab['[PAD]'])
            token_ids.append(idx)
            if self.calculate_bigram and prev is not None and idx < self.vocab['[PAD]'] and prev < self.vocab['[PAD]']:
                bigram_matrix[prev, idx] += 1
            prev = idx

        # Padding if needed
        if len(token_ids) < self.model_max_length:
            token_ids += [self.vocab['[PAD]']] * (self.model_max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.model_max_length]

        return token_ids, bigram_matrix
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Hugging Face calls this when using AutoTokenizer
        tokenizer = cls(**kwargs)
        return tokenizer