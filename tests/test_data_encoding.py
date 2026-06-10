from types import SimpleNamespace

from coconut_qwen.data import CoconutExample, encode_for_stage, strip_gsm8k_calculator_annotations
from coconut_qwen.modeling import EOT_TOKEN


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(EOT_TOKEN, i):
                ids.append(10)
                i += len(EOT_TOKEN)
            elif text.startswith(self.eos_token, i):
                ids.append(1)
                i += len(self.eos_token)
            else:
                ids.append((ord(text[i]) % 20) + 11)
                i += 1
        if add_special_tokens:
            ids = [2] + ids
        return SimpleNamespace(input_ids=ids)


def test_coconut_encoding_supervises_eos_token():
    ex = CoconutExample("What is 1+1?", ["1+1=2."], "2")
    enc = encode_for_stage(ex, FakeTokenizer(), latent_steps=2, stage=0)

    assert enc.target_text.endswith("<eos>")
    assert enc.suffix_ids[-1] == 1
    assert enc.labels[-1] == 1


def test_answer_only_loss_keeps_eos_supervised():
    ex = CoconutExample("What is 1+1?", ["1+1=2."], "2")
    enc = encode_for_stage(
        ex,
        FakeTokenizer(),
        latent_steps=2,
        stage=0,
        supervise_reasoning=False,
    )

    answer_start = enc.target_text.index("\n#### 2")
    unsupervised_prefix_ids = FakeTokenizer()(enc.target_text[:answer_start], add_special_tokens=False).input_ids
    suffix_label_start = len(enc.prefix_ids) + 2
    labels_before_answer = enc.labels[suffix_label_start : suffix_label_start + len(unsupervised_prefix_ids)]

    assert all(label == -100 for label in labels_before_answer)
    assert enc.labels[-1] == 1


def test_strip_gsm8k_calculator_annotations():
    text = "She has 3 * 4 = <<3*4=12>>12 apples."

    assert strip_gsm8k_calculator_annotations(text) == "She has 3 * 4 = 12 apples."
