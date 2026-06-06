import torch
from transformers import GPT2Config, GPT2LMHeadModel

from coconut_qwen.modeling import CoconutConfig, CoconutForCausalLM, make_labels


def tiny_model():
    cfg = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(cfg)


def test_latent_positions_extend_sequence_and_backprop():
    base = tiny_model()
    model = CoconutForCausalLM(base, CoconutConfig(latent_steps=2, eot_token_id=3))
    prefix = torch.tensor([[2, 4, 5]])
    suffix = torch.tensor([[3, 6, 7]])
    labels = make_labels(prefix.shape[1], 2, suffix[0].tolist(), [False, True, True])

    out = model(prefix, suffix, labels)
    assert out["logits"].shape[1] == 8
    out["loss"].backward()
    assert base.transformer.wte.weight.grad is not None


def test_cached_forward_matches_suffix_training_shape_and_backprop():
    base = tiny_model()
    model = CoconutForCausalLM(base, CoconutConfig(latent_steps=2, eot_token_id=3))
    prefix = torch.tensor([[2, 4, 5]])
    suffix = torch.tensor([[3, 6, 7]])
    labels = make_labels(prefix.shape[1], 2, suffix[0].tolist(), [False, True, True])

    out = model.forward_cached(prefix, suffix, labels)
    assert out["logits"].shape[1] == suffix.shape[1]
    out["loss"].backward()
    assert base.transformer.wte.weight.grad is not None


def test_make_labels_masks_prefix_latents_and_unsupervised_suffix():
    labels = make_labels(3, 2, [9, 10, 11], [False, True, True])
    assert labels.tolist() == [[-100, -100, -100, -100, -100, -100, 10, 11]]
