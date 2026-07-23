import pytest
import torch

from cs336_basics.decoding import generate_token_ids, generate_token_ids_cached
from cs336_basics.multihead_attention import CausalMultiHeadSelfAttention
from cs336_basics.transformer_lm import TransformerLM

def test_kv_cache():
    torch.manual_seed(0)

    layer = CausalMultiHeadSelfAttention(
        d_model=8,
        num_heads=2,
        theta=10000,
        max_seq_len=16,
        device="cpu",
        dtype=torch.float32
    )

    layer.eval()
    x = torch.randn(1,4,8)

    with torch.inference_mode():
        full_output = layer(x)
        expected = full_output[:,3:4,:]
        prefill_output, cache_after_prefill = layer(x[:,:3,:], use_cache=True)    
        past_K, past_V = cache_after_prefill
        cached_output, updated_cache = layer(x[:,3:4,:],
                                            past_key_value = cache_after_prefill,
                                            use_cache = True
                                            )
        updated_K, updated_V = updated_cache
        outputs_close = torch.allclose(expected, cached_output, atol=1e-5, rtol=1e-5)
        max_absolute_diff = torch.max(torch.abs(expected - cached_output)).item()
    
    print("max absolute difference:", max_absolute_diff)

    assert past_K.shape ==(1,2,3,4)
    assert past_V.shape == (1,2,3,4)
    assert updated_K.shape == (1,2,4,4) 
    assert updated_V.shape == (1,2,4,4)
    assert outputs_close 


def test_transformer_lm_kv_cache():
    torch.manual_seed(0)

    model = TransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=8,
        num_layers=2,
        num_heads=2,
        d_ff=16,
        rope_theta=10000,
        device="cpu",
        dtype=torch.float32
    )

    model.eval()

    token_ids = torch.randint(low=0, high=32, size=(1,4), dtype=torch.long)

    with torch.inference_mode():
        full_logits = model(token_ids)
        expected = full_logits[:,3:4,:]
        prefill_logits, cache_after_prefill = model(token_ids[:,:3], use_cache=True)    
        assert len(cache_after_prefill) == 2
        for layer_cache in cache_after_prefill:
            past_K, past_V = layer_cache
            assert past_K.shape ==(1,2,3,4)
            assert past_V.shape == (1,2,3,4)
        cached_logits, updated_cache = model(token_ids[:,3:4],
                                            past_key_values = cache_after_prefill,
                                            use_cache = True
                                            )
        assert len(updated_cache) == 2
        for layer_cache in updated_cache:
            updated_K, updated_V = layer_cache
            assert updated_K.shape == (1,2,4,4) 
            assert updated_V.shape == (1,2,4,4)
        outputs_close = torch.allclose(expected, cached_logits, atol=1e-5, rtol=1e-5)
        max_absolute_diff = torch.max(torch.abs(expected - cached_logits)).item()
    
    print("max absolute difference:", max_absolute_diff)

    assert outputs_close 


def make_test_model(context_length=16):
    torch.manual_seed(0)

    model = TransformerLM(
        vocab_size=32,
        context_length=context_length,
        d_model=8,
        num_layers=2,
        num_heads=2,
        d_ff=16,
        rope_theta=10000,
        device="cpu",
        dtype=torch.float32,
    )
    model.eval()
    return model


def test_decoding_kv_cache_matches_uncached_greedy():
    model = make_test_model()
    prompt_ids = [1, 2, 3, 4]

    uncached_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=16,
        max_new_tokens=8,
        temperature=0,
        top_p=1.0,
    )
    cached_ids = generate_token_ids_cached(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=16,
        max_new_tokens=8,
        temperature=0,
        top_p=1.0,
    )
    repeated_cached_ids = generate_token_ids_cached(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=16,
        max_new_tokens=8,
        temperature=0,
        top_p=1.0,
    )

    assert cached_ids == uncached_ids
    assert repeated_cached_ids == cached_ids
    assert len(cached_ids) == len(prompt_ids) + 8


def test_decoding_kv_cache_zero_tokens_and_eos():
    model = make_test_model()
    prompt_ids = [1, 2, 3, 4]

    zero_token_result = generate_token_ids_cached(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=16,
        max_new_tokens=0,
        temperature=0,
        top_p=1.0,
    )
    assert zero_token_result == prompt_ids

    first_step = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=16,
        max_new_tokens=1,
        temperature=0,
        top_p=1.0,
    )
    first_generated_id = first_step[-1]

    uncached_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=first_generated_id,
        context_length=16,
        max_new_tokens=8,
        temperature=0,
        top_p=1.0,
    )
    cached_ids = generate_token_ids_cached(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=first_generated_id,
        context_length=16,
        max_new_tokens=8,
        temperature=0,
        top_p=1.0,
    )

    assert cached_ids == uncached_ids
    assert len(cached_ids) == len(prompt_ids) + 1


def test_decoding_kv_cache_context_boundary():
    model = make_test_model(context_length=8)
    prompt_ids = [1, 2, 3, 4, 5, 6, 7]

    uncached_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=8,
        max_new_tokens=2,
        temperature=0,
        top_p=1.0,
    )
    cached_ids = generate_token_ids_cached(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=8,
        max_new_tokens=2,
        temperature=0,
        top_p=1.0,
    )

    assert cached_ids == uncached_ids

    with pytest.raises(ValueError):
        generate_token_ids_cached(
            model=model,
            prompt_ids=prompt_ids,
            eos_id=-1,
            context_length=8,
            max_new_tokens=3,
            temperature=0,
            top_p=1.0,
        )
