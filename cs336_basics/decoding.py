import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    # logits shape: (vocab_size,)

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return torch.argmax(logits).item()

    scaled_logits = logits.float() / temperature

    probabilities = torch.softmax(
        scaled_logits,
        dim=-1,
    )

    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        probability_before_token = (
            cumulative_probabilities - sorted_probabilities
        )

        remove_mask = probability_before_token >= top_p

        sorted_probabilities = sorted_probabilities.masked_fill(
            remove_mask,
            0.0,
        )

        sorted_probabilities = (
            sorted_probabilities
            / sorted_probabilities.sum()
        )

        sampled_position = torch.multinomial(
            sorted_probabilities,
            num_samples=1,
        )

        sampled_token_id = sorted_indices[
            sampled_position
        ]

        return sampled_token_id.item()

    sampled_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return sampled_token.item()




def generate_token_ids(
    model: torch.nn.Module,
    prompt_ids: list[int],
    eos_id: int,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated_ids = list(prompt_ids)

    device = next(model.parameters()).device
    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-context_length:]

            input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
            )

            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]

            next_token_id = sample_next_token(next_token_logits,
                                              temperature,
                                              top_p
                                              )
            generated_ids.append(next_token_id)

            if next_token_id == eos_id:
                break

    return generated_ids


def generate_token_ids_cached(
    model: torch.nn.Module,
    prompt_ids: list[int],
    eos_id: int,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated_ids = list(prompt_ids)

    device = next(model.parameters()).device
    model.eval()

    if max_new_tokens==0:
        return generated_ids
    if len(prompt_ids) + max_new_tokens - 1 > context_length:
        raise ValueError(
            "prompt length plus max_new_tokens - 1 must not exceed context_length"
        )
    
    with torch.inference_mode():
        context_ids = generated_ids[-context_length:]

        input_ids = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=device,
        )

        logits, past_key_values = model(input_ids, use_cache=True)
        next_token_logits = logits[0, -1, :]

        next_token_id = sample_next_token(next_token_logits,
                                            temperature,
                                            top_p
                                            )
        generated_ids.append(next_token_id)

        if next_token_id == eos_id:
            return generated_ids
        for _ in range(max_new_tokens-1):
            input_ids = torch.tensor(
                [[generated_ids[-1]]],
                dtype=torch.long,
                device=device
                )
            logits, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache = True
                )
            next_token_logits = logits[0,-1,:]
            next_token_id = sample_next_token(next_token_logits,
                                              temperature,
                                              top_p
                                              )
            generated_ids.append(next_token_id)

            if next_token_id == eos_id:
                break
        return generated_ids


    
    
