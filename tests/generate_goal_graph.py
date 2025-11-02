import argparse
import json
import os

import torch

from tests.generate_standard_inputs import generate_standard_inputs
from tests.generate_standard_lm import generate_standard_lm
from tests.generate_switch_head_lm import (
    generate_switch_head_lm_single_expert,
    generate_switch_head_multiple_experts_lm,
)
from tests.graph_utils import get_computation_graph


def _setup_model_and_inputs(lm_generator=generate_standard_lm):
    inputs = generate_standard_inputs()
    model = lm_generator()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        inputs, None, None, None
    )
    embedding.requires_grad_(True)
    return model, embedding, attention_mask, sequence_length


def get_basic_attention_tensor():
    model, embedding, attention_mask, sequence_length = _setup_model_and_inputs()
    torch.manual_seed(42)
    attention = model.transformer.body_layers[0].attention

    attention_output, _ = attention(
        q_src=embedding,
        k_src=embedding,
        v_src=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector={},
    )
    return attention_output


def get_basic_ffn_tensor():
    model, embedding, _, _ = _setup_model_and_inputs()
    torch.manual_seed(42)

    ffn = model.transformer.body_layers[0].ffn
    ffn_output, _ = ffn(embedding, None, {})

    return ffn_output


def get_sigma_moe_tensor():
    model, embedding, attention_mask, sequence_length = _setup_model_and_inputs()
    torch.manual_seed(42)

    sigma_moe_layer = model.transformer.body_layers[1]
    sigma_moe_output, _, _, _ = sigma_moe_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=1,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector={},
    )

    return sigma_moe_output


def get_switch_head_multiple_experts_tensor():
    model = generate_switch_head_multiple_experts_lm()
    inputs = generate_standard_inputs()
    embedding, attention_mask, sequence_length = model.embedding_stage(
        inputs, None, None, None
    )
    embedding.requires_grad_(True)

    model = model.to("cuda")  # type: ignore
    embedding = embedding.to("cuda")
    attention_mask = attention_mask.to("cuda")
    sequence_length = sequence_length.to("cuda")

    torch.manual_seed(42)

    switch_head_layer = model.transformer.head_layers[0]
    switch_head_layer.eval()

    switch_head_output, _, _, _ = switch_head_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=0,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector={},
    )

    return switch_head_output


def get_switch_head_single_expert_tensor():
    model = generate_switch_head_lm_single_expert()
    inputs = generate_standard_inputs()
    embedding, attention_mask, sequence_length = model.embedding_stage(
        inputs, None, None, None
    )
    embedding.requires_grad_(True)

    model = model.to("cuda")  # type: ignore
    embedding = embedding.to("cuda")
    attention_mask = attention_mask.to("cuda")
    sequence_length = sequence_length.to("cuda")

    torch.manual_seed(42)

    switch_head_layer = model.transformer.head_layers[0]
    switch_head_layer.eval()

    switch_head_output, _, _, _ = switch_head_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=0,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector={},
    )

    return switch_head_output


def get_transformer_tensor():
    model, embedding, attention_mask, sequence_length = _setup_model_and_inputs()
    torch.manual_seed(42)

    transformer_output, _ = model.transformer(
        x=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        e=embedding.clone(),
        input_ids=generate_standard_inputs(),
        collector={},
    )

    return transformer_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate graphs for model components."
    )
    parser.add_argument(
        "component", type=str, help="The component to generate the graph for."
    )
    args = parser.parse_args()

    tensor_generators = {
        "basic_attention": get_basic_attention_tensor,
        "basic_ffn": get_basic_ffn_tensor,
        "sigma_moe": get_sigma_moe_tensor,
        "switch_head_multiple_experts": get_switch_head_multiple_experts_tensor,
        "switch_head_single_expert": get_switch_head_single_expert_tensor,
        "transformer": get_transformer_tensor,
    }

    if args.component in tensor_generators:
        output_tensor = tensor_generators[args.component]()

        graph = get_computation_graph(output_tensor)

        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, "graph_jsons")
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"standard_{args.component}_graph.json"
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, "w") as f:
            json.dump(graph, f, indent=2)

        print(
            f"Successfully generated and saved the {args.component} "
            f"computation graph to {output_path}."
        )
    else:
        print(
            f"Error: Unknown component '{args.component}'. "
            f"Supported components are: {list(tensor_generators.keys())}"
        )


if __name__ == "__main__":
    main()
