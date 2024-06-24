from functools import singledispatch

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator, SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp
from outlines.models.mlxlm import MLXLM
from outlines.models.vllm import VLLM
from outlines.samplers import Sampler, multinomial


@singledispatch
def boost(model, sampler: Sampler = multinomial()):
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the
    regular expression.

    """
    fsm = RegexGuide("(\S*[\u3131-\u314e|\u314f-\u3163|\uac00-\ud7a3]+\S*)", model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@boost.register(MLXLM)
def boost_mlxlm(
    model: MLXLM,
    sampler: Sampler = multinomial(),
):
    from outlines.processors import BoostLogitsProcessor

    logits_processor = BoostLogitsProcessor("(\S*[\u3131-\u314e|\u314f-\u3163|\uac00-\ud7a3]+\S*)", tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@boost.register(LlamaCpp)
def boost_llamacpp(
    model: LlamaCpp,
    sampler: Sampler = multinomial(),
):
    from outlines.integrations.llamacpp import BoostLogitsProcessor

    logits_processor = BoostLogitsProcessor("(\S*[\u3131-\u314e|\u314f-\u3163|\uac00-\ud7a3]+\S*)", llm=model.model)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@boost.register(VLLM)
def boost_vllm(
    model: VLLM,
    sampler: Sampler = multinomial(),
):
    from outlines.integrations.vllm import BoostLogitsProcessor

    logits_processor = BoostLogitsProcessor("(\S*[\u3131-\u314e|\u314f-\u3163|\uac00-\ud7a3]+\S*)", model.model)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@boost.register(OpenAI)
def boost_openai(
    model: OpenAI,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use regex-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
