"""The matching judge prompt and its SIMILAR / MAYBE / DIFFERENT scoring."""

import asyncio
import random
from typing import Literal, TypeAlias

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from aspd.eval.matching.examples import Example, format_feature_examples

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]

SYSTEM_PROMPT = """
    ### TASK:
    We're studying a pair of features in a neural network.
    You will rate whether the feature in the pairs are similar, 3 possible outputs are: SIMILAR / MAYBE / DIFFERENT.

    ### MEANING OF FEATURE and FEATURE CAPTIONING:
    Each feature activates on some particular word/words/substring/concept in a short document.
    The activating words in each document are indicated with << ... >>.
    We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating.
    Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on.
    Try not to be overly specific in your explanation.
    Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept.
    Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate).
    Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant.
    Keep the explanation as short and simple as possible, limited to 20 words or less.
    Omit punctuation and formatting.
    You should avoid giving long lists of words.
    Some examples: "This feature activates on the word 'knows' in rhetorical questions", and "This feature activates on verbs related to decision-making and preferences", and "This feature activates on the substring 'Ent' at the start of words", and "This feature activates on text about government economic policy".

    ### INPUT FORMAT:
    "
    FEATURE 1:
    1. (example 1)
    2, ....

    FEATURE 2:
    ...
    "

    ### STEPS:
    1, First you need to caption the meaning of FEATURES.
    2, Determine if the two features have similar meaning or not. If any feature has "NO EXAMPLE.", you must output DIFFERENT.

    ### OUTPUT FORMAT:
    "
        FEATURE CAPTION:

    Feature 1: (caption)
    Feature 2: (caption)

    =====================
        ANSWER: SIMILAR / MAYBE / DIFFERENT
    "
    """


def _side(examples: list[Example]) -> str:
    """Upstream builds this by looping over a list of features per side; our matching is
    one-to-one, so the loop has exactly one iteration and `Feature 1:` is its label.
    """
    return "\nFeature 1: \n" + format_feature_examples(examples)


def get_generation_prompts(feat_a: list[Example], feat_b: list[Example]) -> Messages:
    feature_a_header = f"""
    FEATURE 1:
    {_side(feat_a)}
    """
    feature_b_header = f"""
    FEATURE 2:
    {_side(feat_b)}
    """
    user_prompt = (
        f"""The activating documents are given below:\n\n{feature_a_header + feature_b_header}"""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_matching_predictions(response: str) -> int:
    """Verbatim upstream. `SIMILAR->3`, `MAYBE->2`, `DIFFERENT->1`, anything else -> 1."""
    score = 1
    answer_block = response.split("ANSWER")[-1].strip()

    parts = answer_block.split(":")
    if len(parts) > 1:
        ans = parts[1].strip().lower()
        if "similar" in ans or '"similar"' in ans:
            score = 3
        elif "maybe" in ans or '"maybe"' in ans:
            score = 2
        elif "different" in ans or '"different"' in ans:
            score = 1
        else:
            score = 1

    return score


async def _one(
    client: AsyncOpenAI,
    messages: Messages,
    sem: asyncio.Semaphore,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> str:
    attempt, base_backoff = 0, 0.5
    while True:
        attempt += 1
        try:
            async with sem:
                result = await client.chat.completions.create(
                    model=model, messages=messages, stream=False, max_tokens=max_tokens
                )  # type: ignore[arg-type]
            content = result.choices[0].message.content
            return content.strip() if content else ""
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt >= max_retries:
                raise
            await asyncio.sleep(base_backoff * (2 ** (attempt - 1)) * (0.5 + random.random()))
        except APIError as e:
            if getattr(e, "status_code", 500) >= 500 and attempt < max_retries:
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)) * (0.5 + random.random()))
            else:
                raise


async def judge_pairs(
    prompts: list[Messages],
    *,
    base_url: str,
    api_key: str = "EMPTY",
    model: str,
    concurrency: int = 64,
    max_tokens: int = 10000,
    timeout: float = 180.0,
) -> list[int]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    responses = await asyncio.gather(
        *[_one(client, m, sem, model, max_tokens, 5) for m in prompts]
    )
    return [parse_matching_predictions(r) for r in responses]
