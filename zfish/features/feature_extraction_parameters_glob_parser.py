# %%
import fnmatch
from itertools import product
from typing import Collection


def main():
    available_channels = (
        [f"DAPI.{i}" for i in range(2)]
        + [f"PCNA.{i}" for i in range(2)]
        + [f"Pol-II-S2P.{i}" for i in range(2)]
        + [f"Pol-II-S5P.{i}" for i in range(2)]
    )
    def test_star_expression_by_name():
        assert _parse_star_expression(["DAPI.1"], available_channels) == set(["DAPI.1"])
        assert _parse_star_expression(["DAPI.1", "PCNA.0"], available_channels) == set(
            ["DAPI.1", "PCNA.0"]
        )

    def test_star_expression_negate():
        assert _parse_star_expression(["!DAPI.1"], available_channels) == set(
            [e for e in available_channels if e != "DAPI.1"]
        )
        assert _parse_star_expression(["!DAPI.1", "!PCNA.0"], available_channels) == set(
            [e for e in available_channels if e not in ["DAPI.1", "PCNA.0"]]
        )

    def test_star_expression_wildcards():
        assert _parse_star_expression(["*"], available_channels) == set(available_channels)
        assert _parse_star_expression(["DAPI.*"], available_channels) == set(
            ["DAPI.0", "DAPI.1"]
        )
        assert _parse_star_expression(["*-*"], available_channels) == set(
            [e for e in available_channels if e.startswith("Pol-II-S")]
        )
        assert _parse_star_expression(["Pol-II-S[25]P.*"], available_channels) == set(
            [e for e in available_channels if e.startswith("Pol-II-S")]
        )

    def test_star_expression_negate_wildcards():
        assert _parse_star_expression(["!*"], available_channels) == set([])
        assert _parse_star_expression(["!DAPI.*"], available_channels) == set(
            [e for e in available_channels if e not in ["DAPI.0", "DAPI.1"]]
        )
        assert _parse_star_expression(["!Pol-II-S[25]P.*"], available_channels) == set(
            [e for e in available_channels if not e.startswith("Pol-II-S")]
        )
        assert _parse_star_expression(["!DAPI.*", "!PCNA.*"], available_channels) == set(
            [e for e in available_channels if e.startswith("Pol-II-S")]
        )
        
    test_star_expression_by_name()
    test_star_expression_negate()
    test_star_expression_wildcards()
    test_star_expression_negate_wildcards()


def _parse_star_expression(
    channel_expressions: Collection[str],
    available_channels: Collection[str],
    negation_char: str = "!",
) -> set[str]:
    channel_expressions = set(channel_expressions)
    available_channels = set(available_channels)
    matching_channels = []
    exclude_flag = False

    # Check where its exclude expression(s).
    if any([ch.startswith(negation_char) for ch in channel_expressions]):
        assert all(
            [ch.startswith(negation_char) for ch in channel_expressions]
        ), f"All channels have to start with `{negation_char}` if any channel starts with `{negation_char}`."
        exclude_flag = True
        # Remove leading ! of exlude expression
        channel_expressions = [ch[1:] for ch in channel_expressions]

    for channel_expression in channel_expressions:
        # Expand glob expression
        match = fnmatch.filter(available_channels, channel_expression)
        # Raise upon non-matching expression
        if len(match) == 0:
            raise RuntimeError(
                f"Non-matching expression `{channel_expression}` for available channels: {sorted(available_channels)}"
            )
        matching_channels.extend(match)

    if exclude_flag:
        return set([ch for ch in available_channels if ch not in matching_channels])
    return set(matching_channels)


def _parse_pairwise_star_expression(
    channel_expression_pairs: Collection[tuple[str, str]],
    available_channels: Collection[str],
    negation_char: str = "!",
) -> set[tuple[str, str]]:
    out_pairs = []
    for channel_expression1, channel_expression2 in channel_expression_pairs:
        matching_channels1 = _parse_star_expression(
            channel_expressions=[channel_expression1],
            available_channels=available_channels,
            negation_char=negation_char,
        )
        matching_channels2 = _parse_star_expression(
            channel_expressions=[channel_expression2],
            available_channels=available_channels,
            negation_char=negation_char,
        )
        out_pairs.extend([p for p in product(matching_channels1, matching_channels2)])
    return set(out_pairs)

# def _parse_star_expression(
#     channels: set[str],
#     available_channels: set[str],
# ) -> set[str]:
#     channels = set(channels)
#     available_channels = set(available_channels)
#     if channels == {"*"}:
#         return available_channels
#     if any([ch.startswith("!") for ch in channels]):
#         assert all(
#             [ch.startswith("!") for ch in channels]
#         ), "All channels have to start with `!` if any channel starts with `!`."
#         excluded_channels = [e[1:] for e in channels]
#         for ch in excluded_channels:
#             assert ch in available_channels, f"`{ch}` not in `{available_channels}`."
#         return set([ch for ch in available_channels if ch not in excluded_channels])
#     else:
#         for ch in channels:
#             assert ch in available_channels, f"`{ch}` not in `{available_channels}`."
#         return channels


# def _parse_pairwise_star_expression(
#     channel_pairs: set[tuple[str, str]],
#     available_channels: set[str],
# ) -> set[tuple[str, str]]:
#     out_pairs = set()
#     for c0, c1 in channel_pairs:
#         if (c0 == "*" or c0.startswith("!")) and (c1 == "*" or c1.startswith("!")):
#             out_pairs.update(
#                 [
#                     tuple(sorted(e))
#                     for e in product(
#                         _parse_star_expression(set([c0]), available_channels),
#                         _parse_star_expression(set([c1]), available_channels),
#                     )
#                 ]
#             )
#         elif c0 == "*" or c0.startswith("!"):
#             out_pairs.update(
#                 [
#                     tuple(sorted(e))
#                     for e in zip(
#                         _parse_star_expression(set([c0]), available_channels),
#                         repeat(c1),
#                     )
#                 ]
#             )
#         elif c1 == "*" or c1.startswith("!"):
#             out_pairs.update(
#                 [
#                     tuple(sorted(e))
#                     for e in zip(
#                         repeat(c0),
#                         _parse_star_expression(set([c1]), available_channels),
#                     )
#                 ]
#             )
#         else:
#             out_pairs.add(tuple(sorted((c0, c1))))
#     return out_pairs