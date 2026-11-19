"""Decoded output classes of the skill-check classifier.

The ONNX model returns 11 logits. This module is the single place that maps a
class index to its meaning and to the decision the bot must take, so the model
contract lives in one file instead of being duplicated across the codebase.

Class semantics (from the training set):

===========  ==================================  =====
index        description                         hit
===========  ==================================  =====
0            nothing on screen                   no
1            repair-heal, needle in great zone   yes
2            repair-heal, needle ante-frontier   yes
3            repair-heal, needle out of the zone no
4            full white, needle in great zone    yes
5            full white, needle out of the zone  no
6            full black, needle in great zone    yes
7            full black, needle out of the zone  no
8            wiggle, needle in great zone        yes
9            wiggle, needle on the frontier      no
10           wiggle, needle out of the zone      no
===========  ==================================  =====
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    id: int
    desc: str
    hit: bool


UNKNOWN = Label(-1, "unknown", False)

#: Index of the "nothing on screen" class.
NONE_ID = 0

#: Index of the ante-frontier hit class, which needs an extra delay before the
#: key press because the needle has not reached the great zone yet.
ANTE_FRONTIER_ID = 2

#: Number of classes the model emits.
NUM_CLASSES = 11

LABELS = (
    Label(0, "None", False),
    Label(1, "repair-heal (great)", True),
    Label(2, "repair-heal (ante-frontier)", True),
    Label(3, "repair-heal (out)", False),
    Label(4, "full white (great)", True),
    Label(5, "full white (out)", False),
    Label(6, "full black (great)", True),
    Label(7, "full black (out)", False),
    Label(8, "wiggle (great)", True),
    Label(9, "wiggle (frontier)", False),
    Label(10, "wiggle (out)", False),
)

_BY_ID = {label.id: label for label in LABELS}


def label_for(class_id: int) -> Label:
    """Return the :class:`Label` for a model class index."""
    return _BY_ID.get(int(class_id), UNKNOWN)


def should_hit(class_id: int) -> bool:
    """Whether the bot must press the skill-check key for this class."""
    return label_for(class_id).hit
