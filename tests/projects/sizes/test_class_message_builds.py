"""The classes whose handlers carry messages build on a mock device (no card): each
script runs to its first load and its five kernels are compiled, as
test_kernel_sizes.py builds the other classes. KMR_SIZES=1 and TT_METAL_HOME."""

import pytest
from kernel_sizes_fixture import HEADROOM, build_sizes

pytestmark = pytest.mark.sizes

CASES = {  # a km.cxx body naming the handler's km::Messages; several dynamic handlers
    "class_forward": ["examples/05_forward/as_class/forward_class.py"],
    "class_message_handlers": [
        "examples/14_message_handlers/as_class/handlers_class.py"
    ],
}


@pytest.mark.parametrize("name", list(CASES))
def test_a_class_with_messages_builds(environment, name):
    row = build_sizes(environment, CASES[name])
    assert row["fits"] and row["total"] + HEADROOM <= row["limit"], row
