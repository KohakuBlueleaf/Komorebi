"""Fails when a pytest run skipped another number of tests than the workflow expects.
Usage: skip_count.py JUNIT.xml EXPECTED
JUNIT.xml is pytest's --junitxml report; a planned skip is a change of EXPECTED.
"""

import sys
import xml.etree.ElementTree as ET


def skipped(report):
    """The number of skipped tests in a pytest junit report."""
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else root.iter("testsuite")
    return sum(int(suite.get("skipped", 0)) for suite in suites)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    report, expected = sys.argv[1], int(sys.argv[2])
    count = skipped(report)
    print(f"{report}: {count} skipped, {expected} expected")
    if count != expected:
        sys.exit(
            f"{count} skipped where the workflow expects {expected}: a new skip is a "
            "test that stopped running; a planned one changes the expected number"
        )


if __name__ == "__main__":
    main()
