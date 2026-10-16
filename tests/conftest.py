import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from enefit import config as C  # noqa: E402


def _have_data() -> bool:
    return all(p.exists() for p in C.RAW_FILES.values()) and C.COUNTY_WEIGHTS_FILE.exists()


requires_data = pytest.mark.skipif(not _have_data(), reason="raw Kaggle data not present (data/ is not in git)")


@pytest.fixture(scope="session")
def raw():
    if not _have_data():
        pytest.skip("raw Kaggle data not present")
    from enefit import data
    return data.load_raw_tables()


# days inside the test block (2023-03-01..04-15) and the live-run block (2023-04-16..05-31)
SCORING_DAYS = ["2023-03-10", "2023-04-05", "2023-04-20", "2023-05-20"]
