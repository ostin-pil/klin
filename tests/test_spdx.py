"""The SPDX register lookup.

No test here touches the network. Every fetcher is injected, which is also the
point of the seam: a register lookup that silently fell back to the real
internet in CI would pass for reasons unrelated to the code.
"""

import pytest

from klin import spdx

REGISTER = {
    "licenses": [
        {"licenseId": "Apache-2.0"},
        {"licenseId": "MIT"},
        {"licenseId": "CC0-1.0"},
    ]
}


@pytest.fixture(autouse=True)
def forget_the_register():
    spdx.reset()
    yield
    spdx.reset()


def a_register(url):
    assert url == spdx.LIST_URL
    return REGISTER


def test_a_vendor_spelling_resolves_to_the_registers_own():
    """The Hub tags Apache as `apache-2.0`; SPDX writes `Apache-2.0`. The text
    URL only resolves under the register's spelling."""
    assert spdx.canonical("apache-2.0", get_json=a_register) == "Apache-2.0"
    assert spdx.canonical("  MIT  ", get_json=a_register) == "MIT"


def test_something_outside_the_register_resolves_to_nothing():
    """A LicenseRef is not an identifier SPDX knows, and inventing a text for
    it is the guess this module exists to avoid."""
    assert spdx.canonical("LicenseRef-Civitai-648058", get_json=a_register) is None
    assert spdx.canonical("openrail++", get_json=a_register) is None
    assert spdx.canonical(None, get_json=a_register) is None


def test_the_register_is_fetched_once_per_process():
    calls = []

    def counted(url):
        calls.append(url)
        return REGISTER

    spdx.canonical("MIT", get_json=counted)
    spdx.canonical("Apache-2.0", get_json=counted)
    assert len(calls) == 1


def test_an_unreachable_register_yields_nothing_rather_than_raising():
    """Offline is not an error. The field stays empty, the project's own rule
    reports it, and nothing has been invented."""

    def broken(url):
        raise IOError("no network")

    assert spdx.canonical("MIT", get_json=broken) is None


def test_text_comes_back_with_the_url_it_came_from():
    got_url, body = spdx.text(
        "apache-2.0", get_json=a_register, get_text=lambda url: "APACHE TERMS"
    )
    assert got_url == "https://spdx.org/licenses/Apache-2.0.txt"
    assert body == "APACHE TERMS"


def test_an_unrecognised_identifier_fetches_no_text_at_all():
    def never(url):
        raise AssertionError("should not have been asked for %s" % url)

    assert spdx.text("openrail++", get_json=a_register, get_text=never) == (None, None)


@pytest.mark.parametrize("body", ["", "   ", None])
def test_an_empty_body_is_not_licence_text(body):
    assert spdx.text(
        "MIT", get_json=a_register, get_text=lambda url: body
    ) == (None, None)
