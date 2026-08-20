# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for XML normalization."""

from safety.safety_benchmark import _normalize_to_atem_xml


class TestNormalizeXml:
    def test_atml_prefix(self):
        result = _normalize_to_atem_xml('<atml:invoke name="x">')
        assert '<atem:invoke name="x">' in result

    def test_atec_prefix(self):
        result = _normalize_to_atem_xml('<atec:invoke name="x">')
        assert '<atem:invoke name="x">' in result

    def test_antem_prefix(self):
        result = _normalize_to_atem_xml('<antem:invoke name="x">')
        assert '<atem:invoke name="x">' in result

    def test_function_tag_conversion(self):
        xml = '<invoke><function>mobile.tap</function><parameter name="x">100</parameter></invoke>'
        result = _normalize_to_atem_xml(xml)
        assert 'name="mobile.tap"' in result

    def test_inline_attr_expansion(self):
        xml = '<atem:invoke name="mobile.tap" x="355" y="770"></atem:invoke>'
        result = _normalize_to_atem_xml(xml)
        assert '<atem:parameter name="x">355</atem:parameter>' in result
        assert '<atem:parameter name="y">770</atem:parameter>' in result

    def test_kwargs_expansion(self):
        xml = """<atem:invoke name="mobile.tap" kwargs='{"x": 456, "y": 612}'></atem:invoke>"""
        result = _normalize_to_atem_xml(xml)
        assert '<atem:parameter name="x">456</atem:parameter>' in result

    def test_bare_tags(self):
        xml = '<function_calls><invoke name="mobile.tap"></invoke></function_calls>'
        result = _normalize_to_atem_xml(xml)
        assert "<atem:function_calls>" in result
        assert "<atem:invoke" in result
