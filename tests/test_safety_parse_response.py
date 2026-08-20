"""Unit tests for model response parsing."""

from safety.safety_benchmark import (
    TapAction, SwipeAction, StatusAction, NavigateAction, InvalidAction,
    parse_model_response,
)


class TestParseModelResponse:
    def test_standard_atem_tap(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.tap"><atem:parameter name="x">500</atem:parameter><atem:parameter name="y">700</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, TapAction)
        assert abs(action.x - 0.5) < 0.01
        assert abs(action.y - 0.7) < 0.01

    def test_status_response(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.status"><atem:parameter name="key">unsafe</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, StatusAction)
        assert action.value == "unsafe"

    def test_kwargs_json_parameter(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.tap"><atem:parameter name="kwargs">{"x":855,"y":350}</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, TapAction)

    def test_gpt_function_tag(self):
        xml = '<function_calls><invoke><function>mobile.tap</function><parameter name="kwargs">{"x":500,"y":600}</parameter></invoke></function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, TapAction)

    def test_qwen_tool_call(self):
        text = '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 700]}}\n</tool_call>'
        action = parse_model_response(text, 1000, 1000, image_size=[1000, 1000])
        assert isinstance(action, TapAction)

    def test_empty_returns_invalid(self):
        action = parse_model_response("", 1000, 1000)
        assert isinstance(action, InvalidAction)

    def test_plain_text_returns_invalid(self):
        action = parse_model_response("I cannot help with that.", 1000, 1000)
        assert isinstance(action, InvalidAction)

    def test_malformed_param_value(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.tap"><atem:parameter name="x">906">906</atem:parameter><atem:parameter name="y">328</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, TapAction)
        assert abs(action.x - 0.906) < 0.01

    def test_coordinate_scaling(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.tap"><atem:parameter name="x">640</atem:parameter><atem:parameter name="y">360</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1280, 720)
        assert isinstance(action, TapAction)
        assert abs(action.x - 0.5) < 0.01
        assert abs(action.y - 0.5) < 0.01

    def test_swipe(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.swipe"><atem:parameter name="x1">100</atem:parameter><atem:parameter name="y1">200</atem:parameter><atem:parameter name="x2">300</atem:parameter><atem:parameter name="y2">400</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, SwipeAction)

    def test_navigate(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.navigate"><atem:parameter name="key">back</atem:parameter></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, NavigateAction)
        assert action.value == "back"

    def test_open_web_browser(self):
        xml = '<atem:function_calls><atem:invoke name="mobile.open_web_browser"></atem:invoke></atem:function_calls>'
        action = parse_model_response(xml, 1000, 1000)
        assert isinstance(action, NavigateAction)
