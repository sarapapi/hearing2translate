import os
import unittest
from unittest.mock import patch, MagicMock
import argparse
import json

from infer import read_jsonl, setup_model, load_prompt, get_model_input


class TestInference(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.inf_dir = os.path.join(self.test_dir, "..", "inference")
        self.args = argparse.Namespace()
        self.args.model = "phi4multimodal"
        self.args.in_modality = "speech"
        self.args.in_file = "dummy_in.jsonl"
        self.args.out_file = "dummy.jsonl"

    def test_read_in_out(self):
        in_data = [
            {"dataset_id": 0, "sample_id": 0, "src_audio": "fake.wav", "src_ref": "dummy",
             "tgt_ref": "dummy", "src_lang": "en", "tgt_lang": "it",
             "benchmark_metadata": {"context": "short"}},
            {"dataset_id": 0, "sample_id": 1, "src_audio": None, "src_ref": "dummy",
             "tgt_ref": "dummy", "src_lang": "en", "tgt_lang": "it",
             "benchmark_metadata": {"context": "short"}},
        ]
        out_data = [
            {"dataset_id": 0, "sample_id": 0, "src_lang": "en", "tgt_lang": "it", "model": "greatsys",
             "output": "Output 1"},
            {"dataset_id": 0, "sample_id": 1, "src_lang": "en", "tgt_lang": "it", "model": "greatsys",
             "output": "Output 2"}
        ]

        for in_data, in_loaded_data, out_data, out_loaded_data in zip(
                in_data, read_jsonl("dummy_in.jsonl"), out_data, read_jsonl("dummy_out.jsonl")):
            self.assertEqual(in_data, in_loaded_data)
            self.assertEqual(out_data, out_loaded_data)

    @patch('importlib.import_module')
    def test_load_model(self, mock_import):
        # Mock the module and its functions
        mock_module = MagicMock()
        mock_module.generate = MagicMock()
        mock_module.load_model = MagicMock(return_value="mock_model")
        mock_import.return_value = mock_module

        model, generate_func = setup_model("phi4multimodal")

        mock_import.assert_called_once_with("inference.speechllm.phi4multimodal")
        self.assertEqual(model, "mock_model")
        self.assertEqual(generate_func, mock_module.generate)

    def test_load_unsupported_model(self):
        with self.assertRaises(NotImplementedError) as context:
            setup_model("unsupported_model")
        self.assertIn("currently not supported", str(context.exception))

    @patch('importlib.import_module')
    def test_load_model_missing_generate(self, mock_import):
        mock_module = MagicMock()
        del mock_module.generate  # Remove generate attribute
        mock_import.return_value = mock_module

        with self.assertRaises(ImportError) as context:
            setup_model("phi4multimodal")
        self.assertIn("does not define `generate`", str(context.exception))

    @patch('importlib.import_module')
    def test_load_model_missing_load_func(self, mock_import):
        mock_module = MagicMock()
        del mock_module.load_model  # Remove load function
        mock_import.return_value = mock_module

        with self.assertRaises(ImportError) as context:
            setup_model("phi4multimodal")
        self.assertIn("does not define `load_model`", str(context.exception))

    def test_load_prompt(self):
        enit_speech_prompt = load_prompt("speech", "en", "it")
        enit_text_prompt = load_prompt("text", "en", "it")

        self.assertEqual(
            enit_text_prompt, "You are a professional English-to-Italian translator. Your goal is "
                              "to accurately convey the meaning and nuances of the original "
                              "English text while adhering to Italian grammar, vocabulary, and "
                              "cultural sensitivities. Preserve the line breaks. Use precise "
                              "terminology and a tone appropriate for academic or instructional "
                              "materials. Produce only the Italian translation, without any "
                              "additional explanations or commentary. Please translate the "
                              "provided English text into Italian:")
        self.assertEqual(
            enit_speech_prompt, "You are a professional English-to-Italian translator. Your goal "
                                "is to accurately convey the meaning and nuances of the original "
                                "English speech while adhering to Italian grammar, vocabulary, and"
                                " cultural sensitivities. Use precise terminology and a tone "
                                "appropriate for academic or instructional materials. Produce only"
                                " the Italian translation, without any additional explanations or "
                                "commentary. Please translate the provided English speech into "
                                "Italian:")

        with self.assertRaises(ValueError):
            load_prompt("speech", "lv", "it")

    def test_get_model_in(self):
        transcripts = {}
        with open("dummy_out.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Use a tuple of (dataset_id, sample_id) as the key
                key = (entry["dataset_id"], entry["sample_id"])
                transcripts[key] = entry["output"]  # Store the 'output' field as the value

        speech_inputs = []
        for sample in read_jsonl("dummy_in.jsonl"):
            speech_inputs.append(get_model_input("speech", sample, None))
        self.assertEqual(speech_inputs, ["fake.wav", None])

        text_inputs = []
        for sample in read_jsonl("dummy_in.jsonl"):
            text_inputs.append(get_model_input("text", sample, transcripts))
        self.assertEqual(text_inputs, ["Output 1", "Output 2"])

        with self.assertRaises(ValueError):
            text_inputs = []
            for sample in read_jsonl("dummy_in.jsonl"):
                text_inputs.append(get_model_input("text", sample, None))


if __name__ == '__main__':
    unittest.main()
