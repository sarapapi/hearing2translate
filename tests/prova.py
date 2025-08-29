import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock, mock_open
import sys
from io import StringIO

# Import the module to test
# Assuming your main script is saved as 'inference_main.py'
# You may need to adjust the import based on your actual file structure
try:
    from inference_main import (
        load_model, load_prompt, read_jsonl, write_jsonl, main,
        MODELS, MODEL_MODULES
    )
except ImportError:
    # If the above doesn't work, you might need to adjust the import path
    import inference_main

    load_model = inference_main.load_model
    load_prompt = inference_main.load_prompt
    read_jsonl = inference_main.read_jsonl
    write_jsonl = inference_main.write_jsonl
    main = inference_main.main
    MODELS = inference_main.MODELS
    MODEL_MODULES = inference_main.MODEL_MODULES


class TestLoadPrompt(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.inference_dir = os.path.join(self.temp_dir, "inference")
        os.makedirs(self.inference_dir)

        # Create mock prompt file
        self.prompt_content = "Translate from {src_lang} to {ref_lang}: {input}"
        self.prompt_file = os.path.join(self.inference_dir, "speech_prompt.txt")
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write(self.prompt_content)

        # Create mock language mapping
        self.lang_mapping = {
            "en": "English",
            "es": "Spanish",
            "fr": "French"
        }
        self.mapping_file = os.path.join(self.inference_dir, "language_mapping.json")
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.lang_mapping, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_prompt_success(self, mock_file, mock_join):
        # Mock file paths
        mock_join.side_effect = lambda *args: "/".join(args)

        # Mock file contents
        prompt_content = "Translate from {src_lang} to {ref_lang}"
        lang_mapping = {"en": "English", "es": "Spanish"}

        mock_file.side_effect = [
            mock_open(read_data=prompt_content).return_value,
            mock_open(read_data=json.dumps(lang_mapping)).return_value
        ]

        result = load_prompt("speech", "en", "es")

        expected = "Translate from English to Spanish"
        self.assertEqual(result, expected)

    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_prompt_missing_language(self, mock_file, mock_join):
        mock_join.side_effect = lambda *args: "/".join(args)

        prompt_content = "Translate from {src_lang} to {ref_lang}"
        lang_mapping = {"en": "English"}  # Missing 'zh'

        mock_file.side_effect = [
            mock_open(read_data=prompt_content).return_value,
            mock_open(read_data=json.dumps(lang_mapping)).return_value
        ]

        result = load_prompt("text", "en", "zh")

        # Should use original code if not in mapping
        expected = "Translate from English to zh"
        self.assertEqual(result, expected)


class TestMain(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create test input file
        self.input_data = [
            {
                "dataset_id": "test1",
                "sample_id": "sample1",
                "src_lang": "en",
                "tgt_lang": "es",
                "src_audio": "audio_path.wav"
            },
            {
                "dataset_id": "test2",
                "sample_id": "sample2",
                "src_lang": "fr",
                "tgt_lang": "en"
            }
        ]

        self.input_file = os.path.join(self.temp_dir, "input.jsonl")
        write_jsonl(self.input_file, self.input_data)

        self.output_file = os.path.join(self.temp_dir, "output.jsonl")

        # Mock arguments
        self.mock_args = MagicMock()
        self.mock_args.model = "phi4multimodal"
        self.mock_args.in_modality = "speech"
        self.mock_args.in_file = self.input_file
        self.mock_args.out_file = self.output_file
        self.mock_args.transcript_file = None

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('inference_main.load_model')
    @patch('inference_main.load_prompt')
    @patch('inference_main.logging')
    def test_main_speech_modality(self, mock_logging, mock_load_prompt, mock_load_model):
        # Mock model and generate function
        mock_model = MagicMock()
        mock_generate = MagicMock(return_value="Generated translation")
        mock_load_model.return_value = (mock_model, mock_generate)
        mock_load_prompt.return_value = "Mock prompt"

        main(self.mock_args)

        # Verify model was loaded
        mock_load_model.assert_called_once_with("phi4multimodal")

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_file))

        # Check output content
        results = list(read_jsonl(self.output_file))
        self.assertEqual(len(results), 2)

        for result in results:
            self.assertIn("dataset_id", result)
            self.assertIn("sample_id", result)
            self.assertIn("src_lang", result)
            self.assertIn("tgt_lang", result)
            self.assertIn("output", result)
            self.assertEqual(result["output"], "Generated translation")

    @patch('inference_main.load_model')
    @patch('inference_main.load_prompt')
    @patch('inference_main.read_jsonl')
    @patch('inference_main.logging')
    def test_main_text_modality_with_transcripts(self, mock_logging, mock_read_jsonl,
                                                 mock_load_prompt, mock_load_model):
        # Setup for text modality
        self.mock_args.in_modality = "text"
        self.mock_args.transcript_file = "transcripts.jsonl"

        # Mock model and generate function
        mock_model = MagicMock()
        mock_generate = MagicMock(return_value="Generated translation")
        mock_load_model.return_value = (mock_model, mock_generate)
        mock_load_prompt.return_value = "Mock prompt"

        # Mock transcripts
        mock_transcripts = {
            ("test1", "sample1"): "Hello world",
            ("test2", "sample2"): "Bonjour monde"
        }

        def mock_read_jsonl_side_effect(path):
            if path == self.input_file:
                return iter(self.input_data)
            else:  # transcript file
                return mock_transcripts

        mock_read_jsonl.side_effect = mock_read_jsonl_side_effect

        main(self.mock_args)

        # Verify generate was called with transcripts
        self.assertEqual(mock_generate.call_count, 2)

    @patch('inference_main.load_model')
    @patch('inference_main.load_prompt')
    @patch('inference_main.read_jsonl')
    @patch('inference_main.logging')
    def test_main_text_modality_missing_transcript(self, mock_logging, mock_read_jsonl,
                                                   mock_load_prompt, mock_load_model):
        # Setup for text modality
        self.mock_args.in_modality = "text"
        self.mock_args.transcript_file = "transcripts.jsonl"

        mock_model = MagicMock()
        mock_generate = MagicMock(return_value="Generated translation")
        mock_load_model.return_value = (mock_model, mock_generate)
        mock_load_prompt.return_value = "Mock prompt"

        # Mock empty transcripts (missing transcripts)
        mock_transcripts = {}

        def mock_read_jsonl_side_effect(path):
            if path == self.input_file:
                return iter(self.input_data)
            else:  # transcript file
                return mock_transcripts

        mock_read_jsonl.side_effect = mock_read_jsonl_side_effect

        main(self.mock_args)

        # Verify warning was logged for missing transcripts
        mock_logging.warning.assert_called()

        # Verify output file exists but might be empty or have fewer entries
        self.assertTrue(os.path.exists(self.output_file))


class TestIntegration(unittest.TestCase):
    """Integration tests that test the full workflow with mocked external dependencies"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('inference_main.load_model')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    def test_full_workflow_integration(self, mock_join, mock_file, mock_load_model):
        # Mock model loading
        mock_model = MagicMock()
        mock_generate = MagicMock(return_value="Translated text")
        mock_load_model.return_value = (mock_model, mock_generate)

        # Mock file operations
        mock_join.side_effect = lambda *args: "/".join(args)

        prompt_content = "Translate {src_lang} to {ref_lang}"
        lang_mapping = {"en": "English", "es": "Spanish"}
        input_data = [{"dataset_id": "test", "sample_id": "1", "src_lang": "en", "tgt_lang": "es"}]

        mock_file.side_effect = [
            mock_open(read_data=prompt_content).return_value,
            mock_open(read_data=json.dumps(lang_mapping)).return_value,
            mock_open(read_data=json.dumps(input_data[0]) + "\n").return_value,
            mock_open().return_value,  # output file
        ]

        # Create mock args
        args = MagicMock()
        args.model = "phi4multimodal"
        args.in_modality = "speech"
        args.in_file = "input.jsonl"
        args.out_file = "output.jsonl"
        args.transcript_file = None

        # This should run without errors
        main(args)

        # Verify model was loaded and generate was called
        mock_load_model.assert_called_once()
        mock_generate.assert_called_once()


if __name__ == '__main__':
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests

    # Run the tests
    unittest.main(verbosity=2)