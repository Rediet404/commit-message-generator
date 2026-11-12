import pytest
from commitgen.model.finetune import CodeDocstringDataset, FineTuneConfig


def test_finetune_config_defaults() -> None:
    config = FineTuneConfig()
    assert config.model_name == "Salesforce/codet5-small"
    assert config.learning_rate == 5e-5
    assert config.batch_size == 4
    assert config.max_input_length > config.max_target_length


def test_code_docstring_dataset_tokenization() -> None:
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, max_length, padding, truncation, return_tensors):
            import torch
            # Return fixed dummy tensor
            token_ids = torch.tensor([[1, 2, 3, 0]])
            return {"input_ids": token_ids, "attention_mask": torch.tensor([[1, 1, 1, 0]])}

    examples = [{"code": "def foo(): pass", "docstring": "Do foo"}]
    dataset = CodeDocstringDataset(examples, FakeTokenizer(), FineTuneConfig())
    assert len(dataset) == 1
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    # Check that pad_token_id was replaced by -100 in labels
    assert (sample["labels"] == -100).any()
