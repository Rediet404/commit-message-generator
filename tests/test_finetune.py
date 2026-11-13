import pytest
from commitgen.model.finetune import CodeDocstringDataset, FineTuneConfig, load_training_records


def test_finetune_config_defaults() -> None:
    config = FineTuneConfig()
    assert config.model_name == "Salesforce/codet5-small"
    assert config.learning_rate == 5e-5
    assert config.batch_size == 4
    assert config.max_input_length > config.max_target_length


def test_code_docstring_dataset_tokenization() -> None:
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, max_length=None, padding=None, truncation=False, return_tensors=None):
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


def test_load_training_records_accepts_commit_pairs(tmp_path) -> None:
    path = tmp_path / "commits.jsonl"
    path.write_text(
        '{"diff": "diff --git a/a.py b/a.py", "commit_subject": "Update a"}\n',
        encoding="utf-8",
    )

    records = load_training_records(path)

    assert records[0]["commit_subject"] == "Update a"


def test_load_training_records_rejects_unpaired_records(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"diff": "only a diff"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="lacks either"):
        load_training_records(path)
