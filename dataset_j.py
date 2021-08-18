from torch.utils.data import Dataset
import torch


class climate_dataset(Dataset):
    def __init__(self, data, is_test):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        input_id = torch.tensor(self.data["input_ids"][idx], dtype=torch.long)
        attention_mask_ = torch.tensor(self.data["attention_mask"][idx], dtype=torch.long)
        token_type_id = torch.tensor(self.data["token_type_ids"][idx], dtype=torch.long)

        if not self.is_test:
            label = torch.tensor(self.data["labels"][idx], dtype=torch.long)
            return {"input_id": input_id, "attention_mask_": attention_mask_, "token_type_id": token_type_id,
                    "label": label}
        else:
            return {"input_id": input_id, "attention_mask_": attention_mask_, "token_type_id": token_type_id}

class climate_dataset2(Dataset):
    def __init__(self, data, is_test):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        input_id = torch.tensor(self.data["input_ids"][idx], dtype=torch.long)
        attention_mask_ = torch.tensor(self.data["attention_mask"][idx], dtype=torch.long)
        token_type_id = torch.tensor(self.data["token_type_ids"][idx], dtype=torch.long)

        if not self.is_test:
            s_label = torch.tensor(self.data["s_labels"][idx], dtype=torch.long)
            m_label = torch.tensor(self.data["m_labels"][idx], dtype=torch.long)
            return {"input_id": input_id, "attention_mask_": attention_mask_, "token_type_id": token_type_id,
                    "s_label": s_label, "m_label": m_label}
        else:
            return {"input_id": input_id, "attention_mask_": attention_mask_, "token_type_id": token_type_id}