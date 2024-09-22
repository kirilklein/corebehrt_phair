
import shap
import torch
import numpy as np

class CustomModel(torch.nn.Module):
    def __init__(self, sequence_length) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(5, 1) 
        self.emb.weight.data[0, 0] = 0
        for i in range(1, 5):
            self.emb.weight.data[i, 0] = i**2
        self.fc = torch.nn.Linear(1, 1)
        print('embedding weights', self.emb.weight.data)
        self.sequence_length = sequence_length

    def __call__(self, x):
        x = torch.from_numpy(x)
        print('input',x)
        print('input shape', x.shape)
        x = self.emb(x)
        output = x.sum(dim=1)
        print('output shape', output.shape)
        return output

class Masker(torch.nn.Module):
    def __init__(self, mask_value=0) -> None:
        super().__init__()
        self.mask_value = mask_value
    def __call__(self, mask, x):
        return np.where(mask, x, self.mask_value)


X = torch.tensor([
    [[0, 2, 3, 4]], 
    [[1, 2, 3, 4]]])

model = CustomModel(sequence_length=X.shape[1])
masker = Masker()
print('original X shape', X.shape)
explainer = shap.PermutationExplainer(model, masker=masker)
shap_values = explainer.shap_values(X)
print("shap_values: ", shap_values)
print("shap_values.shape: ", shap_values.shape)


