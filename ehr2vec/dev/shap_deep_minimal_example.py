
import shap
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb1 = torch.nn.Embedding(5, 1)
        self.emb2 = torch.nn.Embedding(5, 1)
        self.fc = torch.nn.Linear(3, 1)
    def forward(self, hidden_states):
        """Takes hidden states of size bs, emb_dim, seq_len and returns a single value for each sample in the batch."""
        x = hidden_states.sum(dim=[1, 2]).unsqueeze(1)
        print('output shape', x.shape)
        return x

class ModelWrapper(torch.nn.Module):
    def __init__(self, model:MyModel) -> None:
        super().__init__()
        self.model = model
        # we will use embeddings from the model to get something we can pass as hidden states
        self.emb1 = model.emb1
        self.emb2 = model.emb2

    def forward(self, x1, x2, x3):
        """
        This will takes the input from the explainer and pass it to the model.
        Let our input be a 4d array of size (bs, D, seq_len, hidden_size)
        """
        # sum over the list of tensors
        hidden_states = sum([x1, x2, x3])
        print('hidden_states shape', hidden_states.shape)
        return self.model(hidden_states)


# random torch tensor
batch_size = 4
num_features = 3
hidden_size = 5
seq_len = 6
X_background = [torch.rand(batch_size, seq_len, hidden_size) for _ in range(num_features)]
X = [torch.rand(batch_size, seq_len, hidden_size) for _ in range(num_features)]
print('X shape', X[0].shape)

model = MyModel()
wrapped_model = ModelWrapper(model)
explainer = shap.DeepExplainer(wrapped_model, data=X_background)
shap_values = explainer.shap_values(X=X)
#print("shap_values: ", shap_values)
print('shap_values length', len(shap_values))
print("shap_values.shape: ", shap_values[0].shape)

#print('shap_values[0]', shap_values[0])
#print('shap_values[1]', shap_values[1])
