"""
Model architecture for Early Diabetes Risk Prediction.
Assignment: Deep Learning CIA 1

This module implements a configurable Multi-Layer Perceptron (DiabetesMLP)
in PyTorch for binary classification on the Pima Indians Diabetes dataset.

Mathematical Formulations & Layer Mapping:
==========================================

1. Forward Affine Transformation (Linear Layer):
   For each hidden layer l in {1, 2, ..., L-1}:
       z^(l) = W^(l) a^(l-1) + b^(l)
   where:
       - a^(0) = x in R^(d_0) (Input feature vector, d_0 = 8)
       - W^(l) in R^(d_l x d_(l-1)) (Weight matrix for layer l)
       - b^(l) in R^(d_l) (Bias vector for layer l)
       - z^(l) in R^(d_l) (Pre-activation vector)

2. Non-Linear Activation Functions:
   a^(l) = g(z^(l)), evaluated element-wise:
   - ReLU (Rectified Linear Unit):
       g(z) = max(0, z)
   - LeakyReLU (with negative slope alpha = 0.01):
       g(z) = z if z >= 0 else alpha * z
   - Tanh (Hyperbolic Tangent):
       g(z) = (e^z - e^(-z)) / (e^z + e^(-z)) = 2 / (1 + e^(-2z)) - 1

3. Inverted Dropout Regularization (Training Mode Only):
   To prevent co-adaptation of hidden features during training:
       r^(l)_j ~ Bernoulli(1 - p)
       a_tilde^(l)_j = (r^(l)_j * a^(l)_j) / (1 - p)
   where p in [0, 1) is the dropout rate. During inference (eval mode),
   the identity mapping is applied: a_tilde^(l) = a^(l).

4. Output Layer (Raw Unnormalized Logits):
   The final linear transformation outputs a single scalar logit z^(L):
       z^(L) = W^(L) a^(L-1) + b^(L) in R^1
   where W^(L) in R^(1 x d_(L-1)) and b^(L) in R^1.

5. Predicted Probability (Sigmoid Function):
   The posterior class probability y_hat = P(Outcome = 1 | x) is computed via:
       y_hat = sigma(z^(L)) = 1 / (1 + e^(-z^(L))) in [0, 1]

6. Binary Cross-Entropy Loss (BCEWithLogitsLoss):
   Given true binary label y in {0, 1} and predicted probability y_hat:
       L_BCE(y, y_hat) = -1/N * sum_{i=1}^N [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
   For numerical stability, BCEWithLogitsLoss combines sigmoid and BCE using
   the log-sum-exp trick to avoid catastrophic cancellation:
       l_i(z_i, y_i) = max(z_i, 0) - z_i * y_i + log(1 + e^(-|z_i|))
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root and src are available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    import torch.nn as nn
except ImportError as err:
    raise ImportError(
        "\n[PyTorch Import Error] 'torch' was not found in the current Python environment.\n"
        "To resolve this in VS Code:\n"
        "  1. Press Ctrl+Shift+P -> 'Python: Select Interpreter'\n"
        "  2. Choose '.venv' (Python 3.11.9) located at d:\\Deep_Learning\\.venv\\Scripts\\python.exe\n"
    ) from err


class DiabetesMLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron (MLP) for Early Diabetes Risk Prediction.

    Parameters:
    -----------
    input_dim : int, default=8
        Number of input tabular features (Pregnancies, Glucose, BloodPressure,
        SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
    hidden_layers : List[int], default=[32, 16]
        List of integers representing the number of hidden units in each layer.
    activation : str, default='relu'
        Activation function choice. Supported: 'relu', 'leaky_relu' (alpha=0.01), 'tanh'.
    dropout_rate : float, default=0.2
        Dropout probability p in [0, 1). If 0.0, dropout layers are omitted.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.2
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 16]

        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.activation_name = activation.lower()
        self.dropout_rate = float(dropout_rate)

        # Select activation function
        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh
        }
        if self.activation_name not in activation_map:
            raise ValueError(
                f"Unsupported activation '{activation}'. Supported: {list(activation_map.keys())}"
            )
        act_fn = activation_map[self.activation_name]

        # Construct sequential layers
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for l_idx, hidden_dim in enumerate(self.hidden_layers, start=1):
            # Affine transformation: z^(l) = W^(l) a^(l-1) + b^(l)
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear_layer)

            # Non-linear activation: a^(l) = g(z^(l))
            layers.append(act_fn())

            # Dropout regularization: a_tilde^(l)
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            prev_dim = hidden_dim

        # Final classification head: z^(L) = W^(L) a^(L-1) + b^(L) in R^1
        output_layer = nn.Linear(prev_dim, 1)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters:
        -----------
        x : torch.Tensor of shape (N, input_dim)
            Batch of input features a^(0).

        Returns:
        --------
        torch.Tensor of shape (N, 1)
            Unnormalized raw logits z^(L).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted posterior probabilities y_hat = sigma(z^(L)).

        Parameters:
        -----------
        x : torch.Tensor of shape (N, input_dim)
            Batch of input features.

        Returns:
        --------
        torch.Tensor of shape (N, 1)
            Predicted probabilities in range [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_layer_shapes(self) -> List[Dict[str, Any]]:
        """
        Return layer-by-layer shapes, weights, biases, and parameter counts.

        Returns:
        --------
        List[Dict[str, Any]]
            Detailed breakdown of every parameterized layer in the network.
        """
        layer_info = []
        layer_idx = 1
        for name, module in self.network.named_children():
            if isinstance(module, nn.Linear):
                weight_shape = tuple(module.weight.shape)
                bias_shape = tuple(module.bias.shape) if module.bias is not None else None
                weight_params = module.weight.numel()
                bias_params = module.bias.numel() if module.bias is not None else 0
                total_layer_params = weight_params + bias_params

                is_output = (module == self.network[-1])
                role = f"Output Layer (Logit z^({len(self.hidden_layers)+1}))" if is_output else f"Hidden Layer {layer_idx}"

                layer_info.append({
                    "layer_index": layer_idx if not is_output else len(self.hidden_layers) + 1,
                    "module_name": f"network.{name}",
                    "role": role,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weight_shape": weight_shape,
                    "bias_shape": bias_shape,
                    "trainable_parameters": total_layer_params,
                    "formula": f"z^({layer_idx}) = W^({layer_idx}) * a^({layer_idx-1}) + b^({layer_idx})"
                })
                layer_idx += 1
        return layer_info

    def get_trainable_parameters(self) -> int:
        """Return total count of trainable parameters across all layers."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return formatted string detailing network architecture, equations, and shapes."""
        layer_shapes = self.get_layer_shapes()
        total_params = self.get_trainable_parameters()

        lines = [
            "=" * 78,
            f"DiabetesMLP Architecture Summary | Activation: {self.activation_name.upper()} | Dropout: {self.dropout_rate}",
            "=" * 78,
            f"{'Layer':<18} {'Weight Shape':<16} {'Bias Shape':<14} {'Param Count':<12} {'Affine Equation'}",
            "-" * 78
        ]

        for info in layer_shapes:
            w_str = str(info['weight_shape'])
            b_str = str(info['bias_shape'])
            lines.append(
                f"{info['role']:<18} {w_str:<16} {b_str:<14} {info['trainable_parameters']:<12} {info['formula']}"
            )

        lines.extend([
            "-" * 78,
            f"Total Trainable Parameters: {total_params:,}",
            f"Input Dimension: {self.input_dim} | Hidden Layers: {self.hidden_layers} | Output Dimension: 1",
            "=" * 78
        ])
        return "\n".join(lines)


# Helper function for external callers
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in an nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Backward compatibility alias
EarlyDiabetesMLP = DiabetesMLP


if __name__ == "__main__":
    # Test model instantiation and summary
    model = DiabetesMLP(input_dim=8, hidden_layers=[32, 16], activation="relu", dropout_rate=0.2)
    print(model.summary())

    dummy_input = torch.randn(4, 8)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    print(f"\nForward Pass Verification:")
    print(f"  Input shape:       {dummy_input.shape}")
    print(f"  Output logit:      {logits.shape}")
    print(f"  Sigmoid prob:      {probs.shape}")
    print(f"  Sample prob values:\n{probs.flatten().tolist()}")
