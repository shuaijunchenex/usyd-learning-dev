import torch
import torch.nn as nn

from ..transformer_encoder._nn_model_transformer_encoder import TransformerEncoder



class SimpleViT(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    Transforms an image into a sequence of patch tokens and processes them using Transformer Encoder layers.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        image_size: int = 32,
        num_patches_per_dim: int = 8,
        dropout: float = 0.0,
        num_layers: int = 7,
        hidden_dim: int = 384,
        mlp_hidden_dim: int = 384 * 4,
        num_heads: int = 8,
        use_cls_token: bool = True
    ):
        """
        :param in_channels: Number of image input channels (e.g., 3 for RGB)
        :param num_classes: Number of output classes
        :param image_size: Input image size (assumes square image)
        :param num_patches_per_dim: Number of patches along one dimension (image will be divided into num_patches^2 patches)
        :param dropout: Dropout rate
        :param num_layers: Number of Transformer encoder blocks
        :param hidden_dim: Embedding dimension after patch projection
        :param mlp_hidden_dim: Hidden dimension in the MLP layers
        :param num_heads: Number of attention heads
        :param use_cls_token: Whether to use a [CLS] token for classification
        """
        super().__init__()

        self.num_patches_per_dim = num_patches_per_dim
        self.use_cls_token = use_cls_token
        self.patch_size = image_size // num_patches_per_dim
        self.patch_vector_dim = (self.patch_size ** 2) * in_channels  # dimension of each flattened patch

        num_tokens = (num_patches_per_dim ** 2) + 1 if use_cls_token else (num_patches_per_dim ** 2)

        # Linear projection from flattened patch to hidden embedding
        self.patch_embedding = nn.Linear(self.patch_vector_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if use_cls_token else None

        # Positional embedding for patch tokens (and CLS token if used)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))

        # Stack of Transformer encoder layers
        self.encoder = nn.Sequential(*[
            TransformerEncoder(
                feature_dim=hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)])

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Logits of shape (batch_size, num_classes)
        """
        # Convert image to patch tokens
        x_patches = self._image_to_patches(x)  # (batch_size, num_patches, patch_vector_dim)
        x_embed = self.patch_embedding(x_patches)  # (batch_size, num_patches, hidden_dim)

        # Prepend CLS token if used
        if self.use_cls_token:
            batch_size = x_embed.size(0)
            cls_token = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, hidden_dim)
            x_embed = torch.cat([cls_token, x_embed], dim=1)  # (batch_size, num_patches+1, hidden_dim)

        # Add positional embeddings
        x_embed = x_embed + self.positional_embedding

        # Transformer encoder
        x_encoded = self.encoder(x_embed)  # (batch_size, num_tokens, hidden_dim)

        # Classification head
        if self.use_cls_token:
            x_out = x_encoded[:, 0]  # Use [CLS] token
        else:
            x_out = x_encoded.mean(dim=1)  # Mean pooling

        logits = self.head(x_out)  # (batch_size, num_classes)
        return logits

    def _image_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input images to flattened patch vectors.
        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Patch tensor of shape (batch_size, num_patches, patch_vector_dim)
        """
        batch_size, channels, height, width = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # shape: (batch_size, channels, num_patches_y, num_patches_x, patch_h, patch_w)
        x = x.permute(0, 2, 3, 4, 5, 1)  # (batch_size, num_patches_y, num_patches_x, patch_h, patch_w, channels)
        x = x.reshape(batch_size, -1, self.patch_vector_dim)  # (batch_size, num_patches, patch_vector_dim)
        return x

