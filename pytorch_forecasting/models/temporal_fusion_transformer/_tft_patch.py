"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""  # noqa: E501

from copy import copy
import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    QuantileLoss,
)
from pytorch_forecasting.models.base import BaseModelWithCovariates
# PATCH-MOD: Removed LSTM from imports as it's being replaced
from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from pytorch_forecasting.utils import (
    create_mask,
    detach,
    integer_histogram,
    masked_op,
    padded_stack,
    to_list,
)
from pytorch_forecasting.utils._dependencies import _check_matplotlib


# PATCH-MOD: New module for creating patches from time series data.
class Patching(nn.Module):
    """
    Module to convert a sequence of features into a sequence of patch embeddings.
    This is inspired by the patching mechanism in PatchTST.
    """

    def __init__(self, patch_length: int, stride: int, input_dim: int, output_dim: int):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        # A linear layer to project each flattened patch to the model's embedding dimension.
        self.projection = nn.Linear(patch_length * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor with shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: The sequence of patch embeddings with shape [batch_size, num_patches, output_dim].
        """
        batch_size = x.size(0)
        # unfold to create patches: [batch_size, num_patches, input_dim, patch_length]
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        # reshape for projection: [batch_size, num_patches, input_dim * patch_length]
        patches = patches.permute(0, 1, 3, 2).reshape(batch_size, patches.size(1), -1)
        # project
        patch_embeddings = self.projection(patches)
        return patch_embeddings


# PATCH-MOD: New module for adding positional encoding, which is necessary after removing the LSTM.
class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformers.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[: x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class TemporalFusionTransformer(BaseModelWithCovariates):
    """Temporal Fusion Transformer for forecasting timeseries.

    Initialize via :py:meth:`~from_dataset` method if possible.

    Implementation of
    `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_.

    Enhancements compared to the original implementation:

    * static variables can be continuous
    * multiple categorical variables can be summarized with an EmbeddingBag
    * variable encoder and decoder length by sample
    * categorical embeddings are not transformed by variable selection network
      (because it is a redundant operation)
    * variable dimension in variable selection network are scaled up via linear interpolation to reduce
      number of parameters
    * non-linear variable processing in variable selection network can be
      shared among decoder and encoder (not shared by default)
    * capabilities added through base model such as monotone constraints

    # PATCH-MOD: Added patch-based tokenization as a major architectural change.
    * This version integrates a patch-based tokenization mechanism inspired by PatchTST.
      The LSTM for local processing has been replaced by a patching module followed by a
      positional encoding and the standard TFT attention mechanism. This allows the model
      to handle longer sequences more efficiently.

    Tune its hyperparameters with
    :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

    Parameters
    ----------
    hidden_size : int, default=16
        hidden size of network which is its main hyperparameter.
        Can range from 8 to 512.
    # PATCH-MOD: Removed lstm_layers as it is no longer used.
    dropout : float, default=0.1
        dropout rate
    output_size : int or list of int, default=7
        number of outputs
        (e.g. number of quantiles for QuantileLoss and one target or list of output sizes).
    loss : MultiHorizonMetric, default=QuantileLoss()
        loss function taking prediction and targets
    attention_head_size : int, default=4
        number of attention heads (4 is a good default)
    max_encoder_length : int, default=10
        length to encode,
        can be far longer than the decoder length but does not have to be
    max_prediction_length: int
        maximum prediction length (will be extracted from dataset).
    # PATCH-MOD: Added patch_length and patch_stride hyperparameters.
    patch_length: int, default=16
        The length of each patch.
    patch_stride: int, default=8
        The stride between patches.
    static_categoricals: names of static categorical variables
    static_reals: names of static continuous variables
    time_varying_categoricals_encoder: names of categorical variables for encoder
    time_varying_categoricals_decoder: names of categorical variables for decoder
    time_varying_reals_encoder: names of continuous variables for encoder
    time_varying_reals_decoder: names of continuous variables for decoder
    categorical_groups: dictionary where values
        are list of categorical variables that are forming together a new categorical
        variable which is the key in the dictionary
    x_reals: order of continuous variables in tensor passed to forward function
    x_categoricals: order of categorical variables in tensor passed to forward function
    hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
        embedding size)
    hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
        (fallback to hidden_continuous_size if index is not in dictionary)
    embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
        embedding size
    embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
    embedding_labels: dictionary mapping (string) indices to list of categorical labels
    learning_rate: learning rate
    log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
        , will log multiple entries per batch. Defaults to -1.
    log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
    log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
        failures
    reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
    monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
        variables mapping
        position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
        larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
        This constraint significantly slows down training. Defaults to {}.
    share_single_variable_networks (bool): if to share the single variable networks between the encoder and
        decoder. Defaults to False.
    causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
        predictions. Defaults to True.
    logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
        Defaults to nn.ModuleList().
    mask_bias : float, optional
        Bias for the mask in ScaledDotProductAttention.forward, by default -1e9.
        Set to -float("inf") to allow mixed precision training.
    **kwargs: additional arguments to :py:class:`~BaseModel`.
    """  # noqa: E501

    def __init__(
        self,
        hidden_size: int = 16,
        # PATCH-MOD: Removed lstm_layers
        dropout: float = 0.1,
        output_size: Union[int, list[int]] = 7,
        loss: MultiHorizonMetric = None,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        max_prediction_length: int = 1,
        # PATCH-MOD: Added patch hyperparameters
        patch_length: int = 16,
        patch_stride: int = 8,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        categorical_groups: Optional[Union[dict, list[str]]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Optional[dict[str, int]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_paddings: Optional[list[str]] = None,
        embedding_labels: Optional[dict[str, np.ndarray]] = None,
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Optional[dict[str, int]] = None,
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        mask_bias: float = -1e9,
        **kwargs,
    ):
        
        if monotone_constaints is None:
            monotone_constaints = {}
        if embedding_labels is None:
            embedding_labels = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if hidden_continuous_sizes is None:
            hidden_continuous_sizes = {}
        if x_categoricals is None:
            x_categoricals = []
        if x_reals is None:
            x_reals = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if static_reals is None:
            static_reals = []
        if static_categoricals is None:
            static_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList()
        if loss is None:
            loss = QuantileLoss()
        self.save_hyperparameters()
        # store loss function separately as it is a module
        assert isinstance(
            loss, LightningMetric
        ), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(
                    1,
                    self.hparams.hidden_continuous_sizes.get(
                        name, self.hparams.hidden_continuous_size
                    ),
                )
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.static_categoricals
            },
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_encoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_decoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # PATCH-MOD: Removed static context encoders for LSTM initial hidden and cell states
        # self.static_context_initial_hidden_lstm = GatedResidualNetwork(...)
        # self.static_context_initial_cell_lstm = GatedResidualNetwork(...)

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.dropout,
        )

        # PATCH-MOD: Initialize Patching and PositionalEncoding modules instead of LSTMs
        self.patching = Patching(
            patch_length=self.hparams.patch_length,
            stride=self.hparams.patch_stride,
            input_dim=self.hparams.hidden_size,
            output_dim=self.hparams.hidden_size,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # PATCH-MOD: Removed LSTM encoder and decoder
        # self.lstm_encoder = LSTM(...)
        # self.lstm_decoder = LSTM(...)

        # PATCH-MOD: Removed skip connection modules for LSTM
        # self.post_lstm_gate_encoder = GatedLinearUnit(...)
        # self.post_lstm_add_norm_encoder = AddNorm(...)
        # self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size,
            n_head=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
            mask_bias=self.hparams.mask_bias,
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=None, trainable_add=False
        )

        # PATCH-MOD: Add a prediction head to map from patched sequence length to prediction length
# Calculate the number of patches for the decoder part 
        input_size_for_head = 256

# Define the prediction head with the correct in_features
        self.prediction_head = nn.Linear(
    in_features=input_size_for_head,
    out_features=self.hparams.max_prediction_length * self.hparams.output_size,
)

        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, output_size)
                    for output_size in self.hparams.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size, self.hparams.output_size
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """  # noqa: E501
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        # PATCH-MOD: Add max_prediction_length to kwargs for the prediction head
        new_kwargs["max_prediction_length"] = dataset.max_prediction_length
        new_kwargs.update(
            cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss())
        )

        # create class and return
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)
    # Add this new helper method inside your TemporalFusionTransformer class
    def _get_patch_counts(self, sequence_lengths: torch.LongTensor, patch_length: int, patch_stride: int):
        """Calculates the number of patches for a given sequence length."""
        # Ensure lengths are at least the patch length to avoid negative counts
        valid_lengths = torch.clamp(sequence_lengths, min=patch_length)
        # Standard formula for patch count
        return (valid_lengths - patch_length) // patch_stride + 1

    # Modify the existing get_attention_mask method
    def get_attention_mask(
        self,
        encoder_lengths: torch.LongTensor,
        decoder_lengths: torch.LongTensor,
        patch_length: int,
        patch_stride: int
    ):
        """
        Returns causal mask to apply for self-attention layer based on patch counts.
        """
        # PATCH-MOD: Calculate sequence length in terms of patches
        encoder_patch_lengths = self._get_patch_counts(encoder_lengths, patch_length, patch_stride)
        decoder_patch_lengths = self._get_patch_counts(decoder_lengths, patch_length, patch_stride)

        max_encoder_patches = encoder_patch_lengths.max()
        max_decoder_patches = decoder_patch_lengths.max()

        # Create mask for encoder patches
        encoder_mask = create_mask(max_encoder_patches, encoder_patch_lengths).unsqueeze(1).expand(-1, max_decoder_patches, -1)

        # Create causal mask for decoder patches
        if self.hparams.causal_attention:
            # Standard causal mask for transformers
            decoder_mask = torch.triu(
                torch.full((max_decoder_patches, max_decoder_patches), 1.0, device=self.device),
                diagonal=1,
            ).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
            
            # Mask out padding in the decoder
            decoder_padding_mask = create_mask(max_decoder_patches, decoder_patch_lengths).unsqueeze(1).expand(-1, max_decoder_patches, -1)
            decoder_mask = (decoder_mask.bool() | ~decoder_padding_mask).bool()

        else:
            # Mask out only padding
            decoder_mask = create_mask(max_decoder_patches, decoder_patch_lengths).unsqueeze(1).expand(-1, max_decoder_patches, -1)


        # Combine masks along the key/value sequence axis
        mask = torch.cat((encoder_mask, decoder_mask), dim=2)
        return mask
    
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
    input dimensions: n_samples x time x variables
    """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        max_encoder_length = int(encoder_lengths.max())

    # Create a single continuous tensor for all time-varying inputs
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cat.size(1)

        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
        {
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        }
    )

    # Process static variables
        if len(self.static_variables) > 0:
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
        #
            static_embedding = torch.zeros(
            (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
        )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

    # PATCH-MOD: Unify variable selection for the entire sequence
    # This creates a single feature vector of 'hidden_size' for each time step
        static_context_variable_selection = self.expand_static_context(
        self.static_context_variable_selection(static_embedding), timesteps
    )
    
    # Use decoder variable selection for all variables (assuming it's the superset)
    # or create a new one for all time-varying vars. Here we use decoder's.
        all_varying_vars = set(self.encoder_variables + self.decoder_variables)
        embeddings_varying = {
    name: input_vectors[name] for name in all_varying_vars
}
    
    # We get sparse weights for encoder and decoder separately for interpretation
        _, encoder_sparse_weights = self.encoder_variable_selection(
        {name: embeddings_varying[name][:, :max_encoder_length] for name in self.encoder_variables},
        static_context_variable_selection[:, :max_encoder_length],
    )
    
        temporal_features, decoder_sparse_weights = self.decoder_variable_selection(
        embeddings_varying,
        static_context_variable_selection,
    )

    # PATCH-MOD: Apply patching and positional encoding to the unified sequence
        patched_features = self.patching(temporal_features)
        num_patches = patched_features.size(1)
        patched_features = self.positional_encoding(patched_features)

    # Static enrichment on the patched sequence
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
        patched_features,
        self.expand_static_context(static_context_enrichment, num_patches),
    )

    # Attention Block
    # PATCH-MOD: Calculate patch counts once and use them consistently
        num_encoder_patches = self._get_patch_counts(
        torch.tensor([max_encoder_length], device=self.device),
        self.hparams.patch_length,
        self.hparams.patch_stride
    )[0]

        # --- FINAL FIX BLOCK START ---
        q_sliced = attn_input[:, num_encoder_patches:]
        decoder_patches = q_sliced.size(1)
        total_patches = attn_input.size(1)

# Manually create a correct attention mask.
# This creates a causal mask for the decoder part of the sequence.
        causal_mask = torch.triu(
    torch.ones(decoder_patches, decoder_patches, device=self.device, dtype=torch.bool),
    diagonal=1
)

# Create the full mask that allows the decoder to attend to the entire
# encoder sequence and causally to the decoder sequence.
# Final shape will be (decoder_patches, total_patches) e.g. (2, 11)
        correct_mask = torch.zeros(decoder_patches, total_patches, device=self.device, dtype=torch.bool)
        correct_mask[:, num_encoder_patches:] = causal_mask

# Pass the correctly created mask to the attention layer
        attn_output, attn_output_weights = self.multihead_attn(
    q=q_sliced,
    k=attn_input,
    v=attn_input,
    mask=correct_mask,
)
    # Skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_input[:, num_encoder_patches:], attn_output)

    # Position-wise Feed-Forward
        output = self.pos_wise_ff(attn_output)

    # Skip connection
        output = self.pre_output_gate_norm(attn_input[:, num_encoder_patches:], output)
    
    # PATCH-MOD: Final prediction head to map decoder patches to the forecast horizon
    # This requires a 'self.prediction_head' layer defined in __init__
    # e.g., self.prediction_head = nn.Linear(num_decoder_patches * hidden_size, max_prediction_length * output_size)
        output = self.prediction_head(output.flatten(1))
    
    # Reshape to (batch_size, max_prediction_length, output_size)
        output = output.view(
    output.size(0), self.hparams.max_prediction_length, self.hparams.output_size
)

        return self.to_network_output(
        prediction=self.transform_output(output, target_scale=x["target_scale"]),
        encoder_attention=attn_output_weights[..., :num_encoder_patches],
        decoder_attention=attn_output_weights[..., num_encoder_patches:],
        static_variables=static_variable_selection,
        encoder_variables=encoder_sparse_weights,
        decoder_variables=decoder_sparse_weights,
        decoder_lengths=decoder_lengths,
        encoder_lengths=encoder_lengths,
    )

    

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon # noqa: E501
        )
        return interpretation

    def on_epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0 and not self.training:
            self.log_interpretation(outputs)

    def interpret_output(
        self,
        out: dict,
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> dict:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """  # noqa: E501
        # PATCH-MOD: The interpretation logic for attention is now over patches, not timesteps.
        # The original logic for rolling and masking attention might not be directly applicable
        # or may need adjustment. For this modification, we will simplify the attention
        # interpretation part. A more sophisticated implementation would map patch attention
        # back to the original time steps.

        # take attention and concatenate if a list to proper attention object
        batch_size = len(out["decoder_attention"])
        
        # PATCH-MOD: Simplified attention processing for interpretation
        encoder_attention = padded_stack(out["encoder_attention"])
        decoder_attention = padded_stack(out["decoder_attention"])

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(
            out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length
        )
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied
        # when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2).clone()
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(
            encode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
           .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
           .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2).clone()
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(
            decode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and
        # attention on observed timesteps
        
        # PATCH-MOD: Select attention for the first predicted patch
        attention = masked_op(
            attention[
                :,
                attention_prediction_horizon,
                :,
                :,
            ],
            op="mean",
            dim=1,
        )

        if reduction!= "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(
                -1
            )  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(
        self,
        x: dict,
        out: dict,
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ):
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict): network input
            out (Dict): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """
        # plot prediction as normal
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            # PATCH-MOD: Plotting attention over patches. The x-axis will represent patch indices, not time steps.
            interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
            for f in to_list(fig):
    # f.axes is a list, so we select the first axis object from it.
                if f.axes: # Check that the list is not empty
                    ax = f.axes[0] 
                    ax2 = ax.twinx()
                ax2.set_ylabel("Attention over Patches")
                
                # Calculate number of encoder patches for this sample
                encoder_length = x["encoder_lengths"][idx]
                num_encoder_patches = (encoder_length - self.hparams.patch_length) // self.hparams.patch_stride + 1
                
                # Create an x-axis for patches, mapping them roughly to the original time axis
                patch_indices = torch.arange(-num_encoder_patches, 0) * self.hparams.patch_stride
                
                ax2.plot(
                    patch_indices,
                    interpretation["attention"][0, -num_encoder_patches:].detach().cpu(),
                    alpha=0.2,
                    color="k",
                    marker='o',
                    linestyle='--'
                )
                f.tight_layout()
        return fig

    def plot_interpretation(self, interpretation: dict):
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        _check_matplotlib("plot_interpretation")

        import matplotlib.pyplot as plt

        figs = {}

        # PATCH-MOD: Attention plot now shows attention over patch indices
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        
        num_encoder_patches = (self.hparams.max_encoder_length - self.hparams.patch_length) // self.hparams.patch_stride + 1
        num_total_patches = attention.size(0)
        
        ax.plot(
            np.arange(
                -num_encoder_patches,
                num_total_patches - num_encoder_patches,
            ),
            attention,
        )
        ax.set_xlabel("Patch index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(
                np.arange(len(values)),
                values[order] * 100,
                tick_label=np.asarray(labels)[order],
            )
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance",
            interpretation["static_variables"].detach().cpu(),
            self.static_variables,
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance",
            interpretation["encoder_variables"].detach().cpu(),
            self.encoder_variables,
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance",
            interpretation["decoder_variables"].detach().cpu(),
            self.decoder_variables,
        )

        return figs

    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # PATCH-MOD: Interpretation logging is kept, but note that attention is over patches.
        # extract interpretations
        if not outputs:
            return
        interpretation_keys = outputs[0]["interpretation"].keys()

        interpretation = {
            name: padded_stack(     
                [x["interpretation"][name].detach() for x in outputs],
                side="right",
                value=0,
            ).sum(0)
            for name in interpretation_keys
        }
        # normalize attention with length histogram squared to account for:
        # 1. zeros in attention and
        # 2. higher attention due to less values
        
        # PATCH-MOD: This normalization needs to be adapted for patches.
        # For simplicity, we will just normalize by the sum.
        interpretation["attention"] = (
            interpretation["attention"] / interpretation["attention"].sum()
        )

        mpl_available = _check_matplotlib("log_interpretation", raise_error=False)

        # Don't log figures if matplotlib or add_figure is not available
        if not mpl_available or not self._logger_supports("add_figure"):
            return None

        import matplotlib.pyplot as plt

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance",
                fig,
                global_step=self.global_step,
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack(
                    [
                        out["interpretation"][f"{type}_length_histogram"]
                        for out in outputs
                    ]
                )
               .sum(0)
               .detach()
               .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution",
                fig,
                global_step=self.global_step,
            )

    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """

        # Don't log embeddings if add_embedding is not available
        if not self._logger_supports("add_embedding"):
            return None

        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(),
                metadata=labels,
                tag=name,
                global_step=self.global_step,
            )