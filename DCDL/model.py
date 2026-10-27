"""Equations (1)-(20), with explicitly documented finite-boundary conventions."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _StaticSparseMM(torch.autograd.Function):
    """For fixed A, d(AE)/dE uses its already stored transpose, with no graph gradient."""
    @staticmethod
    def forward(ctx, features, matrix, transpose):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(matrix, features)

    @staticmethod
    def backward(ctx, gradient):
        transpose, = ctx.saved_tensors
        return torch.sparse.mm(transpose, gradient), None, None


class SparseGAT(nn.Module):
    """One head, one layer, self-loop edges supplied by original UB preprocessing."""
    def __init__(self, dimension, dropout, negative_slope):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False)
        self.attention = nn.Parameter(torch.empty(2, dimension))
        nn.init.xavier_uniform_(self.attention)
        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, embeddings, edges):
        h = self.projection(F.dropout(embeddings, self.dropout, self.training))
        target, source = edges
        # h^T [W e_m || W e_n], without materializing concatenated edge features.
        target_logits = (h * self.attention[0]).sum(-1)
        source_logits = (h * self.attention[1]).sum(-1)
        logits = F.leaky_relu(target_logits[target] + source_logits[source], self.negative_slope)
        maximum = logits.new_full((len(h),), -torch.inf)
        maximum.scatter_reduce_(0, target, logits.detach(), reduce="amax", include_self=True)
        numerator = (logits - maximum[target]).exp()
        denominator = logits.new_zeros(len(h)).index_add(0, target, numerator)
        attention = numerator / denominator[target]
        attention = F.dropout(attention, self.dropout, self.training)
        output = h.new_zeros(h.shape).index_add(0, target, attention[:, None] * h[source])
        return F.elu(output)


def sinusoidal_embedding(steps, dimension):
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=steps.device, dtype=torch.float32) / half)
    phase = steps.float()[:, None] * frequencies[None, :]
    output = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return F.pad(output, (0, dimension % 2))


class CollaborativeFusionDiffuser(nn.Module):
    def __init__(self, conf):
        super().__init__()
        d = conf["embedding_size"]
        self.dimension = d
        self.steps = conf["diffusion_steps"]
        self.network = nn.Sequential(
            nn.Linear(3 * d, 4 * d), nn.GELU(), nn.Dropout(conf["dropout"]),
            nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d),
        )
        beta = torch.linspace(conf["beta_start"], conf["beta_end"], self.steps, dtype=torch.float64)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        variance = beta * (1 - previous) / (1 - alpha_bar)
        # Eq. (17) is singular at t=1. Use beta_1 for that reconstruction term.
        variance[0] = beta[0]
        weights = previous / (2 * variance)
        self.register_buffer("sqrt_alpha", alpha.sqrt().float())
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt().float())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt().float())
        self.register_buffer("loss_weights", weights.float())

    def forward(self, bundle_users, fused_users, steps, noise):
        """steps uses the paper's 1..T convention; keep both encoder gradient paths."""
        index = steps - 1
        noisy = self.sqrt_alpha_bar[index, None] * bundle_users + self.sqrt_one_minus_alpha_bar[index, None] * noise
        time = sinusoidal_embedding(steps, self.dimension)
        middle = self.network(torch.cat((time, noisy, fused_users), dim=-1))
        residual = bundle_users - self.sqrt_alpha[index, None] * middle
        per_user = self.loss_weights[index] * residual.square().sum(-1)
        return middle, per_user


class DCDL(nn.Module):
    def __init__(self, conf, sizes, graphs, num_train):
        super().__init__()
        self.num_users, self.num_bundles, self.num_items = sizes
        self.num_train = num_train
        self.alpha_1 = conf["alpha_1"]
        self.lambda_1 = conf["lambda_1"]
        self.lambda_2 = conf["lambda_2"]
        d = conf["embedding_size"]
        self.user_embedding = nn.Embedding(self.num_users, d)
        self.bundle_embedding = nn.Embedding(self.num_bundles, d)
        self.item_embedding = nn.Embedding(self.num_items, d)
        # Literal user-indexed w_u in Eq. (18); no bias or unreported shared projection.
        self.boundary_weight = nn.Embedding(self.num_users, d)
        self.gat = SparseGAT(d, conf["dropout"], conf["gat_negative_slope"])
        self.diffuser = CollaborativeFusionDiffuser(conf)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, graph in graphs.items():
            # Stored once separately in checkpoints, not duplicated in state_dict.
            self.register_buffer(name, graph, persistent=False)
        self.graph_names = tuple(graphs)
        self._cpu_sparse_cache = {}

    def sparse_mm(self, name, transpose_name, features):
        matrices = []
        for key in (name, transpose_name):
            matrix = getattr(self, key)
            if features.device.type == "cpu":
                cached = self._cpu_sparse_cache.get(key)
                # Moving the model or changing dtype replaces graph tensors;
                # regenerate the cache instead of reusing stale device/dtype data.
                if cached is None or cached[0] is not matrix:
                    cached = (matrix, matrix.to_sparse_csr())
                    self._cpu_sparse_cache[key] = cached
                matrix = cached[1]
            matrices.append(matrix)
        return _StaticSparseMM.apply(features, matrices[0], matrices[1])

    def item_view(self):
        items = self.item_embedding.weight
        # (Y^T Y) E / ((Y^T Y) 1), exactly Eq. (6), with no I x I allocation.
        user_co = self.sparse_mm("ui_t", "ui", self.sparse_mm("ui", "ui_t", items))
        bundle_co = self.sparse_mm("bi_t", "bi", self.sparse_mm("bi", "bi_t", items))
        items = items + F.relu(user_co / self.ui_co_degree[:, None]) + F.relu(bundle_co / self.bi_co_degree[:, None])
        users = self.sparse_mm("ui", "ui_t", items) / self.ui_row_degree[:, None]
        bundles = self.sparse_mm("bi", "bi_t", items) / self.bi_row_degree[:, None]
        return users, bundles

    def encode(self):
        entities = torch.cat((self.user_embedding.weight, self.bundle_embedding.weight), dim=0)
        gat_entities = self.gat(entities, self.gat_edges)
        bundle_view = self.sparse_mm("relation", "relation", self.sparse_mm("relation", "relation", gat_entities)) / self.relation_norm
        bundle_users, bundle_bundles = bundle_view.split((self.num_users, self.num_bundles))
        item_users, item_bundles = self.item_view()
        return bundle_users + item_users, bundle_bundles + item_bundles, bundle_users

    def forward(self, users, positive, negative, steps, noise):
        fused_users, fused_bundles, bundle_users = self.encode()
        # All declared users contribute to L_d, including users without UB training edges.
        middle, per_user_diffusion = self.diffuser(bundle_users, fused_users, steps, noise)
        boundary = (middle[users] * self.boundary_weight(users)).sum(-1)
        positive_scores = (fused_users[users] * fused_bundles[positive]).sum(-1)
        negative_scores = (fused_users[users] * fused_bundles[negative]).sum(-1)
        uib = (F.softplus(boundary - positive_scores) + self.alpha_1 * F.softplus(negative_scores - boundary)).mean()
        # Divide the ENTIRE full-data Eq. (20) by N_train. The minibatch UIB mean
        # estimates its full-data term; exact auxiliary sums use the same denominator.
        diffusion = per_user_diffusion.sum() / self.num_train
        regularization = sum(parameter.square().sum() for parameter in self.parameters()) / self.num_train
        loss = uib + self.lambda_1 * diffusion + self.lambda_2 * regularization
        return loss, {"uib": uib, "diffusion": diffusion, "l2": regularization}

    def graph_state(self):
        return {name: getattr(self, name).detach().cpu() for name in self.graph_names}
