"""Trainable binding memories and logical inference work accounting."""

import math

import torch
from torch import nn
import torch.nn.functional as F


CAP_BYTES = 32 * 1024
READ_FLOOR = 0.25
RETRIEVE = 4


@torch.no_grad()
def reset_features(body, tokens, block=128):
    """Batch independent blocks through the shared frozen body."""
    batch, length = tokens.shape
    padded = F.pad(tokens, (0, (-length) % block))
    with torch.autocast(tokens.device.type, torch.bfloat16, enabled=tokens.is_cuda):
        features, _ = body._features(padded.reshape(-1, block))
    return features.reshape(batch, -1, features.shape[-1])[:, :length]


def event_key(logits, winners):
    top, indices = logits.topk(winners, dim=-1)
    hard = torch.zeros_like(logits).scatter_(-1, indices, 1)
    soft = (logits - top[..., -1:].detach()).sigmoid()
    return hard + (soft - soft.detach())


def persistent_read(query, key, value, block=128):
    """Causal no-decay outer-product scan. The last value column holds mass."""
    length = query.size(1)
    block = min(block, length)
    pad = (-length) % block
    q, k, v = [F.pad(x, (0, 0, 0, pad)).unflatten(1, (-1, block))
               for x in (query, key, F.pad(value, (0, 1), value=1))]
    updates = k.transpose(-1, -2) @ v
    before = F.pad(updates[:, :-1].cumsum(dim=1), (0, 0, 0, 0, 1, 0))
    local = (q @ k.transpose(-1, -2)).tril() @ v
    read = (q @ before + local).flatten(1, 2)[:, :length]
    return read[..., :-1] / (read[..., -1:] + READ_FLOOR)


def select_positions(x, positions):
    return x.gather(1, positions.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def overlap_read(query, key, value, positions, slots):
    """Search the currently retained FIFO writes, then read the top overlaps."""
    wrote = key.ne(0).any(-1)
    order = wrote.cumsum(1)
    newest = order.gather(1, positions)
    clock = torch.arange(key.size(1), device=key.device)
    live = (clock <= positions.unsqueeze(-1)) & (order[:, None] > newest.unsqueeze(-1) - slots)
    score = (select_positions(query, positions) @ key.transpose(-1, -2)).masked_fill(~live, 0)
    priority = (score * (key.size(1) + 1) + clock).masked_fill(score <= 0, -1)
    indices = priority.topk(min(RETRIEVE, key.size(1)), dim=-1).indices
    weights = score.gather(-1, indices)
    values = value[torch.arange(value.size(0), device=value.device)[:, None, None], indices]
    return (weights.unsqueeze(-1) * values).sum(-2) / (weights.sum(-1, keepdim=True) + READ_FLOOR)


class BindingMemory(nn.Module):
    def __init__(self, width, sparse, addresses=128, values=32, winners=16, cap=CAP_BYTES):
        super().__init__()
        self.width, self.sparse, self.addresses, self.values, self.winners = width, sparse, addresses, values, winners
        if not 1 <= winners <= addresses:
            raise ValueError("winners must be between one and the number of addresses")
        self.mask_bytes = math.ceil(addresses / 8)
        self.row_bytes = 4 * (values + 1)
        # Working key/query vectors and previous trace. The activation source
        # and decoded residual use the shared body buffers.
        self.trace_bytes = self.mask_bytes if sparse else 4 * addresses
        scratch = 3 * self.trace_bytes + 2 * self.row_bytes + 4
        self.state_bytes = addresses * self.row_bytes + scratch
        self.record_bytes = self.mask_bytes + 4 * values
        # Two u32 FIFO indices plus a u32 score and index per shortlisted record.
        overlap_scratch = scratch + 8 + RETRIEVE * 8
        self.slots = (cap - overlap_scratch) // self.record_bytes
        self.overlap_bytes = overlap_scratch + self.slots * self.record_bytes
        if self.state_bytes > cap or (sparse and self.slots < 1):
            raise ValueError(f"mutable memory exceeds the {cap:,}-byte state cap")
        self.trace = nn.Linear(width, addresses)
        self.key = nn.Linear(width, addresses)
        self.query = nn.Linear(width, addresses)
        self.gate = nn.Linear(width, 1)
        self.value = nn.Linear(width, values)
        self.decode = nn.Linear(values, width, bias=False)
        nn.init.normal_(self.gate.weight, std=width ** -.5)
        nn.init.constant_(self.gate.bias, -1.5)
        nn.init.normal_(self.decode.weight, std=.02)

    def encode(self, x):
        trace, key, query = self.trace(x), self.key(x), self.query(x)
        prob = self.gate(x).sigmoid()
        gate = (prob >= .5).to(x.dtype) + (prob - prob.detach())
        if self.sparse:
            trace, key, query = (event_key(a, self.winners) for a in (trace, key, query))
            touched = key * F.pad(trace[:, :-1], (0, 0, 1, 0))
        else:
            touched = (key + F.pad(trace[:, :-1], (0, 0, 1, 0))).sigmoid() / math.sqrt(self.addresses)
            query = query.sigmoid() / math.sqrt(self.addresses)
        return query, touched * gate, self.value(x)

    def forward(self, x, positions):
        q, k, v = self.encode(x)
        delta = self.decode(select_positions(persistent_read(q, k, v), positions))
        return select_positions(x, positions) + delta, (q, k, v)

    def digital(self, x, positions, codes):
        return select_positions(x, positions) + self.decode(overlap_read(*codes, positions, self.slots))

    @torch.no_grad()
    def work(self, codes, digital=False):
        query, key, _ = codes
        written_rows = key.ne(0).sum(-1)
        wrote = written_rows > 0
        tokens, writes = wrote.numel(), int(wrote.sum())
        # Common projections use all features. A deployed value projection
        # can run only for a successful write because it has no other reader.
        projected = tokens * self.width * (3 * self.addresses + 1 + self.values)
        projected += writes * self.width * self.values
        # Charge both stream accesses to the previous trace plus read/write
        # of working query/key and value buffers on every token.
        shared_reads = tokens * (3 * self.trace_bytes + self.row_bytes)
        shared_writes = tokens * (3 * self.trace_bytes + self.row_bytes)
        # Select three winners, test coincidence and open the gate. Comparators
        # count input addresses processed, leaving their implementation open.
        selection = tokens * (4 * self.addresses + 1) if self.sparse else tokens
        # The shared read does one scalar mass offset and divides each value
        # channel. Report divisions separately from matrix operations.
        additions, macs = tokens, projected
        divides = tokens * self.values
        if digital:
            live = wrote.cumsum(1).clamp_max(self.slots)
            candidates = int(live.sum())
            # Budget four fetched values whenever the FIFO has that many;
            # fewer may be retrieved if the query finds no positive overlaps.
            fetched = int(live.clamp_max(RETRIEVE).sum())
            reads = shared_reads + tokens * 8 + candidates * self.mask_bytes + fetched * 4 * self.values
            stores = shared_writes + writes * (self.record_bytes + 8)
            macs += fetched * (self.values + 1)
            selection += candidates  # one packed mask intersection and score/rank per live record
            total_bytes = self.overlap_bytes
        else:
            queried = int(query.ne(0).sum())
            updates = int(written_rows.sum())
            reads = shared_reads + (queried + updates) * self.row_bytes
            stores = shared_writes + updates * self.row_bytes
            operations = (queried + updates) * (self.values + 1)
            if self.sparse:
                additions += operations
            else:
                macs += operations
            total_bytes = self.state_bytes
        return {"tokens": tokens, "successful_writes": writes,
                "state_bytes_read": reads, "state_bytes_written": stores,
                "macs": macs, "adds": additions, "divides": divides, "address_checks": selection,
                "allocated_state_bytes": total_bytes, "reset_bytes": key.size(0) * total_bytes}
