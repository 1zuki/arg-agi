# =====================================================================
# GPT-style token world model (Stage 2 of the IRIS-style WM).
#
# Operates on the tokenizer's output. One transition is laid out as a
# single causal sequence:
#
#   [ f_0 .. f_63 ][ a ][ g_0 .. g_63 ]      length = 64 + 1 + 64 = 129
#     current frame   |     next frame
#                  action
#
# A causal transformer predicts, teacher-forced:
#   - g_0..g_63   next-frame tokens   (head over the codebook)        SHARP
#   - reward      scalar              (head at the action position)
#   - state       4-way: NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER  <- asked for
#
# Combined vocab so one embedding table covers both kinds of token:
#   frame tokens  -> ids [0, CODEBOOK)
#   action tokens -> ids [CODEBOOK, CODEBOOK + N_ACTIONS)
#
# Imagination: feed [current-frame tokens, action], then autoregressively
# sample g_0..g_63, decode via the tokenizer -> a SHARP predicted frame,
# plus predicted reward and game state. No engine / agent imports.
# =====================================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import N_TOKENS, CODEBOOK

N_ACTIONS = 8                       # RESET + ACTION1..7  (ids 0..7)
VOCAB = CODEBOOK + N_ACTIONS        # frame tokens + action tokens
SEQ = 2 * N_TOKENS + 1             # 129: cur frame, action, next frame
ACTION_POS = N_TOKENS              # index of the action token in the sequence

# 4 game states, fixed order (matches arcengine GameState)
STATE_NAMES = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]
N_STATES = len(STATE_NAMES)

D_MODEL = 256
N_LAYER = 4
N_HEAD = 4
DROPOUT = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_head=N_HEAD):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)        # [B,h,T,hd]
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TokenWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ, D_MODEL))
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head_tok = nn.Linear(D_MODEL, CODEBOOK)     # next-frame token logits
        self.head_rew = nn.Linear(D_MODEL, 1)            # reward at action pos
        self.head_state = nn.Linear(D_MODEL, N_STATES)   # game state at action pos
        nn.init.normal_(self.pos_emb, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            w = m.weight
            nn.init.normal_(w, std=0.02)

    def _backbone(self, ids):
        """ids [B,T] combined-vocab tokens -> hidden [B,T,D]."""
        T = ids.size(1)
        x = self.drop(self.tok_emb(ids) + self.pos_emb[:, :T])
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    @staticmethod
    def build_sequence(cur_tokens, action_ids, next_tokens=None):
        """cur_tokens [B,64], action_ids [B] (0..7), next_tokens [B,64] or None.
        Returns combined-vocab id sequence: full [B,129] if next given, else
        the [B,65] prefix (cur frame + action) for imagination."""
        B = cur_tokens.size(0)
        a = (action_ids.long() + CODEBOOK).view(B, 1)        # shift into action range
        if next_tokens is None:
            return torch.cat([cur_tokens, a], dim=1)         # [B,65]
        return torch.cat([cur_tokens, a, next_tokens], dim=1)  # [B,129]

    def forward(self, cur_tokens, action_ids, next_tokens):
        """Teacher-forced training pass. Returns (tok_logits, reward, state_logits).
        tok_logits [B,64,CODEBOOK] are predictions for g_0..g_63."""
        ids = self.build_sequence(cur_tokens, action_ids, next_tokens)   # [B,129]
        h = self._backbone(ids)
        # positions ACTION_POS .. ACTION_POS+63 predict next-frame tokens g_0..g_63
        h_next = h[:, ACTION_POS:ACTION_POS + N_TOKENS]                  # [B,64,D]
        tok_logits = self.head_tok(h_next)
        h_act = h[:, ACTION_POS]                                         # [B,D]
        reward = self.head_rew(h_act).squeeze(-1)
        state_logits = self.head_state(h_act)
        return tok_logits, reward, state_logits

    @torch.no_grad()
    def imagine(self, cur_tokens, action_ids, sample=False, temperature=1.0):
        """Autoregressively roll out the next frame.
        Returns (next_tokens [B,64], reward [B], state_idx [B], state_logits [B,4])."""
        self.eval()
        B = cur_tokens.size(0)
        seq = self.build_sequence(cur_tokens, action_ids)               # [B,65]
        # reward + state read off the action position once
        h = self._backbone(seq)
        reward = self.head_rew(h[:, ACTION_POS]).squeeze(-1)
        state_logits = self.head_state(h[:, ACTION_POS])
        state_idx = state_logits.argmax(-1)
        # then generate 64 next-frame tokens
        for _ in range(N_TOKENS):
            h = self._backbone(seq)
            logits = self.head_tok(h[:, -1])                            # predict next g
            if sample:
                p = F.softmax(logits / max(1e-6, temperature), dim=-1)
                nxt = torch.multinomial(p, 1)                           # [B,1]
            else:
                nxt = logits.argmax(-1, keepdim=True)                   # [B,1]
            seq = torch.cat([seq, nxt], dim=1)
        next_tokens = seq[:, -N_TOKENS:]
        return next_tokens, reward, state_idx, state_logits
