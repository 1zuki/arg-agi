import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 32
N_COLORS = 16
N_STATES = 4

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
    )

class Ls20WorldModel(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, CH)
        
        # GUI encoder (nhận vào 9x64 one-hot)
        self.gui_enc = nn.Sequential(
            nn.Conv2d(N_COLORS, CH, 3, stride=2, padding=1), nn.ReLU(), # 5x32
            nn.Conv2d(CH, CH*2, 3, stride=2, padding=1), nn.ReLU(),     # 3x16
            nn.Flatten(),
            nn.Linear(CH*2 * 3 * 16, CH*4), nn.ReLU()
        )
        
        # Board Encoder (U-Net)
        # Input channels = CH (từ board)
        self.enc1 = _block(CH, CH)
        self.enc2 = _block(CH, CH * 2)
        
        self.bott = _block(CH * 2, CH * 2)
        
        # Action Embedding (8 hành động)
        self.act_emb = nn.Embedding(8, CH * 4)
        
        # FiLM: GUI feature + action_emb -> modulate bottleneck
        self.film = nn.Linear(CH*4, CH*2 * 2) 

        self.up1 = nn.ConvTranspose2d(CH * 2, CH * 2, 2, stride=2) 
        self.dec1 = _block(CH * 3, CH) # CH*2 (từ up) + CH (từ enc1)

        self.head_board = nn.Conv2d(CH, num_tokens, 1)
        
        # GUI decoder (từ bottleneck)
        self.gui_dec = nn.Sequential(
            nn.Linear(CH*2, CH*4), nn.ReLU(),
            nn.Linear(CH*4, N_COLORS * 9 * 64)
        )
        
        self.head_rew = nn.Linear(CH*2, 1)
        self.head_state = nn.Linear(CH*2, N_STATES)

    def forward(self, board, gui, action_id):
        """
        board: [B, 11, 12] ints (Token IDs)
        gui: [B, 9, 64] ints 0-15
        action_id: [B] ints (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
        """
        B, H, W = board.shape
        
        # Encode GUI
        gui_onehot = F.one_hot(gui.long(), N_COLORS).permute(0, 3, 1, 2).float()
        gui_feat = self.gui_enc(gui_onehot) # [B, 4CH]
        
        # Encode Action
        act_feat = self.act_emb(action_id.long()) # [B, 4CH]
        
        # Encode Board
        x = self.token_emb(board.long()).permute(0, 3, 1, 2) # [B, CH, 11, 12]
        
        e1 = self.enc1(x) # [B, CH, 11, 12]
        
        # Pool custom vì 11x12 không chẵn
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2, padding=(1,0)) # padding để H từ 11 thành 12 -> pool còn 6x6
        e2 = self.enc2(p1) # [B, 2CH, 6, 6]
        
        b = self.bott(e2) # [B, 2CH, 6, 6]
        
        # FiLM: Dùng thông tin GUI và Action để modulate board bottleneck
        gamma_beta = self.film(gui_feat + act_feat) # [B, 4CH]
        gamma, beta = gamma_beta.chunk(2, dim=-1) # [B, 2CH]
        b = b * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        
        # Decode Board
        u1 = self.up1(b) # [B, 2CH, 12, 12]
        u1 = u1[:, :, :H, :W] # crop lại thành 11x12
        
        d1 = self.dec1(torch.cat([u1, e1], dim=1)) # [B, CH, 11, 12]
        next_board_logits = self.head_board(d1) # [B, N_TOKENS, 11, 12]
        
        # Pool bottleneck để đoán reward, state và next_gui
        pooled = b.mean(dim=[2, 3]) # [B, 2CH]
        
        next_gui_logits = self.gui_dec(pooled).view(B, N_COLORS, 9, 64)
        reward = self.head_rew(pooled).squeeze(-1)
        state_logits = self.head_state(pooled)
        
        return next_board_logits, next_gui_logits, reward, state_logits

class Ls20ValueNet(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, CH)
        self.gui_enc = nn.Sequential(
            nn.Conv2d(N_COLORS, CH, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(CH * 5 * 32, CH*2), nn.ReLU()
        )
        self.board_enc = nn.Sequential(
            nn.Conv2d(CH, CH*2, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(CH*2 * 11 * 12 + CH*2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board, gui):
        gui_onehot = F.one_hot(gui.long(), N_COLORS).permute(0, 3, 1, 2).float()
        g_feat = self.gui_enc(gui_onehot)
        
        b_emb = self.token_emb(board.long()).permute(0, 3, 1, 2)
        b_feat = self.board_enc(b_emb)
        
        x = torch.cat([b_feat, g_feat], dim=1)
        return self.head(x).squeeze(-1)
