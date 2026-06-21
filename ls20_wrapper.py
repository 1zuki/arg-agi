import numpy as np
from arc_agi import Arcade, OperationMode

class BlockTokenizer:
    def __init__(self):
        self.block_to_id = {}
        self.id_to_block = {}
        self.next_id = 0

    def tokenize(self, block: np.ndarray) -> int:
        """Nhận vào mảng 5x5, trả về ID."""
        key = block.tobytes()
        if key not in self.block_to_id:
            self.block_to_id[key] = self.next_id
            self.id_to_block[self.next_id] = block.copy()
            self.next_id += 1
        return self.block_to_id[key]

    def detokenize(self, token_id: int) -> np.ndarray:
        return self.id_to_block.get(token_id, np.zeros((5, 5), dtype=int))

    def __len__(self):
        return self.next_id


class Ls20Wrapper:
    def __init__(self, env):
        self.env = env
        self.tokenizer = BlockTokenizer()
        
        # Cấu hình grid đã phân tích
        self.start_x = 4
        self.start_y = 0
        self.board_w = 60 # 12 blocks * 5
        self.board_h = 55 # 11 blocks * 5
        
        self.gui_start_y = 55

    def reset(self):
        obs = self.env.reset()
        if obs is None:
            obs = getattr(self.env, "observation_space", None)
            if hasattr(obs, "frame"):
                return self._process_frame(obs.frame)
        return self._process_frame(obs.frame) if obs else None

    def step_dir(self, action_id: int):
        """Action theo phím di chuyển: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT"""
        # Tìm Enum object chính xác trong action_space thay vì tự ép kiểu
        act_enum = next((a for a in self.env.action_space if a.value == action_id), None)
        if act_enum is None:
            return None, None, 0, True, {}
            
        obs = self.env.step(act_enum)
        
        if obs is None:
            return None, None, 0, True, {}
            
        board_tokens, gui_pixels = self._process_frame(obs.frame)
        return board_tokens, gui_pixels, 0, False, {}

    def step_raw(self, act_enum, data=None):
        obs = self.env.step(act_enum, data=data)
        if obs is None:
            return None, None, 0, True, {}
        board_tokens, gui_pixels = self._process_frame(obs.frame)
        return board_tokens, gui_pixels, 0, False, {}

    def _process_frame(self, frame_obj):
        arr = np.array(frame_obj)
        while arr.ndim > 2:
            arr = arr[-1]
        arr = np.clip(arr.astype(int), 0, 15)

        # Trích xuất board 55x60
        board_pixels = arr[self.start_y : self.start_y + self.board_h, 
                           self.start_x : self.start_x + self.board_w]
        
        blocks_y = self.board_h // 5  # 11
        blocks_x = self.board_w // 5  # 12
        
        board_tokens = np.zeros((blocks_y, blocks_x), dtype=int)
        
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = board_pixels[by*5 : by*5+5, bx*5 : bx*5+5]
                board_tokens[by, bx] = self.tokenizer.tokenize(block)
                
        # Trích xuất GUI: từ dòng 55 trở đi, rộng đủ 64
        gui_pixels = arr[self.gui_start_y:, :]
        
        return board_tokens, gui_pixels

    @property
    def num_tokens(self):
        return len(self.tokenizer)
