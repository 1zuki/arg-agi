import torch
from ls20_model import Ls20WorldModel

def expand():
    try:
        saved = torch.load("ls20_model.pt", map_location="cpu")
    except Exception as e:
        print("Lỗi đọc file:", e)
        return
        
    old_num_tokens = saved["num_tokens"]
    new_num_tokens = 1000
    
    if old_num_tokens >= new_num_tokens:
        print("Model đã có đủ token, không cần mở rộng.")
        return

    print(f"Mở rộng Vocab từ {old_num_tokens} -> {new_num_tokens} tokens...")

    # Load mô hình cũ
    old_model = Ls20WorldModel(num_tokens=old_num_tokens)
    old_model.load_state_dict(saved["model"])

    # Khởi tạo mô hình mới với 256 token
    new_model = Ls20WorldModel(num_tokens=new_num_tokens)
    state_dict = new_model.state_dict()
    old_state_dict = old_model.state_dict()

    # Sao chép tịnh tiến trọng số
    for k, v in old_state_dict.items():
        if "token_emb.weight" in k:
            state_dict[k][:old_num_tokens] = v
        elif "head_board.weight" in k:
            state_dict[k][:old_num_tokens] = v
        elif "head_board.bias" in k:
            state_dict[k][:old_num_tokens] = v
        else:
            state_dict[k] = v

    new_model.load_state_dict(state_dict)

    # Cập nhật và lưu lại
    saved["model"] = new_model.state_dict()
    saved["num_tokens"] = new_num_tokens
    
    torch.save(saved, "ls20_model.pt") # Ghi đè luôn file gốc
    print("Mở rộng thành công! Đã ghi đè lên ls20_model.pt")

if __name__ == "__main__":
    expand()
