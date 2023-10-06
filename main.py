"""
A from-scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my YouTube channel!


"""

import torch

from module.transformer import Transformer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = (
        Transformer(src_vocab_size, trg_vocab_size,
                    src_pad_idx, trg_pad_idx, device=device)
        .to(device)
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
    model.train()
    model_path = f'./parameter/model.pth'  # 定义保存模型的文件名
    torch.save(model.state_dict(), model_path)
