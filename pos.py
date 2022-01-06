import torch


# 构建2D正余弦位置嵌入
def build_2d_sincos_position_embedding(self, temperature=10000.):
    h, w = self.patch_embed.grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = self.embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
    pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
    pos_embed = torch.cat([pe_token, pos_emb], dim=1)
    return pos_embed