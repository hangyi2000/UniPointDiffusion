import torch
import torch.nn as nn


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, cate=None, batch=None, key=None, pcclass=None):
        if key is None:
            key = self.key
        output = self.embedding(c)
        return output
    

if __name__ == "__main__":
    class_embedder = ClassEmbedder(768, n_classes=10).cuda()
    cate =  torch.tensor([1, 4]).cuda()
    c = class_embedder(cate=cate)
    print(c)

    for name, param in class_embedder.named_parameters():
        if param.requires_grad:
            print(name)

