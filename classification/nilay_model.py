import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class LearnedPositionalEmbedding2(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe,std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Embedding2(nn.Module):

    def __init__(self, input_dim, max_len, dropout=0.1):
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding2(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):


    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class self_attention(nn.Module):

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=1):


        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob


        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken,std=0.02)


        self.feed_forward_hidden = hidden * 4

        self.embedding = Embedding2(input_dim=input_dim, max_len=max_len+1)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])



    def forward(self, input_vectors):
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()


        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, sample




####################################################################################################
#####      dual_paths - Implementation of the dual_paths for noisy and denoised images         #####
####################################################################################################

class dual_paths(nn.Module):
    def __init__(self, base_model, num_classes, dropout=0.8):
        super(dual_paths, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=dropout)

        self.linear = nn.Linear(in_features=2048, out_features=512, bias=True)
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(base_model.children())[:-2])

        max_length  = 8

        self.self_attention = self_attention(self.hidden_size, max_length , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)

        self.fc_action = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()


    def forward(self, x_noisy, x_denoised):
#        x_noisy, x_denoised = x             #(b,1,128,200,200)

        x_noisy = self.features(x_noisy)        #(b,2048,4,7,7)
        x_denoised = self.features(x_denoised)        #(b,2048,4,7,7)

        x_noisy = x_noisy.permute((0,2,3,4,1))     #(b,4,7,7,2048)
        x_noisy = self.linear(x_noisy)             #(b,4,7,7,512)
        x_noisy = x_noisy.permute((0,4,1,2,3))     #(b,512,4,7,7)

        x_denoised = x_denoised.permute((0,2,3,4,1))      #(b,4,7,7,2048)
        x_denoised = self.linear(x_denoised)              #(b,4,7,7,512)
        x_denoised = x_denoised.permute((0,4,1,2,3))      #(b,512,4,7,7)

        x_noisy = self.avgpool(x_noisy)                            #(b,512,4,1,1)
        x_noisy = x_noisy.view(x_noisy.size(0), self.hidden_size, 4)  #(b,512,4)
        x_noisy = x_noisy.transpose(1, 2)                        #(b,4,512)

        x_denoised = self.avgpool(x_denoised)                        #(b,512,4,1,1)
        x_deonised = x_denoised.view(x_denoised.size(0), self.hidden_size, 4)  #(b,512,4)
        x_denosied = x_denoised.transpose(1, 2)                       #(b,4,512)

        x_cat = torch.cat((x_noisy, x_denosied), 1)                  #(b,8,512)
        output, maskSample = self.self_attention(x_cat)            # output(b,9,512), maskSample(b,9)

        classificationOut = output[:,0,:]                #class(b,512)

        output=self.dp(classificationOut)           #(b,512)
        x = self.fc_action(output)                   #(b,11)
        return x

