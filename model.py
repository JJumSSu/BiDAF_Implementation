import torch
import torch.nn as nn
import torch.nn.functional as F

# Todo : Memory efficient implementation, dropout, initialization

class BiDAF(nn.Module):
    def __init__(self, args, char_vocab_size, glove):
        
        super(BiDAF, self).__init__()

        self.args = args
        self.device     =  torch.device("cuda:{}".format(self.args.GPU) if torch.cuda.is_available() else "cpu")
        self.char_emb   = nn.Embedding(char_vocab_size, self.args.Char_Dim, padding_idx=1) 
        self.char_conv  = nn.Conv2d(1, self.args.Char_Channel_Num, (self.args.Char_Channel_Width, self.args.Char_Dim))
        self.word_emb   = nn.Embedding.from_pretrained(glove, freeze = True) 
        self.hidden_dim = self.args.Char_Channel_Num + self.args.Word_Dim 

        self.highway = Highway(self.hidden_dim, self.hidden_dim, 2)

        self.context_LSTM = nn.LSTM(input_size = self.hidden_dim,
                                    hidden_size = self.hidden_dim,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = self.args.Dropout)

        self.att_weight_alpha = nn.Linear(self.hidden_dim * 6, 1)

        self.modeling_LSTM =  nn.LSTM(input_size = self.hidden_dim * 8,
                                      hidden_size = self.hidden_dim,
                                      bidirectional = True,
                                      batch_first = True,
                                      num_layers = 2,
                                      dropout = self.args.Dropout)

        self.p1_weight = nn.Linear(self.hidden_dim * 10, 1)
        self.p2_weight = nn.Linear(self.hidden_dim * 10, 1)
        
        self.output_LSTM = nn.LSTM(input_size = self.hidden_dim * 2,
                                   hidden_size = self.hidden_dim,
                                   bidirectional = True,
                                   batch_first = True,
                                   dropout = self.args.Dropout)

        self.dropout = nn.Dropout(p = self.args.Dropout)


    def forward(self, batch):

        def char_emb(x):
            
            batch_size = x.size()[0] # batch x seq_len x word_len
            x = self.char_emb(x) 
            x = x.view(-1, x.size(2), self.args.Char_Dim).unsqueeze(1) # (batch*seq_len) x 1 x word_len x char_dim
            x = self.char_conv(x).squeeze() 
            x = self.dropout(x)
            x = F.max_pool1d(x, x.size(2)).squeeze() 
            x = x.view(batch_size, -1, self.args.Char_Channel_Num) # batch x seq_len x char_channel_size 

            return x


        def coattention(c, q): # Todo: memory efficient

                shape = (c.size(0), c.size(1), q.size(1), c.size(2))

                ct  = c.unsqueeze(2).expand(shape)
                qt  = q.unsqueeze(1).expand(shape)
                cq  = torch.mul(ct, qt)
                
                S   = torch.cat([ct, qt, cq], dim = 3)
                del ct, qt, cq

                S   = self.att_weight_alpha(S).squeeze() # batch x c_seq_len x q_seq_len
                
                S1  = F.softmax(S, dim = 2)
                U   = torch.bmm(S1, q)

                del S1

                S2  = F.softmax(torch.max(S, dim = 2)[0], dim = 1).unsqueeze(dim = 1)
                
                del S

                h_t = torch.bmm(S2, c).expand(c.size(0), c.size(1), c.size(2))
                
                del S2
 
                h_t = torch.mul(c, h_t)
                
                out = torch.cat([c, U, torch.mul(c, U), torch.mul(c, h_t)], dim=2)
                

                return out


        def output_layer(g, m):
    
            # hidden = [torch.randn(2, batch_size, self.hidden_dim).to(self.device)] * 2

            p1 = F.log_softmax(self.dropout(self.p1_weight(torch.cat([g, m], dim=2))).squeeze(), dim =1)
            m2, _ = self.output_LSTM(m)
            p2 = F.log_softmax(self.dropout(self.p2_weight(torch.cat([g, m2], dim = 2))).squeeze(), dim = 1)
    
            return p1, p2
    
        # Input Represenation

        c_char = char_emb(batch.c_char)
        q_char = char_emb(batch.q_char)

        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])

        # Highway network

        c = self.highway(torch.cat([c_char, c_word], dim = 2))
        q = self.highway(torch.cat([q_char, q_word], dim = 2))
        del c_char, c_word, q_char, q_word

        # Contextual Embedding Layer

        c, _ = self.context_LSTM(c) 
        q, _ = self.context_LSTM(q)

        # Attention Flow
        
        g = coattention(c,q)

        # Modeling Layer
        
        m, _ = self.modeling_LSTM(g)
                
        # Output Layer 
        
        p1, p2 = output_layer(g, m)

        return p1, p2


class Highway(nn.Module):
    def __init__(self, input_size, output_size, num_layers):

        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear  = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_layers)])
        self.linear     = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_layers)])
        self.gate       = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_layers)])


    def forward(self, x):

        for layer in range(self.num_layers):
            gate      = torch.sigmoid(self.gate[layer](x))
            nonlinear = F.relu(self.nonlinear[layer](x))
            linear    = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
