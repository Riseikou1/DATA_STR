import math

class Value :
    def __init__(self,data,_children=(),_op="",label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data} , grad={self.grad})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data , (self,other),op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data , (self,other),op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only supporting int or float type!"
        out = Value(self.data ** other,(self,),f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self),"exp")
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if not v in visited :
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(child)
        
        build_topo(self)

        self.grad = 1.0
        for node in (reversed(topo)):
            node._backward()

    def __neg__(self):
        return self * -1
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)
    
    def __rmul__(self,other):
        return self * other
    
    def __div__(self,other):
        return self * other ** -1
    
import random

class Neuron :
    def __init__(self,nin):

        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return act
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer :
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP :
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1])for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers :
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
ys = []
xs = []
n = MLP()

for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum((yout-ygt)**2 for yout, ygt in zip(ys,ypred))

    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    for p in n.parameters:
        p.data += 0.005 * p.grad

words = open("")
b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0) + 1

chars = sorted(list(set("".join(words))))

sorted(b.items(),key = lambda kv : -kv[1])
stoi = {s : i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

import math as torch

N = torch.zeros((27,27),dtype=torch.int32)

for w in words :
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N,cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j,i,chstr,ha='center',va='bottom',color='gray')
        plt.text(j,i,N[i,j].item(),ha='center',va='top',color='gray')
plt.axis("off")

P = (N+1).float()
P /= P.sum(1,keepdims=True)

g = torch.Generator().manual_seed(5112)
for i in range(15):
    out = list()
    ix = 0
    while True :
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))

log_likelihood = 0.0
n = 0

for w in words :
    chs = ["."] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n+=1

xs,ys = [],[]

for w in words :
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

import numpy as F
xenc = F.one_hot(xs,num_classes=27).float()

W = torch.randn((27,27),generator=g,requires_grad=True)
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1,keepdims=True)

loss = -probs[torch.arange(5),ys[:5]].log().mean()
xs = torch.tensor(xs)
ys = torch.tensor(ys)

for k in range(10):
    xenc = F.one_hot(xs,num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / logits.sum(1,keepdims=True)
    loss = -probs[torch.arange(xs.element),ys].log().mean() + 0.01 * (W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad

for i in range(15):
    out = []
    ix = 0
    while True :
        xenc = F.one_hot(torch.tensor([ix]),num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / logits.sum(1,keepdims=True)

        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


def build_dataset(block_size=3,n=len(words)):
    X,Y = [],[]
    for w in words[:n]:
        context = [0]*block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            print('.'.join(itos[i] for i in context),'--->',itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

C = torch.randn((27,2))

X,Y= build_dataset()
emb = C[X]
w1 = torch.randn((6,100))
b1 = torch.randn(100)

h = torch.tanh(emb.view(-1,6)@ w1 +b1)

w2 = torch.randn((100,27))
b2 = torch.randn(27)

logits = h @ w2 + b2

counts = logits.exp()
prob = counts / logits.sum(1,keepdim=True)

g = torch.Generator().manual_seed(43141)
C = torch.randn((27,2),generator=g)
W1 = torch.randn((6,100),generator=g)
b1 = torch.randn(100,generator=g)
W2 = torch.randn((100,27),generator=g)
b2 = torch.randn(27,generetor=g)
parameters = [C,W1,b1,W2,b2]

for p in parameters :
    p.requires_grad = True

learning_rate_exp = torch.linspace(-3,0,1000)
lrs = 10 ** learning_rate_exp

lossi = []
lri = []

for i in range(1000):
    ix = torch.randn(0,X.shape[0],(32,))
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1,6)@ W1 +b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Y[ix])

    for p in parameters :
        p.grad = None

    loss.backward()
    lr = lrs[i]
    for p in parameters :
        p.data += -lr * p.grad

    lri.append(lr)
    lossi.append(loss.item())

print(loss.item())

import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,Ytr = build_dataset(words[:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:])


lri = []
lossi = []
stepi = []

for i in range(200000):
    ix = torch.randint(0,Xtr.shape[0],(32,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1,30)@W1 +b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Ytr[ix])

    for p in parameters :
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01
    for p in parameters :
        p.data += -lr * p.grad
    
    lossi.append(loss.log10().item())
    stepi.append(i)

print(loss.item())

plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data,C[:1].data,s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(),C[i,1].item(),itos[i],ha='center',va='center',color='white')

plt.grid("minor")

block_size = 3
g = torch.Generator().manual_seed(32532)
for _ in range(20):
    out = []
    context = [0]*block_size
    while True :
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1,generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)

n_embed = 10
n_hidden = 200

bngain = torch.ones((1,n_hidden))
bnbias = torch.zeros((1,n_hidden))
bnmean_running = torch.zeros((1,n_hidden))
bnstd_running = torch.ones((1,n_hidden))

parameters = [C,W1,b1,W2,b2,bngain,bnbias]

max_steps = 20000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]

    embcat = emb.view(emb.shape[0],-1)
    hpreact = embcat @ W1 + b1
    bnmeani = hpreact.mean(0,keepdims=True)
    bnstdi = hpreact.std(0,keepdims=True)

    hpreact = bngain * (hpreact-bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Yb)

    for p in parameters :
        p.grad = None
    loss.backward()

    for p in parameters :
        p.data += -lr * p.grad


class Linear :
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight = torch.randn((fan_in,fan_out),generator=g) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        self.out = x @ self.weight
        if self.bias is not None :
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self,x):
        if self.training :
            xmean = x.mean(0,keepdim=True)
            xvar = x.var(0,keepdim=True)
        else :
            xmean = self.running_mean
            xvar = self.running_var

        xhat  = (x-xmean) / torch.sqrt(xvar+self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training :
            self.running_mean = (1-self.momentum)* self.running_mean + self.momentum * xmean

        return self.out
    
    def parameters(self):
        return [self.gamma,self.beta]

class Tanh :
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
vocab_size = 27
C = torch.randn((vocab_size,n_embed))
layers = [
    Linear(n_embed*block_size,n_hidden,bias=False) , BatchNorm1d(n_hidden),Tanh()
]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer.Linear):
            layer.weight *= 5/3

def cmp(s,dt,t):
    ex = torch.all(dt==t.grad).item()
    app = torch.allclose(dt,t.grad)
    maxdiff = (dt-t.grad).abs().max().item()
    print(f"{s:15s} | exact : {str(ex):5s} | approximate : {str(app):5s} | maxdiff : {maxdiff}")



emb = C[Xb]
embcat = emb.view(emb.shape[0],-1)
hprebn = embcat @ W1 + b1
bnmeani = 1/n * hprebn.sum(0,keepdim=True)
bndiff = hprebn = bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1)*(bndiff2).sum(0,keepdims=True)
bnvar_inv = (bnvar + 1e-5)** 0.-5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)
logits_maxes = logits.max(1,keepdim=True).values
norm_logits = logits - logits_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1,keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n),Yb].mean()

class Linear :
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight = torch.randn((fan_in,fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        self.out = x @ self.weight
        self.out = self.out + self.bias if self.bias else self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps = eps
        self.training = True
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)

    def __call__(self,x):
        if self.training :
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim,keepdim=True)
            xvar = x.var(dim,keepdim=True)
        else :
            xmean = self.running_mean
            xvar = self.running_std
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training :
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
    
    def parameters(self):
        return [self.gamma,self.beta]

class Tanh:
    def __init__(self,x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
class Embedding :
    def __init__(self,num_embeddings,embedding_dim):
        self.weight = torch.randn((num_embeddings,embedding_dim))
    
    def __call__(self,IX):
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]
    
class FlattenConsecutive:
    def __init__(self,n):
        self.n = n

    def __call__(self,x):
        B,T,C = x.shape
        x = x.view(B,T//self.n,C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []
    
class Sequential:
    def __init__(self,layers):
        self.layers = layers

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
            self.out = X
            return self.out
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n_embed= 24
n_hidden = 128
model = Sequential([
    Embedding(vocab_size,n_embed),
    FlattenConsecutive(2),Linear(n_embed*2,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    Linear(n_hidden,vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
for p in parameters():
    p.requires_grad = True

text = words

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

n_embed = 384
batch_size = 64
block_size = 256
eval_iters = 500
max_iters = 5001
learning_rate = 3e-4
device = "cuda" if torch.cuda._is_available() else "cpu"
eval_iters = 200
n_layer = 6
n_head = 6
dropout = 0.2

def get_batch(split):
    data = train_data  if split == "train" else val_data
    ix = torch.ranint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:block_size+1+i] for i in ix])
    return x,Y

import math as nn

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size,n_embed)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout)
        ])
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return X

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,block_size)
        self.blocks = nn.Sequential(*[Block(n_embed,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T))
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None :
            loss = None
        else :
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx
    
model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

for iter in range(max_iters):
    if iter % eval == 0:
        losses = estimate_loss()

    xb,yb = get_batch("train")

    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context,max_new_tokens=2000)[0].tolist()))


def bubbleSort(A):
    n = len(A)
    for i in range(n-1):
        flag = False
        for j in range(1,n-i):
            if(A[j-1]>A[j]):
                A[j-1],A[j] = A[j],A[j-1]
                flag = True
        if not flag :
            break

def bubbleSort(A):
    n = len(A)
    for i in range(n-1):
        flag = False
        for j in range(1,n-i):
            if A[j-1] > A[j]:
                A[j-1],A[j] = A[j], A[j-1] 
                flag = True
        if not flag :
            break

def selectionSort(A):
    n = len(A)
    for i in range(n-1):
        minidx = i
        for j in range(i+1,n):
            if A[j] < A[minidx]:
                minidx = j
        
        if minidx != i:
            A[i], A[minidx] = A[minidx], A[i]

class SortedArray:
    def __init__(self,capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity
    
    def __str__(self):
        return str(self.array[:self.size])
    
    def contains(self,e):
        for i in range(self.size):
            if self.array[i] == e:
                return True
        return False
    
    def insert(self,e):
        if self.contains(e) or self.isFull():
            return 
        self.array[self.size] = e
        self.size += 1

        for i in range(self.size-1,0,-1):
            if self.array[i-1] < self.array[i]:
                break
            self.array[i-1],self.array[i] = self.array[i] , self.array[i-1]

    def delete(self,e):
        if not self.contains(e) or self.isEmpty():
            return
        
        i = 0
        while self.array[i] < e:
            i += 1
        self.size -= 1
        while i < self.size :
            self.array[i] = self.array[i+1]
            i += 1
        self.array[self.size] = None

    def union(self,setB):
        setC = SortedArray()
        i = j = 0
        while i < self.size and j < setB.size:
            a = self.array[i]
            b = setB.array[j]
            if a == b :
                setC.insert(a)
                i += 1 ; b += 1
            elif a < b :
                setC.insert(a)
                i += 1
            else :
                setC.insert(b)
                j += 1
        
        while i < self.size :
            setC.insert(self.array[i])
            i += 1
        while j < self.size :
            setC.insert(setB.array[j])
            j += 1
        
        return setC
    

import random
def insertionSort(A):
    n = len(A)
    for i in range(1,n):
        key = A[j]
        j = i-1
        while j>= 0 and A[j] > key :
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key

def insertionSort(A):
    n = len(A)
    for i in range(1,n):
        key = A[i]
        j = i -1
        while j >= 0 and A[j] > key:
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key

def seqSearch(list,key):
    n = len(list)
    for i in range(n):
        if list[i] == key :
            return i
        else :
            return -1


def iBinary(A,key):
    low = 0
    high = len(A)-1
    while(low <= high):
        middle = (high + low) //2 
        if A[middle] == key :
            return middle
        elif A[middle] < key :
            low = middle +1
        else : 
            high = middle - 1
    
    return -1

def rBinary(A,key,low,high):
    if low <= high :
        middle = (low + high) // 2

        if key == A[middle] :
            return middle
        elif key < A[middle]:
            return rBinary(A,key,low,middle-1)
        else :
            return rBinary(A,key,middle+1, high)
    
    return -1

class Node :
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

M = 13
class HashTable :
    def __init__(self):
        self.table = [None] * M
    
    def HashFn(self,key):
        return key % M
    
    def insert(self,key):
        bucket = self.HashFn(key)

        node = Node(key)
        node.next = self.table[bucket]
        self.table[bucket] = node

    def display(self):
        for i in range(M):
            print(i)

    def insert(self,key):
        bucket = self.HashFn(key)
        node = Node(key)
        node.next = self.table[bucket]
        self.table[bucket] = node

    def display(self):
        for i in range(M):
            print()
            n = self.table[i]
            while n is not None :
                print(n.data,end=" ")
                n = n.next
            print()


class HashTable :
    def __init__(self):
        self.table = [0] * M

    def hashFn(self,key) :
        return k % M
    
    def hashFn2(self,key):
        return 11- (key%11)
    
    def insert(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            bucket = (hashVal + i) % M
            bucket = (hashVal + i**2) % M
            bucket = (hashVal+i * self.hashFn2(data))%M
            if self.table[bucket] == 0:
                self.table[bucket] = data
                break
    
    def search(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            bucket = (hashVal + i + self.hashFn2(data)) % M
            if self.table[bucket] == data :
                return bucket
        
    def delete(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            bucket = (hashVal + i + self.hashFn2(data)) % M
            bucket = (hashVal + i + self.hashFn2(data)) % M
            if self.table[bucket] == 0 :
                print("No key to delete.")
                break
            elif self.table[bucket] == data:
                self.table[bucket] -= 1
                return bucket
            
    def display(self):
        print("bucket data")
        for i in range(M):
            print()

    def hashFn(self,key):
        return key % M
    
    def hashFn(self,key):
        return 11- (key % 11)
    
import queue
class Node :
    def __init__(self,data,left= None,right=None):
        self.data = data
        self.left = left
        self.right = right

class BinaryTree :
    def __init__(self):
        self.root = None

    def postOrder(self,root):
        if root != None :
            self.postOrder(root.left)
            self.postOrder(root.right)
            print(root.data , end=" ")

    def inOrder(self,root):
        if root != None :
            self.inOrder(root.left)
            print(root.data , end= " ")
            self.inOrder(root.right)

    def levelOrder(self,root):
        Q = queue.Queue()
        Q.put(root)
        while not Q.empty():
            root = Q.get()
            print(root.data)
            if root.left != None :
                Q.put(root.left)
            if root.right != None :
                Q.put(root.right)

        print()

    def nodeCount(self,root):
        if root == None :
            return 0
        else :
            return 1 + self.nodeCount(root.left) + self.nodeCount(root.right)
    
    def isExternal(self,root):
        return root.left == None and root.right == None
    
    def leafCount(self,root):
        if root is None : return 0
        if self.isExternal(root) : return 1
        return self.leafCount(root.left) + self.leafCount(root.right)
    
    def getHeight(self,root):
        if root is None :
            return 0
        else :
            return 1 + max(self.getHeight(root.left),self.getHeight(root.right))
        
    def treeReverse(self,root):
        if root != None :
            root.left , root.right = root.right , root.left
            self.treeReverse(root.left)
            self.treeReverse(root.right)


N = 20
class MaxHeap :
    def __init__(self):
        self.heap = [None] * N
        self.heapsize = 0

    def upHeap(self):
        i = self.heapsize
        key = self.heap[i]
        while (i!=1) and key > self.heap[i//2] :
            self.heap[i] = self.heap[i//2]
            i = i//2
        self.heap[i]=key
    
    def insertItem(self,item):
        self.heapsize += 1
        self.heap[self.heapsize] = item
        self.upHeap()

    def downHeap(self):
        key = self.heap[1]
        p = 1
        c = 2
        while c <= self.heapsize :
            if(c<self.heapsize) and (self.heap[c+1]> self.heap[c]):
                c += 1
            if key >= self.heap[c] : break
            self.heap[p] = self.heap[c]
            p = C
            c *= 2
        self.heap[p] = key


    def deleteItem(self):
        key = self.heap[1]
        self.heap[1] = self.heap[self.heapsize]
        self.heapsize -= 1
        self.downHeap()
        return key
    
