# Model.py

- ModelDimensions: 存储模型的维度信息
- LayerNorm: 继承自n.LayerNorm的子类，对输入进行层归一化操作。
- Linear:  线性变换
- Con1d ：一维卷积操作
- sinusoids: 生成位置编码的正弦函数
- MultiHeadAttention: 多头注意力机制模块，进行自注意力或交叉注意力操作
- ResidualAttentionBlock: 残差注意力模块，包含多头注意力和前馈神经网络
- AudioEncoder: 音频编码模块，用于将音频特征编码为隐藏表示
- TextDecoder：文本编码器模块， 用于将隐藏表示为文本序列
- Whisper: 整个模块的定义， 包含了音频编码器和文本解码器



Whisper模型的输入是一个音频的Mel频谱图和一个文本序列， 输出时对应的文本序列的概率分布。



## class Whisper

start from class Whisper

### init

self.dims

self.encoder = AudioEncoder

self.decoder = TextDecoder

设置默认的对齐头部（alignment heads）。默认情况下，它使用模型的后半部分层进行对齐。具体来说，它将"alignment_heads"张量的后半部分设置为True，而前半部分保持为False。

对齐头部是用于在多头自注意力机制中选择要参与计算的头部。通过设置对齐头部，可以控制模型在不同层次上对输入进行注意力计算的方式。在这里，默认情况下，模型使用后半部分层进行对齐，以便更多地关注输入序列的后半部分。



### set_alignment_heads

接收一个字节类型的参数dump，该参数是经过压缩和编码的数据。将这个数据解压缩、解码，并将其转换为一个布尔类型的张量，然后将这个张量注册为模型的缓冲区。

通过执行这段代码，可以将预先计算好的对齐头部数据加载到模型中，以便在模型的注意力计算中使用。这个对齐头部数据可以通过外部生成或计算，并以字节形式传递给set_alignment_heads方法。



### embed_audio

```python
def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)
```

### logits

```python
def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
```

### forward

```python
def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))
```

### device

```python
@property
    def device(self):
        return next(self.parameters()).device
```

通过调用next()函数，我们可以获取迭代器的下一个元素，即第一个参数。模型的参数是以迭代器的形式存储的，我们需要获取其中的一个参数来获取它所在的设备（CPU或GPU）。通过获取第一个参数，我们可以确定模型所在的设备，并将其作为device属性的返回值。



### install_kv_cache_hooks

```python
def install_kv_cache_hooks(self, cache: Optional[dict] = None):
```

用于安装钩子（hooks）来保存MultiHeadAttention模块中计算的键（key）和值（value）张量，以便在后续计算中重复使用。该方法返回一个字典，其中存储了所有缓存（cache）以及必要的钩子对象列表。



cache : Dict[nn.Module, torch.Tensor]

​            A dictionary object mapping the key/value projection modules to its cache

hooks : List[RemovableHandle]

​            List of PyTorch RemovableHandle objects to stop the hooks to be called



1. 创建了一个空的缓存字典cache，如果传入了非空的缓存字典，则将其合并到cache中。

   `cache = {**cache} *if* cache is not None *else* {}`

   同时，创建了一个空的钩子列表hooks。

2. 定义了一个名为save_to_cache的函数，它是一个钩子函数。钩子函数的作用是在前向传播过程中拦截输出，并将其保存到缓存字典中。

   钩子函数接收三个参数：module表示当前模块，_表示输入，output表示输出。钩子函数首先检查当前模块是否已经在缓存字典中，如果不在或者输出的形状大于self.dims.n_text_ctx，则将输出直接保存到缓存字典中。否则，将输出与缓存字典中的值进行拼接，并使用detach方法将其从计算图中分离，然后保存到缓存字典中。

3. 定义了一个名为install_hooks的函数，它用于安装钩子。该函数接收一个模块作为参数，并检查该模块是否是MultiHeadAttention类型的模块。如果是，则将该模块的键和值的钩子函数注册到钩子列表中。

4. 使用self.decoder.apply(install_hooks)将install_hooks函数应用于self.decoder模块及其子模块。这样，钩子函数将被安装到所有的MultiHeadAttention模块中。



## RedsidualAttentionBlock

包含了自注意力机制(self-attention)和可选的交叉注意力机制(cross-attention)，以及多层感知机(MLP)。

### init

```python
def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
  self.attn = MultiHeadAttention(n_state, n_head)
  self.attn_ln = LayerNorm(n_state)
  self.cross_attn = (
    MultiHeadAttention(n_state, n_head) if cross_attention else None
  )
  self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

  n_mlp = n_state * 4	# 表示MLP的隐藏层维度
  self.mlp = nn.Sequential(
    Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
  )	#包含两个线性层和GELU激活函数的顺序模块。第一个线性层的输入维度是n_state，输出维度是n_mlp；第二个线性层的输入维度是n_mlp，输出维度是n_state。
  self.mlp_ln = LayerNorm(n_state)	# 定义了mlp_ln属性，它是一个LayerNorm模块，用于对MLP计算的结果进行层归一化。
```

### forward

首先对输入x进行自注意力计算，并将结果与x相加。

`x = x + self.attn(self.attn_ln(x), *mask*=mask, *kv_cache*=kv_cache)[0]`

self.attn模块来计算自注意力，通过调用self.attn_ln对输入进行层归一化。注意力计算的结果是一个元组，包含了注意力输出和注意力权重。在这里，我们只取注意力输出部分，即[0]索引。

如果cross_attn不为None，则对输入x进行交叉注意力计算，并将结果与x相加。这里使用了self.cross_attn模块来计算交叉注意力，通过调用self.cross_attn_ln对输入进行层归一化。交叉注意力计算的结果同样是一个元组，我们只取注意力输出部分。

最后，对经过注意力计算的结果x应用多层感知机（MLP）。首先通过self.mlp_ln对x进行层归一化，然后通过self.mlp模块进行线性变换和GELU激活函数的计算。最终，将MLP计算的结果与输入x相加，得到最终的输出。



