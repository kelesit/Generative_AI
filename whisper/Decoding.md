# Decoding.py



## DecodingTask

接收一个whisper模型类和DecodingOptions类作为输入，提供一个run方法来执行解码过程。在解码过程中，使用模型进行音频特征提提取和解码操作，并返回解码结果。

### init

```python
def __init__(self, model: "Whisper", options: DecodingOptions):
```

初始化DecodingTask

1. model, tokenizer, options, n_group(beam搜索的组数), n_ctx(文本上下文长度), sample_len(每个解码步骤中生成的文本长度)
2. 根据option获取起始标记的序列。作为`self.sot_index`
3. 根据模型和起始标记的长度，创建一个PyTorchInference实例，用于执行解码过程中的向前传播。作为`self.inference`
4. 创建一个MaximumLikelihoodRanker实例，用于对一组采样序列进行排序。
5. 根据解码选项中的设置，创建一个解码器实例。如果指定了beam搜索的大小（beam_size），则创建一个BeamSearchDecoder实例；否则，创建一个GreedyDecoder实例。作为`self.decoder`
6. 创建并应用一系列logit过滤器，以对生成的标记进行抑制或惩罚。
   1. 创建一个空的self.logit_filters列表，用于存储logit过滤器的实例。
   2. 根据解码选项中的设置，判断是否需要抑制空白标记（suppress_blank）。如果需要抑制空白标记，则创建一个SuppressBlank过滤器的实例，并将其添加到self.logit_filters列表中。该过滤器会将生成的空白标记的logits设置为负无穷
   3. 判断是否需要抑制特定的标记（suppress_tokens）。如果需要抑制特定的标记，则创建一个SuppressTokens过滤器的实例，并将其添加到self.logit_filters列表中。该过滤器会将指定的标记的logits设置为负无穷。通过调用self._get_suppress_tokens()方法，获取需要抑制的标记列表。
   4. 判断是否需要应用时间戳规则（without_timestamps）。如果不需要应用时间戳规则，则跳过以下步骤。
      1. 根据音频上下文的长度和时间戳的精度，计算出每个时间戳的持续时间。通常，时间戳的精度为0.02秒。
      2. 判断是否设置了最大初始时间戳（max_initial_timestamp）。如果设置了最大初始时间戳，则计算出对应的时间戳索引。计算方法是将最大初始时间戳除以时间戳的持续时间，并四舍五入到最接近的整数。
      3. 接下来，创建一个ApplyTimestampRules过滤器的实例，并将其添加到self.logit_filters列表中。该过滤器会根据解码选项中的设置，对logits进行时间戳规则的应用。它使用tokenizer、self.sample_begin和max_initial_timestamp_index作为参数。

### run

```python
def run(self, mel: Tensor) -> List[DecodingResult]:
```

接收一个mel频谱图作为输入，并返回解码结果

1. 对Mel频谱图进行预处理：`_get_audio_features`（将Mel频谱图转化为audio_features: Tensor )
2. 判断语言：`_detect_language`，并检查任务是否为判断语言，若是则返回结果
3. 扩展tokens张量的形状，并将其移动到与audio_features张量相同的设备上，以支持解码过程中的束搜索或最佳N采样。
4. 调用`_main_loop`方法来执行解码循环。它接受音频特征audio_features和初始的token序列tokens作为输入，并返回解码过程中生成的token序列、对数概率和无语音概率。并进行处理重塑，得到最终*candidates for each group, and slice between the first sampled token and EOT*
5. *select the top-ranked sample in each group*最后返回结果



### _main_loop

```python
def _main_loop(self, audio_features: Tensor, tokens: Tensor):
```

函数接受两个参数：audio_features和tokens。audio_features是音频特征的张量，tokens是当前的文本序列的张量.

通过一个循环迭代self.sample_len次，进行解码的每个步骤。在每个步骤中，首先调用self.inference.logits方法，传入tokens和audio_features，获取当前步骤的logits（对数概率）。

如果是第一步（i == 0），并且存在self.tokenizer.no_speech，则保存起始标记处的无语音概率。这是为了后续评估解码结果时使用。

对logits应用一系列的logit过滤器，例如用于抑制或对logits应用惩罚的过滤器。这些过滤器可以根据解码选项中的设置进行自定义。

然后，使用self.decoder.update方法，根据当前的tokens、logits和sum_logprobs，更新文本序列。这个方法会根据解码选项中的设置，选择下一个token，并将其添加到tokens中。同时，更新sum_logprobs以累积对数概率。

在每个步骤的末尾，检查是否已完成解码或文本序列的长度是否超过了self.n_ctx的限制。如果满足任一条件，则跳出循环，结束解码过程。

最后，在finally块中，调用self.inference.cleanup_caching()方法，清理解码过程中的缓存。

函数返回三个结果：tokens，sum_logprobs和no_speech_probs。tokens是最终的文本序列的张量，sum_logprobs是累积的对数概率的张量，no_speech_probs是无语音概率的列表。



## decode

decoding.py中的顶级函数，用于对30秒音频片段进行解码，输入为Mel频谱图。

```python
@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
```

**函数接受几个参数**：
\- model：Whisper模型的实例。
\- mel：形状为(80, 3000)或(*, 80, 3000)的Mel频谱图张量，表示一个或多个音频片段。
\- options：一个DecodingOptions的数据类实例，包含解码30秒片段所需的所有选项。
\- **kwargs：额外的关键字参数，可以用于覆盖options中的选项。

**函数的返回类型**: Union[DecodingResult, List[DecodingResult]]，表示解码结果，可以是单个DecodingResult实例或DecodingResult实例的列表。



1. 检查输入的Mel频谱图维度，若为2，则表示只有一个音频片段，需要将其扩展为形状为(1, 80, 3000)的张量。
2. 若存在额外关键字参数kwargs, 则使用replace 函数将其应用与options，以覆盖默认选项。
3. 创建一个DecodingTask实例，传入model和options，并调用其run方法对Mel频谱图进行解码。
4. 最后根据输入的Mel频谱图是否只有一个片段，返回解码结果。如果只有一个片段，则返回结果列表中的第一个元素；否则，返回整个结果列表。



## PytorchInference

用于在推理过程中进行Pytorch模型的向前传播和缓存管理

### init

接受两个参数：Whisper的模型实例，initial_token_length整数，表示初始令牌的长度

### logits

```python
def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
```

1. 首先检查kv_cache是否为空，如果为空，则调用model的install_kv_cache_hooks方法来初始化kv_cache和hooks。
2. 根据tokens的形状调整令牌的长度，只保留最后一个令牌（除非在第一次前向传播时）。
3. 调用model的decoder方法，传递tokens、audio_features和kv_cache参数，返回模型的logits。



## GreedyDecoder

### init

接受两个参数：temperature和eot。

temperature是一个浮点数，用于控制贪婪程度，当temperature为0时，选择具有最高logits值的标记；当temperature大于0时，根据logits的概率分布进行随机采样。eot是一个整数，表示结束标记。

### update

接收tokens、logits和sum_logprobs作为输入，返回更新后的tokens序列和一个表示是否所有序列都已完成的布尔值completed。

1. 首先根据temperature的值选择下一个标记。如果temperature为0，则选择具有最高logits值的标记；否则，根据logits的概率分布进行随机采样。
2. 使用F.log_softmax函数对logits进行softmax操作，并指定dim=-1表示在最后一个维度上进行softmax。这将得到一个概率分布，表示每个标记的概率。
3. 使用torch.arange(logprobs.shape[0])生成一个序列，用于索引logprobs的第一个维度。这个序列的长度与logprobs的第一个维度相同，用于选择每个序列中下一个标记的log概率。
4. 通过next_tokens索引logprobs，获取每个序列中下一个标记的log概率。这是通过使用torch.arange(logprobs.shape[0])作为行索引，next_tokens作为列索引来实现的。
5. 将当前标记的log概率乘以一个布尔值(tokens[:, -1] != self.eot)，用于判断是否需要将当前标记的log概率加入到累积log概率sum_logprobs中。这个布尔值是通过比较tokens的最后一个标记是否等于结束标记self.eot来得到的。
6. 检查tokens中的最后一个token是否为结束标记（self.eot）来确定哪些序列已经完成。如果是，则将next_tokens中对应位置的token设置为结束标记。
7. 使用torch.cat函数将next_tokens中的token添加到tokens序列的末尾，形成更新后的tokens序列。
8. 通过检查更新后的tokens序列中的最后一个token是否全部为结束标记，确定是否所有序列都已经完成。如果是，则将completed设置为True，否则为False



### finalize

确保每一个序列末尾都至少有一个EOF标记，同时将sum_logprobs转换为python列表.