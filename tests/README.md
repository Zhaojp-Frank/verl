# Tests layout

Each folder under tests/ corresponds to a test category for a sub-namespace in verl. For instance:
- `tests/trainer` for testing functionality related to `verl/trainer`
- `tests/models` for testing functionality related to `verl/models`
- ...

There are a few folders with `special_` prefix, created for special purposes:
- `special_distributed`: unit tests that must run with multiple GPUs
- `special_e2e`: end-to-end tests with training/generation scripts
- `special_npu`: tests for NPUs
- `special_sanity`: a suite of quick sanity tests
- `special_standalone`: a set of test that are designed to run in dedicated environments

Accelerators for tests 
- By default tests are run with GPU available, except for the ones under `special_npu`, and any test script whose name ends with `on_cpu.py`.
- For test scripts with `on_cpu.py` name suffix would be tested on CPU resources in linux environment.

# Workflow layout

All CI tests are configured by yaml files in `.github/workflows/`. Here's an overview of all test configs:
1. A list of always triggered CPU sanity tests: `check-pr-title.yml`, `secrets_scan.yml`, `check-pr-title,yml`, `pre-commit.yml`, `doc.yml`
2. Some heavy multi-GPU unit tests, such as `model.yml`, `vllm.yml`, `sgl.yml`
3. End-to-end tests: `e2e_*.yml`
4. Unit tests
  - `cpu_unit_tests.yml`, run pytest on all scripts with file name pattern `tests/**/test_*_on_cpu.py`
  - `gpu_unit_tests.yml`, run pytest on all scripts with file without the `on_cpu.py` suffix.
  - Since cpu/gpu unit tests by default runs all tests under `tests`, please make sure tests are manually excluded in them when
    - new workflow yaml is added to `.github/workflows`
    - new tests are added to workflow mentioned in 2.
   
## 1) vLLM Rollout阶段generate_sequences()的返回输出

__相关文件__: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:generate_sequences()`

__返回的DataProto包含以下batch字段__:

- __prompts__: `[batch_size, prompt_length]` - 来自数据集的原始prompt token ids
- __responses__: `[batch_size, response_length]` - LLM生成的response token ids（包括生成的tokens和observation tokens）
- __response_mask__: `[batch_size, response_length]` - 1表示LLM生成的tokens，0表示observation/padding tokens
- __input_ids__: `[batch_size, prompt_length + response_length]` - 完整序列token ids（prompt + response）
- __attention_mask__: `[batch_size, prompt_length + response_length]` - 0表示padding tokens，1表示其他tokens
- __position_ids__: `[batch_size, prompt_length + response_length]` 或 `[batch_size, 3/4, prompt_length + response_length]` - 递增的position ids（对于qwen2vl等模型可能是3D/4D）
- __rollout_log_probs__ (可选): `[batch_size, response_length]` - rollout时计算的log概率（当config.rollout.calculate_log_probs=True时）

__meta_info字段__:

- __timing__: dict - 包含生成耗时统计

__重要函数调用链__:

1. `recipe/dapo/dapo_ray_trainer.py:RayDAPOTrainer.fit()` →
2. `verl/workers/fsdp_workers.py:ActorRolloutRefWorker.generate_sequences()` →
3. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:vLLMRollout.generate_sequences()`

## 2) DataParallelPPOActor.compute_log_prob()的输入输出

__相关文件__: `verl/workers/actor/dp_actor.py:compute_log_prob()`

__输入参数__:

- __data__: DataProto对象，包含:

  - `input_ids`: `[batch_size, sequence_length]` - 完整序列（prompt+response）
  - `attention_mask`: `[batch_size, sequence_length]` - attention mask
  - `position_ids`: `[batch_size, sequence_length]` - position ids
  - `responses`: `[batch_size, response_length]` - response部分
  - `multi_modal_inputs` (可选): 多模态输入数据

- __calculate_entropy__: bool - 是否计算entropy

__data.meta_info必需字段__:

- `micro_batch_size`: int - 微批次大小
- `temperature`: float - 温度参数
- `use_dynamic_bsz`: bool - 是否使用动态批次
- `max_token_len` (可选): int - 最大token长度

__输出__:

- __log_probs__: `[batch_size, response_length]` - response部分的log概率
- __entropys__: `[batch_size, response_length]` 或 None - entropy值（如果calculate_entropy=True）

## 3) compute_reward()的输入输出

__相关文件__: `verl/trainer/ppo/reward.py:compute_reward()`

__输入参数__:

- __data__: DataProto - 包含生成的序列和相关信息
- __reward_fn__: AbstractRewardManager - 奖励函数管理器

__输出__:

- __reward_tensor__: `[batch_size, response_length]` - token级别的奖励分数
- __reward_extra_infos_dict__: dict[str, list] - 额外的奖励信息（如准确率、通过率等）

## 4) apply_kl_penalty()的输入输出

__相关文件__: `verl/trainer/ppo/ray_trainer.py:apply_kl_penalty()`

__输入参数__:

- __data__: DataProto，必须包含:

  - `response_mask`: `[batch_size, response_length]`
  - `token_level_scores`: `[batch_size, response_length]` - 原始奖励分数
  - `old_log_probs`: `[batch_size, response_length]` - 当前策略的log概率
  - `ref_log_prob`: `[batch_size, response_length]` - 参考策略的log概率

- __kl_ctrl__: AdaptiveKLController - KL控制器

- __kl_penalty__: str - KL惩罚类型（"kl", "abs", "mse"等）

__输出__:

- __data__: DataProto - 更新后的数据，新增:
  - `token_level_rewards`: `[batch_size, response_length]` - 应用KL惩罚后的奖励

- __metrics__: dict - 包含:

  - `actor/reward_kl_penalty`: float - 当前KL散度
  - `actor/reward_kl_penalty_coeff`: float - KL系数beta

__计算公式__: `token_level_rewards = token_level_scores - beta * kld`

## 5) compute_grpo_outcome_advantage()的输入输出

__相关文件__: `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage()`

__输入参数__:

- __token_level_rewards__: `[batch_size, response_length]` - token级别奖励
- __response_mask__: `[batch_size, response_length]` - response mask
- __index__: `[batch_size]` - numpy数组，用于分组（相同prompt的不同响应有相同index）
- __epsilon__: float = 1e-6 - 数值稳定性参数
- __norm_adv_by_std_in_grpo__: bool = True - 是否用标准差归一化
- __config__: Optional[AlgoConfig] - 算法配置

__输出__:

- __advantages__: `[batch_size, response_length]` - 优势值
- __returns__: `[batch_size, response_length]` - 回报值（与advantages相同）

__计算逻辑__:

1. 计算每个序列的总奖励: `scores = token_level_rewards.sum(dim=-1)`

2. 按index分组，计算每组的均值和标准差

3. 对每个样本:

   - 如果`norm_adv_by_std_in_grpo=True`: `advantage = (score - mean) / (std + epsilon)`
   - 否则: `advantage = score - mean`

4. 广播到token维度: `advantages = advantage.unsqueeze(-1) * response_mask`

## 6) DataParallelPPOActor.update_policy()详细注释

__相关文件__: `verl/workers/actor/dp_actor.py:update_policy()`

```python
def update_policy(self, data: DataProto):
    # 设置模型为训练模式
    self.actor_module.train()
    
    # 从meta_info获取温度参数（必需，避免静默错误）
    temperature = data.meta_info["temperature"]
    
    # 选择需要的batch字段
    select_keys = [
        "responses",           # 生成的响应
        "response_mask",       # 响应掩码
        "input_ids",          # 完整输入序列
        "attention_mask",     # attention掩码
        "position_ids",       # 位置编码
        "old_log_probs",      # 旧策略的log概率
        "advantages",         # 优势值
    ]
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")  # 参考策略log概率
    if "rollout_is_weights" in data.batch.keys():
        select_keys.append("rollout_is_weights")  # 重要性采样权重
    
    # 选择数据
    data = data.select(batch_keys=select_keys, ...)
    
    # 分割为mini-batch用于PPO更新
    mini_batches = data.split(self.config.ppo_mini_batch_size)
    
    # 判断是否为on-policy（单个mini-batch且单个epoch）
    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
    
    metrics = {}
    
    # PPO的多个epoch训练
    for _ in range(self.config.ppo_epochs):
        for batch_idx, mini_batch in enumerate(mini_batches):
            # 进一步分割为micro-batch用于梯度累积
            if self.config.use_dynamic_bsz:
                # 动态批次：根据token数量分割
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                # 固定批次大小
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            
            # 清零梯度
            self.actor_optimizer.zero_grad()
            
            # 遍历每个micro-batch
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())  # 移到GPU
                micro_batch_metrics = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                
                # 提取必要的张量
                response_mask = model_inputs["response_mask"]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]
                
                # 获取配置参数
                entropy_coeff = self.config.entropy_coeff  # entropy系数
                loss_agg_mode = self.config.loss_agg_mode  # 损失聚合模式
                
                # 计算损失缩放因子（用于梯度累积）
                if self.config.use_dynamic_bsz:
                    loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                else:
                    loss_scale_factor = 1 / self.gradient_accumulation
                
                # 前向传播：计算当前策略的log_prob和entropy
                calculate_entropy = (entropy_coeff != 0)
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
                
                # 处理on-policy情况：使用当前log_prob作为old_log_prob
                if on_policy:
                    old_log_prob = log_prob.detach()
                else:
                    old_log_prob = model_inputs["old_log_probs"]
                
                # 获取策略损失函数
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                policy_loss_fn = get_policy_loss_fn(loss_mode)
                
                # 提取rollout重要性采样权重（如果存在）
                rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                
                # 计算策略损失（返回4个值）
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=rollout_is_weights,
                )
                
                # 添加entropy损失（如果启用）
                if entropy_coeff != 0:
                    entropy_loss = agg_loss(
                        loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                    )
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss
                
                # 添加KL损失（如果启用）
                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    kld = kl_penalty(
                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                
                # 应用损失缩放并反向传播
                loss = policy_loss * loss_scale_factor
                loss.backward()
                
                # 记录指标
                micro_batch_metrics.update({
                    "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                })
                append_to_dict(metrics, micro_batch_metrics)
            
            # 执行优化器步骤（梯度裁剪+参数更新）
            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
    
    # 清零梯度
    self.actor_optimizer.zero_grad()
    return metrics
```

## 7) 权重同步流程序列

__从训练完成到下一轮rollout的权重同步流程__:

### 调用序列:

1. __训练完成__: `verl/trainer/ppo/ray_trainer.py:RayPPOTrainer.fit()`

   - 调用: `actor_output = self.actor_rollout_wg.update_actor(batch)`

2. __下一轮生成开始__: 同一文件

   - 调用: `gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)`

3. __Worker端切换到rollout模式__: `verl/workers/fsdp_workers.py:ActorRolloutRefWorker.generate_sequences()`

   ```python
   loop.run_until_complete(self.rollout_mode())
   ```

4. __rollout_mode()执行权重同步__: `verl/workers/fsdp_workers.py:ActorRolloutRefWorker.rollout_mode()`

   __详细步骤__:

   a. __加载FSDP模型到GPU__ (如果启用了offload):

   ```python
   if self._is_offload_param:
       load_fsdp_model_to_gpu(self.actor_module_fsdp)
   ```

   b. __收集参数__ (LoRA或完整参数):

   ```python
   if hasattr(peft_model, "peft_config"):  # LoRA模式
       params = collect_lora_params(
           module=self.actor_module_fsdp,
           layered_summon=self.config.rollout.get("layered_summon", False),
           base_sync_done=self.base_sync_done,
       )
   else:  # 完整模型
       params = self.actor_module_fsdp.state_dict()
   ```

   c. __转换权重键名__:

   ```python
   params = convert_weight_keys(params, self.actor_module_fsdp)
   ```

   d. __卸载FSDP模型__ (如果启用了offload):

   ```python
   if self._is_offload_param:
       offload_fsdp_model_to_cpu(self.actor_module_fsdp)
   ```

   e. __唤醒vLLM引擎__ (如果启用了free_cache_engine):

   ```python
   await self.rollout.resume(tags=["weights"])
   ```

   f. __更新vLLM权重__:

   ```python
   await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
   ```

   g. __唤醒KV cache__:

   ```python
   await self.rollout.resume(tags=["kv_cache"])
   ```

5. __vLLM端更新权重__: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:vLLMRollout.update_weights()`

   ```python
   model.load_weights(weights)  # 调用vLLM的load_weights方法
   ```

### 关键文件和函数:

- __`verl/workers/fsdp_workers.py`__:

  - `ActorRolloutRefWorker.rollout_mode()` - 主要同步逻辑
  - `ActorRolloutRefWorker.trainer_mode()` - 切换回训练模式

- __`verl/utils/fsdp_utils.py`__:

  - `collect_lora_params()` - 收集LoRA参数
  - `load_fsdp_model_to_gpu()` - 加载模型到GPU
  - `offload_fsdp_model_to_cpu()` - 卸载模型到CPU

- __`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`__:

  - `vLLMRollout.update_weights()` - 更新vLLM权重
  - `vLLMRollout.resume()` - 唤醒vLLM引擎
  - `vLLMRollout.release()` - 释放vLLM资源

### 特殊处理:

- __LoRA模式__: 如果`sleep_level=2`，需要分别同步base模型权重和LoRA权重
- __DTensor处理__: FSDP2模式下需要将DTensor转换为完整tensor
- __异步执行__: 使用`asyncio`实现异步权重同步，避免阻塞

## 1. 训练和Rollout并行方式不同时的权重准备工作

### 核心挑战

当训练（FSDP/FSDP2）和rollout（vLLM的TP/PP）采用不同的并行策略和rank数时，需要进行权重的重分布和格式转换。

### 权重准备流程

#### 1.1 FSDP/FSDP2权重收集

__源文件__: `verl/utils/fsdp_utils.py`

__关键函数__:

```python
# FSDP1模式
def get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
    # 使用FullStateDictConfig收集完整权重
    # 位置: verl/utils/fsdp_utils.py:~line 300

# FSDP2模式  
def fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload)
    # 使用StateDictOptions进行权重分发
    # 位置: verl/utils/fsdp_utils.py:~line 350
```

__准备步骤__:

a. __加载FSDP模型到GPU__ (如果启用了offload):

```python
# verl/utils/fsdp_utils.py:load_fsdp_model_to_gpu()
if self._is_offload_param:
    load_fsdp_model_to_gpu(self.actor_module_fsdp)
```

b. __收集分片参数__:

- __FSDP1__: 使用`FSDP.state_dict()`配合`FullStateDictConfig`
- __FSDP2__: 使用`get_model_state_dict()`配合`StateDictOptions`

```python
# verl/workers/fsdp_workers.py:rollout_mode()
if hasattr(peft_model, "peft_config"):  # LoRA模式
    params = collect_lora_params(
        module=self.actor_module_fsdp,
        layered_summon=self.config.rollout.get("layered_summon", False),
        base_sync_done=self.base_sync_done,
    )
else:  # 完整模型
    params = self.actor_module_fsdp.state_dict()
```

c. __DTensor转换为本地Tensor__:

```python
# verl/workers/fsdp_workers.py:rollout_mode()
device = get_device_id()
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor() 
     if isinstance(param, DTensor) else param)
    for name, param in params.items()
)
```

__关键点__:

- DTensor需要调用`.full_tensor()`获取完整tensor
- 使用`non_blocking=True`实现异步传输
- 对于FSDP2，需要处理`Shard`到`Replicate`的placement转换

d. __权重键名转换__:

```python
# verl/utils/model.py:convert_weight_keys()
params = convert_weight_keys(params, self.actor_module_fsdp)
```

#### 1.2 LoRA特殊处理

__源文件__: `verl/utils/fsdp_utils.py`

__关键函数__:

```python
def collect_lora_params(module, layered_summon, base_sync_done)
    # 位置: verl/utils/fsdp_utils.py:~line 600

def layered_summon_lora_params(fsdp_module)
    # 逐层召唤LoRA参数，减少内存峰值
    # 位置: verl/utils/fsdp_utils.py:~line 550
```

__两种模式__:

- __layered_summon=True__: 逐层召唤，内存友好
- __layered_summon=False__: 一次性召唤所有参数

__LoRA权重键名替换__:

```python
def replace_lora_wrapper(k, peft_config)
    # 将LoRA参数名转换为base layer名称
    # 例如: "q_proj.weight" -> "q_proj.base_layer.weight"
    # 位置: verl/utils/fsdp_utils.py:~line 650
```

### 1.3 重要源文件总结

| 维度 | Colocated模式 | 分离部署模式 |
| --- | --- | --- |
| 显存管理 | 需要pause/resume | 无需pause/resume |
| 权重传输 | GPU内存直接引用（零拷贝） | 跨GPU/跨节点传输 |
| 传输延迟 | 几乎为0 | 取决于互连带宽 |
| 并发性 | 时分复用，无法并发 | 可以并发执行 |
| 资源利用率 | GPU利用率可能较低 | GPU利用率更高 |
| 实现复杂度 | 需要精细的显存管理 | 相对简单 |
| 适用场景 | GPU资源受限 | GPU资源充足 |

---

## 2. 异步权重同步的具体体现

### 2.1 异步体现在哪里？

__核心异步机制__: 使用Python `asyncio`实现的协程异步

__源文件__: `verl/workers/fsdp_workers.py`

```python
async def rollout_mode(self):
    """异步切换到rollout模式"""
    # 1. 卸载FSDP模型
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    
    # 2. 异步唤醒vLLM引擎（释放weights显存）
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])  # 异步点1
    
    # 3. 异步更新权重
    await self.rollout.update_weights(per_tensor_param, ...)  # 异步点2
    
    # 4. 异步唤醒KV cache
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["kv_cache"])  # 异步点3
```

### 2.2 并发运行的工作

__在权重同步期间，以下工作可能并发进行__:

#### a. 多Worker并发同步

```python
# verl/single_controller/ray/base.py:RayWorkerGroup.execute_all_async()
def execute_all_async(self, method_name, *args, **kwargs):
    """所有worker异步执行方法"""
    return [
        self._execute_remote_single_worker(worker, method_name, *args, **kwargs) 
        for worker in self._workers
    ]
```

__并发场景__:

- 多个GPU worker同时进行权重同步
- 每个worker独立执行`rollout_mode()`
- Ray框架管理异步任务调度

#### b. vLLM内部异步操作

__源文件__: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

```python
async def resume(self, tags: list[str]):
    """异步唤醒vLLM引擎"""
    if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
        self.inference_engine.wake_up(tags=tags)
    else:
        self.inference_engine.wake_up()
```

__vLLM异步操作__:

- KV cache预分配
- CUDA kernel预热
- 模型层加载到GPU

#### c. CPU-GPU异步传输

```python
# verl/workers/fsdp_workers.py:rollout_mode()
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor() 
     if isinstance(param, DTensor) else param)
    for name, param in params.items()
)
```

__关键__: `non_blocking=True`实现CPU到GPU的异步拷贝

### 2.3 异步的最终同步点

__同步点1__: `await self.rollout.update_weights()`完成

- 所有权重传输完成
- vLLM模型加载完成

__同步点2__: `generate_sequences()`调用前

```python
# verl/workers/fsdp_workers.py:generate_sequences()
loop.run_until_complete(self.rollout_mode())  # 等待异步完成
output = self.rollout.generate_sequences(prompts=prompts)
```

__同步点3__: Ray的`ray.get()`

```python
# verl/trainer/ppo/ray_trainer.py
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
# ray.get()隐式调用，等待所有worker完成
```

---

## 3. 权重同步触发时机与粒度

### 3.1 触发时机

__触发点__: __训练完成后，下一轮rollout开始前__

__不是per-layer方式，而是全部layer完成后触发__

__证据__:

```python
# verl/trainer/ppo/ray_trainer.py:RayPPOTrainer.fit()
with marked_timer("update_actor", timing_raw, color="red"):
    actor_output = self.actor_rollout_wg.update_actor(batch)  # 训练完成

# 下一轮迭代
with marked_timer("gen", timing_raw, color="red"):
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  # 触发同步
```

### 3.2 为什么不是per-layer？

__原因1__: FSDP的梯度计算是全局同步的

```python
# verl/workers/actor/dp_actor.py:update_policy()
for _ in range(self.config.ppo_epochs):
    for mini_batch in mini_batches:
        self.actor_optimizer.zero_grad()
        for micro_batch in micro_batches:
            loss.backward()  # 所有层梯度累积
        grad_norm = self._optimizer_step()  # 全局优化器步骤
```

__原因2__: vLLM需要完整模型才能推理

- vLLM的`load_weights()`接口要求完整权重字典
- 不支持逐层加载

__原因3__: 内存管理策略

```python
# verl/workers/fsdp_workers.py:rollout_mode()
# 先卸载整个FSDP模型
if self._is_offload_param:
    offload_fsdp_model_to_cpu(self.actor_module_fsdp)

# 再加载到vLLM
await self.rollout.update_weights(per_tensor_param, ...)
```

### 3.3 LoRA的layered_summon

__唯一的per-layer优化__: LoRA的`layered_summon`模式

__源文件__: `verl/utils/fsdp_utils.py:layered_summon_lora_params()`

```python
def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    """逐层召唤LoRA参数，减少内存峰值"""
    lora_params = OrderedDict()
    prefix_list = [
        "base_model.model.model.layers.",  # 逐层处理
        ...
    ]
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    # 只召唤当前层
                    sub_lora_params = get_peft_model_state_dict(...)
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()  # 立即释放
    return lora_params
```

__优势__:

- 逐层召唤，立即释放
- 降低内存峰值
- 仅适用于LoRA，不适用于完整模型

---

## 4. 权重传输粒度与方式分析

### 4.1 传输粒度

__当前实现__: __Tensor级别的传输__

__证据__:

```python
# verl/workers/fsdp_workers.py:rollout_mode()
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor() 
     if isinstance(param, DTensor) else param)
    for name, param in params.items()  # 逐个tensor迭代
)

await self.rollout.update_weights(per_tensor_param, ...)
```

__vLLM端接收__:

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:update_weights()
async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
    model.load_weights(weights)  # weights是generator，逐个tensor加载
```

### 4.2 传输方式分析

#### 场景1: Colocated部署（训练和rollout在同一GPU）

__传输方式__: __GPU内存直接引用，无实际传输__

```python
# verl/workers/fsdp_workers.py:rollout_mode()
# 权重已在GPU上
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor())
    for name, param in params.items()
)
# vLLM直接使用GPU上的tensor
```

__特点__:

- 零拷贝（zero-copy）
- 仅涉及指针传递
- 最快的方式

#### 场景2: 分离部署（训练和rollout在不同GPU，同一节点）

__传输方式__: __GPU-to-GPU直接传输（P2P）__

```python
# PyTorch自动使用NCCL进行GPU间传输
param.to(target_device, non_blocking=True)
```

__如果支持GPUDirect P2P__:

- 使用PCIe或NVLink直接传输
- 不经过CPU内存
- 高带宽，低延迟

#### 场景3: 多机部署

__传输方式__: __通过网络（NCCL over InfiniBand/RoCE）__

__当前实现不支持GPUDirect RDMA的直接利用__，原因：

1. __Ray的序列化机制__:

```python
# verl/single_controller/ray/base.py
def execute_all_async(self, method_name, *args, **kwargs):
    return [
        self._execute_remote_single_worker(worker, method_name, *args, **kwargs) 
        for worker in self._workers
    ]
```

Ray会序列化参数，经过：

- GPU → CPU (源节点)
- CPU → 网络 → CPU (目标节点)
- CPU → GPU (目标节点)

2. __Generator传输__:

```python
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor())
    for name, param in params.items()
)
```

Generator在Ray中会被序列化为list，失去流式传输优势

### 4.3 是否写入磁盘？

__不写入磁盘__（正常权重同步流程）

__例外情况__:

- Checkpoint保存: 写入磁盘
- HDFS同步: 写入分布式文件系统

### 4.4 传输流程序列图

```javascript
训练完成后的权重同步流程（Colocated模式）:

┌─────────────┐
│ Ray Trainer │
│  (Driver)   │
└──────┬──────┘
       │ 1. actor_rollout_wg.generate_sequences()
       ↓
┌──────────────────────┐
│ RayWorkerGroup       │
│ execute_all_async()  │
└──────┬───────────────┘
       │ 2. Ray RPC调用所有workers
       ↓
┌──────────────────────────────┐
│ ActorRolloutRefWorker        │
│ (每个GPU worker)             │
└──────┬───────────────────────┘
       │ 3. loop.run_until_complete(self.rollout_mode())
       ↓
┌──────────────────────────────┐
│ rollout_mode() [async]       │
├──────────────────────────────┤
│ 4. offload_fsdp_model_to_cpu │ (如果启用offload)
│ 5. collect_lora_params()     │ (收集权重)
│ 6. convert_weight_keys()     │ (转换键名)
│ 7. DTensor.full_tensor()     │ (DTensor→Tensor)
│ 8. offload_fsdp_model_to_cpu │ (卸载FSDP)
│ 9. await rollout.resume()    │ (唤醒vLLM)
│ 10. await update_weights()   │ (更新权重)
└──────┬───────────────────────┘
       │ 11. Generator传递权重
       ↓
┌──────────────────────────────┐
│ vLLMRollout.update_weights() │
├──────────────────────────────┤
│ 12. model.load_weights()     │ (vLLM加载)
└──────────────────────────────┘

关键模块和函数:
- verl/trainer/ppo/ray_trainer.py: RayPPOTrainer.fit()
- verl/single_controller/ray/base.py: RayWorkerGroup.execute_all_async()
- verl/workers/fsdp_workers.py: ActorRolloutRefWorker.rollout_mode()
- verl/utils/fsdp_utils.py: collect_lora_params(), offload_fsdp_model_to_cpu()
- verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py: vLLMRollout.update_weights()
```

---

## 5. Colocated vs 分离部署的实现差异

### 5.1 Colocated模式（时分复用GPU）

__配置示例__:

```python
# 训练和rollout使用相同的GPU资源池
resource_pool_spec = {
    "global_pool": [8, 8, 8, 8],  # 4节点，每节点8卡
}
mapping = {
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
}
```

__实现特点__:

#### a. 内存管理

```python
# verl/workers/fsdp_workers.py:rollout_mode()
async def rollout_mode(self):
    # 1. 卸载FSDP模型到CPU
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    
    # 2. 释放vLLM的weights显存
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])
    
    # 3. 更新权重（GPU内存直接引用）
    await self.rollout.update_weights(per_tensor_param, ...)
    
    # 4. 恢复KV cache
    await self.rollout.resume(tags=["kv_cache"])
```

__关键__: `free_cache_engine=True`时，vLLM会释放weights和KV cache

#### b. 上下文切换

```python
# verl/workers/fsdp_workers.py
async def trainer_mode(self):
    """切换回训练模式"""
    # 1. 释放vLLM资源
    if self.config.rollout.free_cache_engine:
        await self.rollout.release()
    
    # 2. 恢复FSDP模型
    self.actor_module_fsdp.train()
    
    # 3. 恢复随机状态
    self.gen_random_states = get_torch_device().get_rng_state()
    get_torch_device().set_rng_state(self.torch_random_states)
```

__显存占用模式__:

```javascript
训练阶段: [FSDP模型] [FSDP优化器] [梯度] [激活]
          ↓ offload
Rollout阶段: [vLLM模型] [KV Cache]
          ↓ release
训练阶段: [FSDP模型] [FSDP优化器] ...
```

### 5.2 分离部署模式（不同GPU）

__配置示例__:

```python
# 训练和rollout使用不同的GPU资源池
resource_pool_spec = {
    "train_pool": [8, 8],      # 2节点用于训练
    "rollout_pool": [8, 8, 8], # 3节点用于rollout
}
mapping = {
    Role.ActorRollout: "rollout_pool",
    Role.Critic: "train_pool",
}
```

__实现特点__:

#### a. 无需pause/resume

```python
# 分离部署时，free_cache_engine通常设为False
config.rollout.free_cache_engine = False

async def rollout_mode(self):
    # 不需要释放显存
    # if self.config.rollout.free_cache_engine:  # False
    #     await self.rollout.resume(tags=["weights"])
    
    # 直接更新权重
    await self.rollout.update_weights(per_tensor_param, ...)
```

#### b. 权重传输开销

```python
# 需要跨GPU/跨节点传输
per_tensor_param = (
    (name, param.to(target_device, non_blocking=True).full_tensor())
    for name, param in params.items()
)
```

__传输路径__:

- 同节点不同GPU: PCIe/NVLink
- 跨节点: 网络（InfiniBand/RoCE）

#### c. 并发执行

```python
# 训练和rollout可以并发
# 训练worker继续训练下一个batch
# Rollout worker同时进行推理
```

### 5.3 关键差异总结

| 维度 | Colocated模式 | 分离部署模式 |
| --- | --- | --- |
| 显存管理 | 需要pause/resume | 无需pause/resume |
| 权重传输 | GPU内存直接引用（零拷贝） | 跨GPU/跨节点传输 |
| 传输延迟 | 几乎为0 | 取决于互连带宽 |
| 并发性 | 时分复用，无法并发 | 可以并发执行 |
| 资源利用率 | GPU利用率可能较低 | GPU利用率更高 |
| 实现复杂度 | 需要精细的显存管理 | 相对简单 |
| 适用场景 | GPU资源受限 | GPU资源充足 |

### 5.4 额外的实现差异

#### Colocated模式特有:

1. __随机状态管理__:

```python
# verl/workers/fsdp_workers.py:rollout_mode()
self.torch_random_states = get_torch_device().get_rng_state()
get_torch_device().set_rng_state(self.gen_random_states)
```

2. __优化器状态offload__:

```python
# verl/workers/fsdp_workers.py:update_actor()
if self._is_offload_optimizer:
    load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
# ... 训练 ...
if self._is_offload_optimizer:
    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
```

3. __expandable_segments控制__:

```python
# verl/workers/fsdp_workers.py
async def rollout_mode(self):
    set_expandable_segments(False)  # 禁用PyTorch的内存扩展

async def trainer_mode(self):
    set_expandable_segments(True)   # 启用PyTorch的内存扩展
```

#### 分离部署模式特有:

1. __持久化vLLM引擎__: vLLM引擎始终保持活跃状态

2. __无状态切换__: 不需要保存/恢复随机状态

3. __更简单的内存管理__: 各自独立管理显存

---

## 总结

当前VERL的权重同步实现：

1. __不是per-layer方式__，而是全部layer训练完成后一次性同步
2. __传输粒度是tensor级别__，通过generator逐个传递
3. __异步体现在asyncio协程__，但Ray的序列化限制了真正的流式传输
4. __Colocated模式需要精细的显存管理__（pause/resume），分离模式更简单但需要跨设备传输
5. __当前实现未充分利用GPUDirect RDMA__，多机场景下存在优化空间

__潜在优化方向__:

- 实现真正的流式权重传输（避免Ray序列化）
- 利用GPUDirect RDMA减少多机传输延迟
- 探索per-layer权重同步（需要vLLM支持）
- 优化LoRA的layered_summon机制

