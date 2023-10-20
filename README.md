# KoNEFTune
Random Noisy Embeddings with fine-tuning 방법론을 한국어 LLM에 간단히 적용할 수 있는 repo

# Introduction about NEFTune
![image](https://github.com/Marker-Inc-Korea/KoNEFTune/assets/98331298/251da313-9ff0-4e55-853c-32a247841f93)   
![image](https://github.com/Marker-Inc-Korea/KoNEFTune/assets/98331298/f26d794d-62ef-461b-ac92-4c9bee6db741)  
> More detail: [NEFTune github](https://github.com/neelsjain/NEFTune/tree/main) and [NEFTune paper](https://arxiv.org/abs/2310.05914).  

# Core Code
```python
### add noise to embeds
#print(model)
#print("############## training_step here!!")
embed_device = model.module.base_model.model.model.embed_tokens.weight.device
embeds_init = model.module.base_model.model.model.embed_tokens.forward(inputs['input_ids'].to(embed_device))

### add noise to embeds
input_mask = inputs['attention_mask'].to(embeds_init) # B x L
input_lengths = torch.sum(input_mask, 1) # B

noise_ = torch.zeros_like(embeds_init).uniform_(-1,1)
delta = noise_ * input_mask.unsqueeze(2)
dims = input_lengths * embeds_init.size(-1)
mag = 5 / torch.sqrt(dims) # args.neftune_alpha / torch.sqrt(dims) // alpha-> 5
delta = (delta * mag.view(-1, 1, 1)).detach()
inputs['inputs_embeds'] = delta + embeds_init
inputs['input_ids'] = None
### add noise to embeds
```
You can apply above code, in your custom code.  
Or you also apply [NEFTune function](https://github.com/neelsjain/NEFTune/tree/main).  
  
# Method: How to applying code
(coming soon...)  

```python
# In trainer.py, maybe line 2750.
def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """

    model.train()
    inputs = self._prepare_inputs(inputs)

    ### add noise to embeds
    #print(model)
    #print("############## training_step here!!")
    embed_device = model.module.base_model.model.model.embed_tokens.weight.device
    embeds_init = model.module.base_model.model.model.embed_tokens.forward(inputs['input_ids'].to(embed_device))

    ### add noise to embeds
    input_mask = inputs['attention_mask'].to(embeds_init) # B x L
    input_lengths = torch.sum(input_mask, 1) # B
    
    noise_ = torch.zeros_like(embeds_init).uniform_(-1,1)
    delta = noise_ * input_mask.unsqueeze(2)
    dims = input_lengths * embeds_init.size(-1)
    mag = 5 / torch.sqrt(dims) # args.neftune_alpha / torch.sqrt(dims) // alpha-> 5
    delta = (delta * mag.view(-1, 1, 1)).detach()
    inputs['inputs_embeds'] = delta + embeds_init
    inputs['input_ids'] = None
    ### add noise to embeds

    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(self.args.device)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    loss.requires_grad_(True)

    if self.do_grad_scaling:
        self.scaler.scale(loss).backward()
    elif self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        self.accelerator.backward(loss) # here

    return loss.detach() / self.args.gradient_accumulation_steps
```
You change `training_step` function like above code.  

# Consideration (Some trick option)
```python
# In trainer.py, maybe line 1598.
def _inner_training_loop(...):
  ##################################
  # line pass until maybe 1800 line.
  # This code is abstrct.
  ##################################

  # Maybe line 1844,
  for epoch in range(epochs_trained, num_train_epochs):

    # ...(some training code)...

    # Maybe line 1950,
    else:
        try:
            self.accelerator.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
        except:
            pass

    # Optimizer step
    optimizer_was_run = True
    if is_torch_tpu_available():
        if self.do_grad_scaling:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
            self.optimizer.step()
    elif self.do_grad_scaling:
        #print("here?")
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        optimizer_was_run = scale_before <= scale_after
    else:
        try:
            self.optimizer.step()
        except: 
            print("Ignoring")
            pass
        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

    # ...(Other code)...

    return TrainOutput(self.state.global_step, train_loss, metrics)
```
I add the *try~except* condition in `clip_grad_norm_` and `self.optimizer.step()`.   

```python
# In finetune.py
if NEFTune:
  print("Our transformers version is 4.34.1")
  print('transformers.trainer -> find `compute loss(maybe line 2810)` or `training_step(maybe line 2750)` function.')
  print("Default alpha value:", 5)
else:
  print("Done!!")
```
> Consider the `transformers` version.

# Model benchmark
(coming soon...)  

# References
[NEFTune github](https://github.com/neelsjain/NEFTune/tree/main)  
[KO-platypus🥮](https://github.com/Marker-Inc-Korea/KO-Platypus)  
