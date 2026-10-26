# ============================================
# 실험용 확장·설정 변경: 명시적으로 실행하는 소형 Metal 검증 도구
# ============================================
"""Tiny random-weight Metal smoke test; this is not trained Qwen validation.
질문도 loss에 포함하는 MLX 설정을 점검한다. 자동 CPU 테스트에서는 실행하지 않는다.
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.trainer import Trainer, VLMTrainingArgs, prepare_model_for_training
from mac_pipeline import MODEL, REVISION, vision_dataset


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 64)
        self.up_proj = nn.Linear(64, 64)
        self.down_proj = nn.Linear(64, 64)
    def __call__(self, x):
        return x + self.down_proj(nn.silu(self.gate_proj(x))*self.up_proj(x))


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block()]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {'model_type': 'tiny'}
        self.embed = nn.Embedding(64, 64)
        self.language_model = Language()
        self.head = nn.Linear(64, 64)
    def __call__(self, ids, pixels, attention_mask, **kwargs):
        x = self.embed(ids) + pixels.mean()
        for block in self.language_model.layers:
            x = block(x)
        return self.head(x)


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    mx.random.seed(3407)
    model = Tiny()
    nn.quantize(model, bits=4, group_size=64)
    prepare_model_for_training(model, lora_rank=8, lora_alpha=8,
                               target_modules=['gate_proj', 'up_proj', 'down_proj'])
    before = {k: mx.array(v) for k,v in tree_flatten(model.trainable_parameters())}
    mx.eval(before)
    samples = [{'input_ids': mx.array([1,2,3,4,5]), 'attention_mask':mx.ones((5,),dtype=mx.int32),
                'completion_mask':mx.array([0,0,1,1,1]), 'pixel_values':mx.ones((6,3,4,4))}]
    cfg = VLMTrainingArgs(iters=2, grad_checkpoint=True, train_on_completions=False,
                          max_seq_length=64, gradient_accumulation_steps=2, val_batches=1,
                          steps_per_eval=2, steps_per_report=2, adapter_file=str(root/'adapters.safetensors'))
    trainer = Trainer(model,cfg,train_dataset=samples,eval_dataset=samples,prepared=True)
    metrics = trainer.train()
    after = dict(tree_flatten(model.trainable_parameters()))
    assert any(bool(mx.any(before[k] != after[k])) for k in before)
    assert (root/'adapter_config.json').exists()
    restored = Tiny()
    nn.quantize(restored,bits=4,group_size=64)
    prepare_model_for_training(restored, checkpoint_path=root, verbose=False)
    restored_weights = dict(tree_flatten(restored.trainable_parameters()))
    assert all(bool(mx.all(after[k] == restored_weights[k])) for k in after)
    print('PASS: Metal 4bit LoRA update, full-sequence CE, checkpointing, validation, adapter round trip')

    from huggingface_hub import snapshot_download
    from mlx_vlm.utils import load_processor
    import json
    metadata = snapshot_download(MODEL, revision=REVISION, allow_patterns=['*.json','*.txt','*.model'])
    processor = load_processor(Path(metadata))
    config = json.loads((Path(metadata)/'config.json').read_text())
    from PIL import Image
    image_path = root/'camera.jpg'
    Image.new('RGB',(320,180),(100,150,200)).save(image_path)
    row={'images':[str(image_path)]*6,'messages':[{'role':'user','content':'What is ahead?'},
        {'role':'assistant','content':'<answer>A car.</answer>'}]}
    sample=vision_dataset([row],config,processor)[0]
    assert sample['pixel_values'] is not None
    assert sample['image_grid_thw'].shape[0] == 6
    assert 'completion_mask' not in sample
    assert int(sample['attention_mask'].sum()) > 0
    print('PASS: real Qwen processor, 6 resized camera images, no completion-only mask',sample['input_ids'].shape,sample['image_grid_thw'].tolist())
