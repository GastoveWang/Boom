# 檔案功能：作者歷史 JAX 權重檢查腳本。
# 所屬分工：legacy：上游歷史參考。
# 輸入／輸出：保留原作者資料／權重介面；部分路徑與額外依賴尚未本機化。
# 執行方式：研究參考；正式實驗請執行 ASTT 根目錄 train.py／test.py。
# 設定與說明：歷史參數見 legacy/config.py；適配限制見 legacy/README.md。

# 允許歷史腳本從父目錄取得 model.py 相容入口；同目錄工具仍優先。
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))

import jax
import numpy as onp
import flax
from vit_jax import models
from vit_jax import checkpoint
from vit_jax import log
logger = log.setup_logger('./logs')

import torch
from src.data_loaders import *
from src.model import VisionTransformer
from src.checkpoint import load_jax, convert_jax_pytorch


def main():
    image_size = 384

    # jax model
    jax_model = models.KNOWN_MODELS['ViT-B_16'].partial(num_classes=1000, representation_size=None)
    _, params = jax_model.init_by_shape(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        [((4, image_size, image_size, 3), 'float32')])
    params = checkpoint.load_pretrained(
        pretrained_path='/home/hchen/Projects/vision_transformer/weights/jax/imagenet21k+imagenet2012_ViT-B_16.npz',
        init_params=params,
        model_config=models.CONFIGS['ViT-B_16'],
        logger=logger)
    params_repl = flax.jax_utils.replicate(params)
    # Then map the call to our model's forward pass onto all available devices.
    vit_apply_repl = jax.pmap(jax_model.call)

    # torch_model
    keys, values = load_jax('/home/hchen/Projects/vision_transformer/weights/jax/imagenet21k+imagenet2012_ViT-B_16.npz')
    state_dict = convert_jax_pytorch(keys, values)

    torch_model = VisionTransformer(
                 image_size=(image_size, image_size),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    data_loader = ImageNetDataLoader(
        data_dir='/home/hchen/Projects/vat_contrast/data/ImageNet',
        split='val',
        image_size=image_size,
        batch_size=16,
        num_workers=0)

    for batch_idx, (data, target) in enumerate(data_loader):

        # jax prediction
        target_numpy = target.cpu().numpy()
        data_numpy = data.cpu().numpy().transpose(0, 2, 3, 1).reshape(1, -1, image_size, image_size, 3)
        jax_predicted_logits = vit_apply_repl(params_repl, data_numpy)._value[0]
        jax_predicted = onp.argmax(jax_predicted_logits, axis=-1)

        # torch prediction
        with torch.no_grad():
            torch_predicted = torch_model(data)
        torch_predicted_logits = torch_predicted.cpu().numpy()
        torch_predicted = onp.argmax(torch_predicted_logits, axis=-1)

        # check difference
        # diff = onp.abs(jax_predicted_logits - torch_predicted_logits)
        # assert onp.allclose(jax_predicted_logits, torch_predicted_logits, rtol=1e-1, atol=1e-1), "diff {}, max {}, sum {}".format(diff, onp.max(diff), onp.sum(diff))

        diff = onp.abs(jax_predicted - torch_predicted)
        print(diff)
        # assert onp.allclose(jax_predicted, torch_predicted, rtol=1e-5, atol=1e-8), "diff {}, max {}, sum {}".format(diff, onp.max(diff), onp.sum(diff))


if __name__ == '__main__':
    main()
