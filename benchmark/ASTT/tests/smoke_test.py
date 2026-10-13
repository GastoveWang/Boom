# 檔案功能：暫存資料自我檢查：模型、窗口、事件、指標、CLI 與續訓。
# 所屬分工：tests：相當於 Boom 的測試目錄。
# 輸入／輸出：系統暫存 fixtures → PASS／例外；測試結束清除 fixtures。
# 執行方式：從 Boom 根目錄執行 python benchmark/ASTT/tests/smoke_test.py。
# 設定與說明：只使用微型模型驗證流程，不代表正式煙霧準確率或機載 FPS。

"""Disposable fixtures only; never inserts generated images into project data."""
import json
from pathlib import Path
import subprocess
import sys

_source = str(Path(__file__).resolve().parents[3])
if _source not in sys.path:
    sys.path.insert(0, _source)
import tempfile
from argparse import Namespace
import cv2
import numpy as np
import torch
from benchmark.ASTT.src.dataset import VideoDataset
from benchmark.ASTT.src.candidates import EventMerger, candidate_boxes
from benchmark.ASTT.src.model import VisionTransformer, CrossAttention
from benchmark.ASTT.src.runner import auc_score, evaluate, create_output, load_checkpoint


def main():
    torch.set_num_threads(2)
    for script in ['scripts/train.py', 'scripts/test.py', 'tools/extract_frames.py']:
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/script), '--help'], check=True, stdout=subprocess.DEVNULL)
    config = dict(image_size=32, patch_size=16, emb_dim=32, mlp_dim=64,
                  num_heads=4, spatial_layers=1, temporal_layers=1, num_frames=4, dropout_rate=0.)
    model = VisionTransformer(**config).eval()
    with torch.no_grad():
        result = model(torch.randn(2, 4, 3, 32, 32))
    assert result.shape == (2, 3, 32, 32) and torch.isfinite(result).all()
    # Check attention against an independent explicit single-head equation.
    ca = CrossAttention(4, 1).eval()
    with torch.no_grad():
        for layer in [ca.align1,ca.wq,ca.wk,ca.wv,ca.align2]:
            layer.weight.copy_(torch.eye(4))
        tokens = torch.randn(2, 5, 4)
        expected = 2*tokens[:,:1] + ((tokens[:,:1] @ tokens.transpose(1,2))*.5).softmax(-1) @ tokens
        torch.testing.assert_close(ca(tokens), expected)
    assert auc_score([0,1,0,1], [0.,1.,0.,1.]) == 1.
    assert auc_score([0,1], [1.,1.]) == .5
    assert auc_score([0,0], [0.,1.]) is None
    merger = EventMerger(3,.3,1)
    for frame in range(7):
        merger.update(frame, [(1,1,9,9)], .2)
    assert len(merger.finish()) == 1 and merger.finished[0]['hits'] == 7
    merger = EventMerger(3,.3,1)
    merger.update(0,[(1,1,9,9)],.2)
    merger.update(2,[(1,1,9,9)],.2)
    assert not merger.finish()  # Nonconsecutive hits must not pass debounce.
    error = np.zeros((32,32),np.float32)
    error[5:15,5:15] = .5
    assert candidate_boxes(error,.1,25,3) == [(5,5,15,15)]
    with tempfile.TemporaryDirectory(prefix='astt_smoke_') as temporary:
        root = Path(temporary)
        for video in ['a','b']:
            folder = root/'frames'/video
            folder.mkdir(parents=True)
            for frame in range(7):
                cv2.imwrite(str(folder/f'frame_{frame}.jpg'), np.full((32,32,3), frame*20, np.uint8))
        labels = root/'labels'
        labels.mkdir()
        np.save(labels/'a.npy',np.array([0,0,0,0,0,1,1]))
        ds = VideoDataset(root/'frames',32,4,labels)
        assert len(ds) == 6
        for video,start in ds.samples:
            assert len({p.parent for p in ds.videos[video][start:start+5]}) == 1
        assert ds[0]['inputs'].shape == (4,3,32,32)
        assert ds[0]['frame_index'] == 4
        normal = VideoDataset(root/'frames',32,4,labels,normal_only=True)
        assert len(normal) == 4
        args = Namespace(output_dir=root,experiment_name='inference',batch_size=2,num_workers=0,
            min_event_frames=2,event_iou=.3,event_max_gap=1,threshold=.1,min_area=25,morph_kernel=3)
        output = create_output(args)
        metrics = evaluate(model,ds,args,torch.device('cpu'),output)
        assert metrics['evaluated_frames'] == 6 and metrics['labelled_frames'] == 3
        assert np.isnan(np.load(output/'scores'/'a.npy')[:4]).all()
        assert len(list((output/'heatmaps'/'a').glob('*.npy'))) == 3
        torch.save({'model':model.state_dict(),'model_config':config},root/'fixture.pth')
        loaded = load_checkpoint(root/'fixture.pth',torch.device('cpu'))
        model.load_state_dict(loaded['model'])
        # One real backward/update on disposable tensors, not a full training run.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        loss = torch.nn.functional.mse_loss(model(torch.randn(2,4,3,32,32)),torch.randn(2,3,32,32))
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        # Actual CLI train/val/test on temporary fixtures, tiny model only.
        for split,video in [('train','normal_train'),('val','heldout_val'),('test','heldout_test')]:
            frames_root = root/'data'/'astt'/split
            folder = (frames_root if split == 'train' else frames_root/'frames')/video
            folder.mkdir(parents=True)
            for frame in range(7):
                cv2.imwrite(str(folder/f'frame_{frame}.jpg'),np.full((32,32,3),frame*20,np.uint8))
            if split != 'train':
                masks = frames_root/'test_frame_masks'
                masks.mkdir()
                np.save(masks/f'{video}.npy',np.array([0,0,0,0,0,1,1]))
        common = ['--data-dir',str(root/'data'),'--output-dir',str(root/'output'),'--device','cpu','--batch-size','2']
        training_args = ['--epochs','2','--image-size','32','--patch-size','16','--emb-dim','32',
            '--mlp-dim','64','--num-heads','4','--spatial-layers','1','--temporal-layers','1','--dropout-rate','0']
        base = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable,str(base/'scripts'/'train.py'),*common,*training_args,
            '--experiment-name','cli_train'],check=True)
        checkpoint_path = root/'output'/'astt'/'cli_train'/'checkpoints'/'best.pth'
        assert checkpoint_path.is_file()
        subprocess.run([sys.executable,str(base/'scripts'/'test.py'),*common,'--checkpoint',str(checkpoint_path),
            '--experiment-name','cli_test'],check=True)
        assert (root/'output'/'astt'/'cli_test'/'events.csv').is_file()
        # Simulate an interrupted second epoch; resume from the real epoch-1 state.
        from benchmark.ASTT.src import runner
        original_evaluate = runner.evaluate
        calls = [0]
        def interrupt_second_validation(*a, **kw):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('simulated interruption')
            return original_evaluate(*a, **kw)
        runner.evaluate = interrupt_second_validation
        try:
            runner.train(runner.parser().parse_args([*common,*training_args,'--experiment-name','interrupted']))
        except RuntimeError as exc:
            assert str(exc) == 'simulated interruption'
        finally:
            runner.evaluate = original_evaluate
        resume_path = root/'output'/'astt'/'interrupted'/'checkpoints'/'last.pth'
        subprocess.run([sys.executable,str(base/'scripts'/'train.py'),*common,*training_args,
            '--resume',str(resume_path),'--experiment-name','resumed'],check=True)
        assert (root/'output'/'astt'/'resumed'/'checkpoints'/'best.pth').is_file()
        video_path = root/'short.mp4'
        video_writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),20.,(32,32))
        assert video_writer.isOpened()
        for frame in range(20):
            video_writer.write(np.full((32,32,3),frame*10,np.uint8))
        video_writer.release()
        subprocess.run([sys.executable,str(base/'tools'/'extract_frames.py'),str(video_path),
            '--data-dir',str(root/'extracted'),'--fps','10','--name','extracted_clip'],check=True)
        assert len(list((root/'extracted'/'astt'/'train'/'extracted_clip').glob('*.jpg'))) == 10
        (root/'frames'/'a'/'frame_3.jpg').unlink()
        assert len(VideoDataset(root/'frames',32,4)) == 3  # only video b; no bridging missing frame
        np.save(labels/'a.npy',np.zeros(2))
        try:
            VideoDataset(root/'frames',32,4,labels)
        except ValueError:
            pass
        else:
            raise AssertionError('Label length mismatch must fail')
    print(json.dumps({'status':'PASS','forward_shape':list(result.shape),
        'checks':['imports','CLI help','video boundaries','missing frames','label alignment',
                  'CSA equation','AUROC ties','event debounce','inference outputs','checkpoint','backward',
                  'train-val-test CLI','interrupted resume','MP4 extraction']}))


if __name__ == '__main__':
    main()
