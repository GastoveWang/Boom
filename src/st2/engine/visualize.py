"""Export actual ST2 forward computation graphs with torchview and Graphviz."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import uuid

from st2.common.utils import write_json

ROOT = Path(__file__).resolve().parents[3]
PARTS = ('model', 'predictor', 'registration')


def pipeline_graph(part, direction):
    """Explicit schematic for the non-differentiable OpenCV boundary, not a trace."""
    from graphviz import Digraph
    graph = Digraph(f'ST2_{part}')
    graph.attr(rankdir=direction)
    graph.attr('node', shape='box')
    graph.node('sequence', 'Last historical RGB + current')
    graph.node('matcher', 'Frozen outdoor LoFTR\nCNN + Transformer correspondences')
    graph.node('ransac', 'RANSAC similarity matrix\nquality checks')
    graph.node('warp', 'Inverse pixel warp + valid overlap\nmasked RGB MSE + quality')
    graph.edges([('sequence', 'matcher'), ('matcher', 'ransac'), ('ransac', 'warp')])
    if part == 'model':
        graph.node('history', 'Past frames only')
        graph.node('predictor', 'ST2Predictor\ncausal normal-frame prediction')
        graph.node('raw', 'Raw prediction MSE vs current')
        graph.node('output', 'Separate diagnostics:\nraw prediction MSE / compensated actual-frame MSE\nNo Gate, ROI or learned weighting')
        graph.edges([('history', 'predictor'), ('predictor', 'raw'), ('sequence', 'raw'),
                     ('raw', 'output'), ('warp', 'output')])
    return graph


def find_graphviz():
    """Find dot without modifying persistent user or system environment settings."""
    found = shutil.which('dot')
    if found:
        return Path(found)
    candidates = [Path(sys.executable).parent/'Library/bin/dot.exe',
                  Path(sys.prefix)/'Library/bin/dot.exe',
                  Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Graphviz/bin/dot.exe',
                  Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Graphviz/bin/dot.exe']
    return next((path for path in candidates if path.is_file()), None)


@contextmanager
def graphviz_path(executable):
    previous = os.environ.get('PATH')
    try:
        if executable is not None:
            os.environ['PATH'] = str(executable.parent) + os.pathsep + (previous or '')
        yield
    finally:
        if previous is None:
            os.environ.pop('PATH', None)
        else:
            os.environ['PATH'] = previous


def input_shapes(config, height, width, batch_size, part):
    history = (batch_size, config.history_frames, 3, height, width)
    current = (batch_size, 3, height, width)
    return {'model': [history, current], 'predictor': history,
            'registration': (batch_size, 2, 3, height, width)}[part]


def export_graphs(output=ROOT/'output/st2/visualization', checkpoint=None, *,
                  height=None, width=None, batch_size=1, depth=2, device='meta',
                  part='all', formats=('svg',), direction='TB', expand_nested=True):
    """Create a unique run; checkpoint/input files and model math are never changed.

    No existing live training model is accepted: tracing uses a fresh model per
    part. RNG and the MHA fastpath setting are restored after each trace.
    """
    import torch
    try:
        from torchview import draw_graph
    except ImportError as exc:
        raise ImportError('Install visualization dependencies: python -m pip install -e ".[st2-visualization]"') from exc
    from st2.models.network import ST2Config, ST2Model
    formats = tuple(dict.fromkeys(formats))
    if (part not in (*PARTS, 'all') or not formats or any(f not in ('svg', 'dot') for f in formats)
            or device not in ('cpu', 'meta') or direction not in ('TB', 'LR')):
        raise ValueError('Invalid graph part, format, device or direction')
    if type(depth) is not int or depth < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Depth and batch size must be positive integers')
    state = None
    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve(strict=True)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if state.get('format') not in ('st2.checkpoint.v1', 'boom.roi_heatmap.checkpoint.v1'):
            raise ValueError('Expected an ST2 checkpoint')
        config = ST2Config.from_checkpoint(state['model_config'])
    else:
        config = ST2Config()
    height = height if height is not None else (state['args']['height'] if state else 576)
    width = width if width is not None else (state['args']['width'] if state else 768)
    if type(height) is not int or type(width) is not int:
        raise ValueError('Image height and width must be integers')
    config.validate_image_size(height, width)
    executable = find_graphviz()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    run = Path(output).resolve()/f'st2_torchview_{stamp}_{uuid.uuid4().hex[:8]}'
    run.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'tool': 'torchview',
               'torchview_version': importlib.metadata.version('torchview'),
               'torch_version': str(torch.__version__), 'model_name': config.model_name,
               'model_version': config.model_version, 'model_config': asdict(config),
               'checkpoint': str(checkpoint_path) if checkpoint_path else None,
               'weights': 'checkpoint' if state else 'random initialization; architecture only',
               'height': height, 'width': width, 'batch_size': batch_size,
               'device': device, 'depth': depth, 'direction': direction,
               'expand_nested': expand_nested,
               'graphviz_dot': str(executable) if executable else None,
               'graph_kind': 'model/registration: explicit schematic; predictor: torchview forward trace',
               'parts': {}}
    write_json(run/'summary.json', summary)
    for name in PARTS if part == 'all' else (part,):
        record = {'status': 'running', 'input_shapes': input_shapes(config, height, width, batch_size, name),
                  'files': [], 'errors': []}
        summary['parts'][name] = record
        try:
            # torchview records unfused attention operations. Disable only within
            # this export and restore the original setting even on an exception.
            with torch.random.fork_rng(devices=[]):
                model = ST2Model(config).eval()
                if state:
                    model.load_state_dict(state['model'], strict=True)
                target = model if name == 'model' else getattr(model, name)
                record['parameters'] = sum(p.numel() for p in target.parameters())
                record['graph_kind'] = ('explicit pipeline schematic; OpenCV RANSAC cannot be tensor-traced'
                                        if name in ('model', 'registration') else 'actual torchview forward trace')
                previous_fastpath = torch.backends.mha.get_fastpath_enabled()
                try:
                    torch.backends.mha.set_fastpath_enabled(False)
                    if name in ('model', 'registration'):
                        visual = pipeline_graph(name, direction)
                    else:
                        graph = draw_graph(target, input_size=record['input_shapes'], graph_name=f'ST2_{name}',
                                           device=device, depth=depth, mode='eval', expand_nested=expand_nested,
                                           graph_dir=direction, show_shapes=True, save_graph=False,
                                           hide_inner_tensors=True, hide_module_functions=True)
                        visual = graph.visual_graph
                finally:
                    torch.backends.mha.set_fastpath_enabled(previous_fastpath)
            # Use Graphviz's built-in font metrics rather than torchview's Linux
            # font default, which can clip labels on Windows installations.
            visual.node_attr.update(fontname='Times-Roman')
            visual.edge_attr.update(fontname='Times-Roman')
            kind = 'pipeline schematic' if name in ('model', 'registration') else 'torchview forward trace'
            visual.graph_attr.update(label=f'ST2 v3 box-free | {name} | {kind} | {height}x{width} | {record["parameters"]:,} parameters',
                                     labelloc='t', fontname='Times-Roman', fontsize='18')
            dot_name = name+'.dot'
            (run/dot_name).write_text(visual.source, encoding='utf-8')
            record['files'].append(dot_name)
            for fmt in formats:
                if fmt == 'dot':
                    continue
                if executable is None:
                    record['errors'].append(f'Graphviz dot is missing; {fmt} not rendered. DOT source is available.')
                    continue
                try:
                    with graphviz_path(executable):
                        rendered = visual.pipe(format=fmt, quiet=True)
                    filename = name+'.'+fmt
                    (run/filename).write_bytes(rendered)
                    record['files'].append(filename)
                except (OSError, RuntimeError) as exc:
                    record['errors'].append(f'{fmt}: {exc}')
            record['status'] = 'partial' if record['errors'] else 'complete'
        except Exception as exc:
            record['status'] = 'failed'
            record['errors'].append(f'{type(exc).__name__}: {exc}')
        write_json(run/'summary.json', summary)
    statuses = [record['status'] for record in summary['parts'].values()]
    summary['status'] = ('failed' if 'failed' in statuses else 'partial' if 'partial' in statuses else 'complete')
    write_json(run/'summary.json', summary)
    return run, summary
