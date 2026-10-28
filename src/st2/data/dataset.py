"""Box-free normal clips; existing annotations are preserved but never read as targets."""
from st2.data.sequence import ST2SequenceDataset


class ST2Dataset(ST2SequenceDataset):
    def __init__(self, root, split, image_size=(576, 768), history_frames=8):
        super().__init__(root, split, image_size, history_frames=history_frames + 1, read_targets=False)
        self.samples = [sample for sample in self.samples if sample['normal_history']]
        if not self.samples:
            raise ValueError(f'No wholly normal windows in {self.root / split}')
        # No box-derived labels, masks, positive sampling or target metrics.
        del self.positive

    def __getitem__(self, index):
        frames = super().__getitem__(index)
        return frames[:-1], frames[-1]
