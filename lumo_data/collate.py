from torch.utils.data.dataloader import default_collate


class CollateBase:

    def __init__(self, collate_fn=default_collate, *args, **kwargs) -> None:
        super().__init__()
        self._collate_fn = collate_fn
        self.initial(*args, **kwargs)

    def initial(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.wraped_collate(*args, **kwargs)

    def before_collate(self, sample_list):
        return sample_list

    def collate(self, sample_list):
        return self._collate_fn(sample_list)

    def wraped_collate(self, sample_list):
        sample_list = self.before_collate(sample_list)
        batch = self.collate(sample_list)
        batch = self.after_collate(batch)
        return batch

    def after_collate(self, batch):
        return batch
