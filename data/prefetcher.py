import torch


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after = None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, list):
                self.next_input = [item.cuda(device = self.device, non_blocking = True) for item in self.next_input]
                self.next_target = [item.cuda(device = self.device, non_blocking = True) for item in self.next_target]
            else:
                self.next_input = self.next_input.cuda(device = self.device, non_blocking = True)
                self.next_target = self.next_target.cuda(device = self.device, non_blocking = True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break