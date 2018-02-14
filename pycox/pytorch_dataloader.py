'''
Small changes to the pytorch dataloader. 
Now works with with batches (slicing rather than loop over indexes), 
and can potentially stop workers from shutting down at each batch.
'''
import torch
import torch.utils.data as data
import torch.multiprocessing as multiprocessing
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.dataloader import ExceptionWrapper
import collections
import sys
import traceback
import threading
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)



def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    '''As torch.utils.data.dataloader._worker_loop but works on
    full batches instead of iterating through the batch.
    '''
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
#             samples = collate_fn([dataset[i] for i in batch_indices])
            samples = collate_fn(dataset[batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class DataLoaderIterBatch(data.dataloader.DataLoaderIter):
    '''Change DataLoaderIter (which is used by DataLoader) to make
    it process batches instead of single elements.
    '''
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()


    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
#             batch = self.collate_fn([self.dataset[i] for i in indices])
            batch = self.collate_fn(self.dataset[indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__ #python2 compatability



class DataLoaderBatch(data.DataLoader):
    '''Like DataLoader but works on batches instead of iterating
    through the batch.
    '''
    def __init__(self, *args, **kwargs):
        if not ('collate_fn' in kwargs):
            kwargs['collate_fn'] = DataLoaderBatch._identity

        super().__init__(*args, **kwargs)

    def __iter__(self):
        return DataLoaderIterBatch(self)

    @staticmethod
    def _identity(x):
        '''Function returning x'''
        return x


class RandomSamplerContinuous(data.sampler.RandomSampler):
    """Samples elements randomly, without replacement, and continues for ever.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __iter__(self):
        while True:
            for i in iter(torch.randperm(len(self.data_source)).long()):
                yield i
