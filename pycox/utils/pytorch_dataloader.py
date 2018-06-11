'''
Small changes to the pytorch dataloader. 
Now works with with batches (slicing rather than loop over indexes), 
and can potentially stop workers from shutting down at each batch.
'''
import random
import torch
# import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader, _DataLoaderIter, ExceptionWrapper, \
    _set_SIGCHLD_handler, _worker_manager_loop, pin_memory_batch
import torch.multiprocessing as multiprocessing
from torch._C import _set_worker_signal_handlers, _update_worker_pids
from torch.utils.data.sampler import RandomSampler 
import sys
import threading

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    if init_fn is not None:
        init_fn(worker_id)

    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            # samples = collate_fn([dataset[i] for i in batch_indices])
            samples = collate_fn(dataset[batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
            del samples


class DataLoaderIterSlice(_DataLoaderIter):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [multiprocessing.SimpleQueue() for _ in range(self.num_workers)]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queues[i],
                          self.worker_result_queue, self.collate_fn, base_seed + i,
                          self.worker_init_fn, i))
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            # batch = self.collate_fn([self.dataset[i] for i in indices])
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
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility



class DataLoaderSlice(DataLoader):
    '''Like DataLoader but works on batches instead of iterating
    through the batch.
    '''
    def __init__(self, *args, **kwargs):
        if not ('collate_fn' in kwargs):
            kwargs['collate_fn'] = DataLoaderSlice._identity

        super().__init__(*args, **kwargs)

    def __iter__(self):
        return DataLoaderIterSlice(self)

    @staticmethod
    def _identity(x):
        '''Function returning x'''
        return x


class RandomSamplerContinuous(RandomSampler):
    """Samples elements randomly, without replacement, and continues for ever.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __iter__(self):
        while True:
            for i in iter(torch.randperm(len(self.data_source)).long()):
                yield i











###############################################################################################################
# import torch
# # import torch.utils.data as data
# from torch.utils.data.dataloader import DataLoader, _DataLoaderIter, ExceptionWrapper, \
#     _set_SIGCHLD_handler, _worker_manager_loop, pin_memory_batch
# import torch.multiprocessing as multiprocessing
# from torch._C import _set_worker_signal_handlers, _update_worker_pids
# from torch.utils.data.sampler import RandomSampler 
# import sys
# import threading

# if sys.version_info[0] == 2:
#     import Queue as queue
# else:
#     import queue


# # We rewrite worker loop so it use slicing rather than a for loop.
# def _worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id):
#     global _use_shared_memory
#     _use_shared_memory = True

#     # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
#     # module's handlers are executed after Python returns from C low-level
#     # handlers, likely when the same fatal signal happened again already.
#     # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
#     _set_worker_signal_handlers()

#     torch.set_num_threads(1)
#     torch.manual_seed(seed)

#     if init_fn is not None:
#         init_fn(worker_id)

#     while True:
#         r = index_queue.get()
#         if r is None:
#             break
#         idx, batch_indices = r
#         try:
#             # samples = collate_fn([dataset[i] for i in batch_indices])
#             samples = collate_fn(dataset[batch_indices])
#         except Exception:
#             data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
#         else:
#             data_queue.put((idx, samples))


# class DataLoaderIterSlice(_DataLoaderIter):
#     "Iterates once over the DataLoader's dataset, as specified by the sampler"

#     def __init__(self, loader):
#         # We need init because it call _worker_loop.
#         # No changes to the code.
#         self.dataset = loader.dataset
#         self.collate_fn = loader.collate_fn
#         self.batch_sampler = loader.batch_sampler
#         self.num_workers = loader.num_workers
#         self.pin_memory = loader.pin_memory and torch.cuda.is_available()
#         self.timeout = loader.timeout
#         self.done_event = threading.Event()

#         self.sample_iter = iter(self.batch_sampler)

#         if self.num_workers > 0:
#             self.worker_init_fn = loader.worker_init_fn
#             self.index_queue = multiprocessing.SimpleQueue()
#             self.worker_result_queue = multiprocessing.SimpleQueue()
#             self.batches_outstanding = 0
#             self.worker_pids_set = False
#             self.shutdown = False
#             self.send_idx = 0
#             self.rcvd_idx = 0
#             self.reorder_dict = {}

#             base_seed = torch.LongTensor(1).random_()[0]
#             self.workers = [
#                 multiprocessing.Process(
#                     target=_worker_loop,
#                     args=(self.dataset, self.index_queue, self.worker_result_queue, self.collate_fn,
#                           base_seed + i, self.worker_init_fn, i))
#                 for i in range(self.num_workers)]

#             if self.pin_memory or self.timeout > 0:
#                 self.data_queue = queue.Queue()
#                 if self.pin_memory:
#                     maybe_device_id = torch.cuda.current_device()
#                 else:
#                     # do not initialize cuda context if not necessary
#                     maybe_device_id = None
#                 self.worker_manager_thread = threading.Thread(
#                     target=_worker_manager_loop,
#                     args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
#                           maybe_device_id))
#                 self.worker_manager_thread.daemon = True
#                 self.worker_manager_thread.start()
#             else:
#                 self.data_queue = self.worker_result_queue

#             for w in self.workers:
#                 w.daemon = True  # ensure that the worker exits on process exit
#                 w.start()

#             _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
#             _set_SIGCHLD_handler()
#             self.worker_pids_set = True

#             # prime the prefetch loop
#             for _ in range(2 * self.num_workers):
#                 self._put_indices()

#     def __next__(self):
#         if self.num_workers == 0:  # same-process loading
#             indices = next(self.sample_iter)  # may raise StopIteration
#             # batch = self.collate_fn([self.dataset[i] for i in indices])
#             batch = self.collate_fn(self.dataset[indices])
#             if self.pin_memory:
#                 batch = pin_memory_batch(batch)
#             return batch

#         # check if the next sample has already been generated
#         if self.rcvd_idx in self.reorder_dict:
#             batch = self.reorder_dict.pop(self.rcvd_idx)
#             return self._process_next_batch(batch)

#         if self.batches_outstanding == 0:
#             self._shutdown_workers()
#             raise StopIteration

#         while True:
#             assert (not self.shutdown and self.batches_outstanding > 0)
#             idx, batch = self._get_batch()
#             self.batches_outstanding -= 1
#             if idx != self.rcvd_idx:
#                 # store out-of-order samples
#                 self.reorder_dict[idx] = batch
#                 continue
#             return self._process_next_batch(batch)

#     next = __next__ #python2 compatability

