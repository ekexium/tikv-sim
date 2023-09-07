class Server:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.task_queue = []


class Client:
    def __int__(self, num_workers):
        self.num_workers = num_workers
