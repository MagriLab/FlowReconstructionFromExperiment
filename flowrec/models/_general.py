class BaseModel():
    def __init__(self) -> None:
        pass

    def init(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError
