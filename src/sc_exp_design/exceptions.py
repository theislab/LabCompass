def ConfigurationError(Exception):
    base_message: str = "CONFIGURATION ERROR"
    def __init__(
        self,
        msg: str
    ) -> None:
        """"""
        msg = f"{self.base_message}: {msg}"
        super().__init__(msg)
