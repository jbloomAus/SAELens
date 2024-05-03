import pickle


class BackwardsCompatibleUnpickler(pickle.Unpickler):
    """
    An Unpickler that can load files saved before the "sae_lens" package namechange
    """

    def find_class(self, module: str, name: str):
        module = module.replace("sae_training", "sae_lens.training")
        return super().find_class(module, name)


class BackwardsCompatiblePickleClass:
    Unpickler = BackwardsCompatibleUnpickler
