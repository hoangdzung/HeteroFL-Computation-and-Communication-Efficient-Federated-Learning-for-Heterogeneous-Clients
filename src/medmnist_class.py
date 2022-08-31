from medmnist.dataset import PathMNIST, DermaMNIST, OCTMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST

class WrapPathMNIST(PathMNIST):
    def __init__(self, **kwargs):
        super(WrapPathMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapPathMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapDermaMNIST(DermaMNIST):
    def __init__(self, **kwargs):
        super(WrapDermaMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapDermaMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapOCTMNIST(OCTMNIST):
    def __init__(self, **kwargs):
        super(WrapOCTMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOCTMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapBloodMNIST(BloodMNIST):
    def __init__(self, **kwargs):
        super(WrapBloodMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapBloodMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapTissueMNIST(TissueMNIST):
    def __init__(self, **kwargs):
        super(WrapTissueMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapTissueMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapOrganAMNIST(OrganAMNIST):
    def __init__(self, **kwargs):
        super(WrapOrganAMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOrganAMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapOrganCMNIST(OrganCMNIST):
    def __init__(self, **kwargs):
        super(WrapOrganCMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOrganCMNIST, self).__getitem__(idx)
        return image, int(label[0])

class WrapOrganSMNIST(OrganSMNIST):
    def __init__(self, **kwargs):
        super(WrapOrganSMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOrganSMNIST, self).__getitem__(idx)
        return image, int(label[0])

medmnist_classes = {
    'pathmnist': WrapPathMNIST,
    'dermamnist': WrapDermaMNIST,
    'octmnist': WrapOCTMNIST,
    'bloodmnist': WrapBloodMNIST,
    'tissuemnist': WrapTissueMNIST,
    'organamnist': WrapOrganAMNIST,
    'organcmnist': WrapOrganCMNIST,
    'organsmnist': WrapOrganSMNIST
}

medmnist_n_classes = {
    'pathmnist': 9,
    'dermamnist': 7,
    'octmnist': 4,
    'bloodmnist': 8,
    'tissuemnist': 8,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11
}

medmnist_n_channels = {
    'pathmnist': 3,
    'dermamnist': 3,
    'octmnist': 1,
    'bloodmnist': 3,
    'tissuemnist': 1,
    'organamnist': 1,
    'organcmnist': 1,
    'organsmnist': 1
}