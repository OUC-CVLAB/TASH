from config import *


class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None):
        self.img_dir = img_dir
        # self.file_list = np.loadtxt(txt_dir, dtype='str')  # qss
        self.file_list = open(txt_dir).readlines()
        self.NB_CLS = NB_CLS

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        image = Image.open(img_name)

        if self.NB_CLS != None:
            if len(self.file_list[idx]) > 2:
                label = [int(self.file_list[idx][i]) for i in range(1, self.NB_CLS+1)]
                label = T.FloatTensor(label)
            else:
                label = int(self.file_list[idx][1])  # qss
            return transforms.ToTensor()(image), label
        else:
            return transforms.ToTensor()(image)


class ImageList(object):
    def __init__(self, data_path, image_list):
        if len(image_list[0].split()) > 2:
            self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in
                         image_list]  # TODO qss change 1: to 2: for SUN multi-attribute training
        else:
            self.imgs = [(data_path + val.split()[0],  int(val.split()[1])) for val in image_list]

        self.names = [val.split()[0].split('/')[-1] for val in image_list]##bgr

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = transforms.ToTensor()(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)
