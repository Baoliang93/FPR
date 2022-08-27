import random
import json

DATA_DIR = "../../datasets/tid2013/distorted_images/"
REF_DIR = "../../datasets/tid2013/reference_images/"
MOS_WITH_NAMES = "../datasets/tid2013/mos_with_names.txt"


EXCLUDE_INDICES = (25,)
EXCLUDE_TYPES = ()


data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]
def _write_list_into_file(l, f):
    with open(f, "w") as h:
        for line in l:
            h.write(line)
            h.write('\n')


N = 25 - len(EXCLUDE_INDICES)
idcs = list(range(1,N+1))


for prot in range(10):
        random.shuffle(idcs)
        print('idcs:', idcs)        
        train_idcs = idcs[:15]
        val_idcs = idcs[15:20]
        test_idcs = idcs[20:]
        
        train_images, train_labels, train_mos = [], [], []
        val_images, val_labels, val_mos = [], [], []
        test_images, test_labels, test_mos = [], [], []
        
        for mos, image in data_list:
            ref = REF_DIR + "I" + image[1:3] + '.BMP'
            img = DATA_DIR + image
            idx = int(image[1:3])
            tpe = int(image[4:6])
            if idx not in EXCLUDE_INDICES and tpe not in EXCLUDE_TYPES:
                if idx in train_idcs:
                    train_images.append(img)
                    train_labels.append(ref)
                    train_mos.append(float(mos))
                if idx in val_idcs:
                    val_images.append(img)
                    val_labels.append(ref)
                    val_mos.append(float(mos))
                if idx in test_idcs:
                    test_images.append(img)
                    test_labels.append(ref)
                    test_mos.append(float(mos))            

        
        ns = vars()
        for ph in ('train', 'val', 'test'):
            data_dict = dict(img=ns['{}_images'.format(ph)], ref=ns['{}_labels'.format(ph)], score=ns['{}_mos'.format(ph)])
            with open('TID_List_6-2-2-Wo25/{}_'.format(ph)+str(prot)+'_data.json', 'w') as fp:
                json.dump(data_dict, fp)




