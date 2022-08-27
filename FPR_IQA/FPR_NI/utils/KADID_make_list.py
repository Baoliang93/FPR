import random
import json

DATA_DIR = "../../datasets/kadid10k/images/"
REF_DIR = "../../datasets/kadid10k/images/"
MOS_WITH_NAMES = "../../datasets/kadid10k/names_with_mos.txt"


EXCLUDE_INDICES = ()
EXCLUDE_TYPES = ()
data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]
# Split the dataset by index
N = 81-len(EXCLUDE_INDICES)
idcs = list(range(1,N+1))


for prot in range(10):
        random.shuffle(idcs)

        
        train_idcs = idcs[:49]
        val_idcs = idcs[49:65]
        test_idcs = idcs[65:]
        print('train_idcs:',train_idcs)
        print('val_idcs:',val_idcs)
        print('test_idcs:',test_idcs)
               
        def _write_list_into_file(l, f):
            with open(f, "w") as h:
                for line in l:
                    h.write(line)
                    h.write('\n')
        
        train_images, train_labels, train_mos = [], [], []
        val_images, val_labels, val_mos = [], [], []
        test_images, test_labels, test_mos = [], [], []
        
        for image, mos  in data_list:
            ref = REF_DIR + "I" + image[1:3] + '.png'
            img = DATA_DIR + image
            idx = int(image[1:3])
            if idx not in EXCLUDE_INDICES :
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
            with open('./scripts/{}_'.format(ph)+str(prot)+'_data.json', 'w') as fp:
                json.dump(data_dict, fp)




