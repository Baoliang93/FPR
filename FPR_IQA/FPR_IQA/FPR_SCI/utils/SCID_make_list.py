import random
import json

DATA_DIR = "../../datasets/SCID/DistortedSCIs/"
REF_DIR = "../../datasets/SCID/ReferenceSCIs/"
MOS_WITH_NAMES = "../../datasets/SCID/MOS_SCID.txt"



EXCLUDE_INDICES = ()
EXCLUDE_TYPES = ( )

data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]

N = 40 - len(EXCLUDE_INDICES)
def _write_list_into_file(l, f):
    with open(f, "w") as h:
        for line in l:
            h.write(line)
            h.write('\n')

for prot in range(1,31):
    
    train_images, train_labels, train_mos = [], [], []
    val_images, val_labels, val_mos = [], [], []
    test_images, test_labels, test_mos = [], [], []

    idcs = list(range(N))
    idcs = [i + 1 for i in idcs]
    random.shuffle(idcs)
    print('idcs:', idcs)
    train_idcs = idcs[:24] #40*0.6=24, *0.8=32
    val_idcs = idcs[24:32]
    test_idcs = idcs[32:]
    
    for ref,image,mos in data_list:
        idx = int(image.split('_')[0][3:])
        ref = REF_DIR + ref+'.bmp'
        img = DATA_DIR + image+'.bmp'
          
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
        with open('sci_scripts/scid-scripts-6-2-2/{}_'.format(ph)+str(prot)+'_data.json', 'w') as fp:
            json.dump(data_dict, fp)




