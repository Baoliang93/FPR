# A script to make data lists for pytorch code
# Database: TID2013
# Date: 2018-11-6
# 
# Edited: 2019-5-7
# Change log:
#    + txt -> json
#    + mos saved as float values

import random
import json

DATA_DIR = "../../datasets/databaserelease2/"
REF_DIR = "../../datasets/databaserelease2/refimgs/"
MOS_WITH_NAMES = "../../datasets/databaserelease2/MOS_NAME_REF.txt"
data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]


idcs = ['bikes.bmp', 'building2.bmp', 'buildings.bmp', 'caps.bmp', 'carnivaldolls.bmp',
        'cemetry.bmp', 'churchandcapitol.bmp', 'coinsinfountain.bmp', 'dancers.bmp', 
        'flowersonih35.bmp', 'house.bmp', 'lighthouse.bmp', 'lighthouse2.bmp', 
        'manfishing.bmp',' monarch.bmp', 'ocean.bmp', 'paintedhouse.bmp', 
        'parrots.bmp', 'plane.bmp', 'rapids.bmp', 'sailing1.bmp', 'sailing2.bmp', 
        'sailing3.bmp', 'sailing4.bmp', 'statue.bmp', 'stream.bmp', 'studentsculpture.bmp',
        'woman.bmp', 'womanhat.bmp']

for prot in range(10):

        random.shuffle(idcs)    
        train_idcs = idcs[:17]
        val_idcs = idcs[17:23]
        test_idcs = idcs[23:29]
        print('train_idcs:',train_idcs,'\n')
        print('val_idcs:',val_idcs,'\n')
        print('test_idcs:',test_idcs,'\n')
        
        def _write_list_into_file(l, f):
            with open(f, "w") as h:
                for line in l:
                    h.write(line)
                    h.write('\n')
        
        train_images, train_labels, train_mos = [], [], []
        val_images, val_labels, val_mos = [], [], []
        test_images, test_labels, test_mos = [], [], []
        
        for mos, image,ref in data_list:
            idx = ref
            ref = REF_DIR + ref
            img = DATA_DIR + image
            
        
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
            data_dict = dict(img=ns['{}_images'.format(ph)], \
                             ref=ns['{}_labels'.format(ph)], \
                             score=ns['{}_mos'.format(ph)])
            with open('{}_'.format(ph)+str(prot)+'_data.json', 'w') as fp:
                json.dump(data_dict, fp)




