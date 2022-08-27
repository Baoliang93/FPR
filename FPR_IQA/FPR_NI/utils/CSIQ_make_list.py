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

DATA_DIR = "../../datasets/CSIQ/dst_imgs/"
REF_DIR = "../../datasets/CSIQ/src_imgs/"
MOS_WITH_NAMES = "../../datasets/CSIQ/ref_dist_mos.txt"


data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]

idcs = ['1600.png','aerial_city.png','boston.png','bridge.png','butter_flower.png','cactus.png',
        'child_swimming.png','couple.png','elk.png','family.png','fisher.png','foxy.png',
        'geckos.png','lady_liberty.png','lake.png','log_seaside.png','monument.png','native_american.png',
        'redwood.png','roping.png','rushmore.png','shroom.png','snow_leaves.png','sunsetcolor.png',
        'sunset_sparrow.png','swarm.png','trolley.png','veggies.png','woman.png','turtle.png',]

dist_list = ['AWGN','JPEG','jpeg2000','fnoise','BLUR','contrast']

for prot in range(10):

        random.shuffle(idcs)    
        train_idcs = idcs[:18]
        val_idcs = idcs[18:24]
        test_idcs = idcs[24:]
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
        
        for ref_ini, dist_id,dist_type,dist_level,mos_std,mos in data_list:
            idx = ref_ini+'.png'
            ref = REF_DIR + idx
            img = DATA_DIR +dist_list[int(dist_id)-1].lower()+'/'+\
                  ref_ini+'.'+dist_list[int(dist_id)-1]+'.'+str(dist_level)+'.png'
            
        
            if idx in train_idcs:
                train_images.append(img)
                train_labels.append(ref)
                train_mos.append(float(mos)*100)
            if idx in val_idcs:
                val_images.append(img)
                val_labels.append(ref)
                val_mos.append(float(mos)*100)
            if idx in test_idcs:
                test_images.append(img)
                test_labels.append(ref)
                test_mos.append(float(mos)*100)           
        
        
        ns = vars()
        for ph in ('train', 'val', 'test'):
            data_dict = dict(img=ns['{}_images'.format(ph)], \
                             ref=ns['{}_labels'.format(ph)], \
                             score=ns['{}_mos'.format(ph)])
            with open('scripts/{}_'.format(ph)+str(prot)+'_data.json', 'w') as fp:
                json.dump(data_dict, fp)




