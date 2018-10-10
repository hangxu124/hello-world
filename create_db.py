import commands
import os
import re




def create_db(caffe_root, images_path, txt_save_path):

    lmdb_name = 'distraction_img_train.lmdb'

    lmdb_save_path = caffe_root + 'examples/distraction_32/' + lmdb_name
 
    convert_imageset_path = caffe_root1 + 'build/tools/convert_imageset'
    cmd = """%s --shuffle --resize_height=200 --resize_width=200 %s %s %s"""
    status, output = commands.getstatusoutput(cmd % (convert_imageset_path, images_path, 
        txt_save_path, lmdb_save_path))
    print 'output:',output



if __name__ == '__main__':
    caffe_root1 = '/data/DTAA/code/z228757/ssd-caffe/SSD/caffe-ssd/'
    caffe_root = '/mnt/DTAA_data/DTAA/code/z638420/code/code/caffe-ssd/caffe-ssd/'

    my_caffe_project = caffe_root + 'examples/distraction_32/'

    images_path = caffe_root + 'examples/distraction/IMGS/'

    txt_name = 'train_val.txt'

    txt_save_path = my_caffe_project + txt_name

    #createFileList(images_path, txt_save_path)

    create_db(caffe_root, images_path, txt_save_path)

