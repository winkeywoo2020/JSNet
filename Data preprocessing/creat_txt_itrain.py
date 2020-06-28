# creat data loading txt
import os.path
import glob
import os
if __name__ == "__main__":
    realpath = os.path.realpath(__file__)
    dirname1 = '/home/amax/wyq/jpegmoni/90/itrain/'
    dirname = os.path.dirname(realpath)
    extension = 'bmp'
    file_list = glob.glob('*.' + extension)
    filetxt = open(os.path.join(dirname, 'itrain.txt'), 'w')
    for index, filename in enumerate(file_list):
        str_index = str(index)
        filepath = os.path.join(dirname1, filename)
        filetxt.write('%s\n' % (filepath))
    filetxt.close()




