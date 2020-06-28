% resize to 256*256
file_path1='E:\wyq\rgb\';
file_path2='E:\wyq\rgb256\';

img_path_list = dir(strcat(file_path1,'*.JPEG'));       % get all the JPEG images  
img_num = length(img_path_list);                        % get the total number of images

if img_num > 0                                                   
    for j = 1:img_num                                            
       image_name_old = img_path_list(j).name;                  
       image = imread(strcat(file_path1,image_name_old));
       new = imcrop(image,[20,20,256,256]);                 
       imwrite(new,strcat(file_path2,image_name_old))
    end
end

       
       
       

       
