% Get goal JPEG images

file_path1='E:\wyq\itrain\';      % raw image path
file_path2='E:\wyq\itrain90goal\'; % goal image path

img_path_list = dir(strcat(file_path1,'*.bmp'));         
img_num = length(img_path_list);                         

if img_num > 0                                                    
    for j = 1:img_num                                            
       image_name_old = img_path_list(j).name;                  
        if j<10
            image_name_new = strcat(num2str(j),'.bmp');
            image_name_new2 = strcat(num2str(j),'.jpg');
            image = imread(strcat(file_path1,image_name_new));
            imwrite(image,strcat(file_path2,image_name_new2),'jpeg','Quality',90);  % JPEG compression with quality factor **
        elseif j>=10 && j<100
            image_name_new = strcat(num2str(j),'.bmp');
            image_name_new2 = strcat(num2str(j),'.jpg');
            image = imread(strcat(file_path1,image_name_new));
            imwrite(image,strcat(file_path2,image_name_new2),'jpeg','Quality',90);
        elseif j>=100 
            image_name_new = strcat(num2str(j),'.bmp');
            image_name_new2 = strcat(num2str(j),'.jpg');
            image = imread(strcat(file_path1,image_name_new));
            imwrite(image,strcat(file_path2,image_name_new2),'jpeg','Quality',90);
        end
    end 
end










       
       
       

       
