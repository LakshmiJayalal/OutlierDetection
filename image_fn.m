function data1=image_fn(dataset_id,res)
    dir_path=''; %directory in which datasets are placed
     %id of the dataset: 1 for waving trees
    
    %list of video datasets:
    %the images from the dataset are to be kept in the folder named as any one of the entry from 'dataset_name'
    dataset_name={'WavingTrees','MovedObject','Camouflage'};
    
    %format of images in respective dataset
    dataset_format={'bmp','bmp','bmp'};
    d=dir(strcat(dir_path,dataset_name{dataset_id},'/*.',dataset_format{dataset_id}));
    str1=strcat(dir_path,dataset_name{dataset_id},'/');
    
    %collect images from datasets as columns of the variable
    for ii=1:length(d)-1
        a=d(ii).name; a=strcat(str1,a);
        %frame id can be collected from this variable 'a'
        img=imresize(imread(a),res);
        imSize=size(img);
        if length(imSize)>=3
            img=rgb2gray(img);
            imSize=imSize(1:2);
        end
        img=img(:);
        data(:,ii)=img;
    end
    
    %test whether data is collected appropriately:
     data1=data;%(:,241:260);
%     imagesc(reshape(data(:,243),imSize)); colormap('gray');
end