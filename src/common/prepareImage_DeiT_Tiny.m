function [ preprocessed_image ] = prepareImage_DeiT_Tiny( ImageIndexPrefix, ImageIndex, input_size)

    %% set mean std and crop_pct
    mean = [0.485, 0.456, 0.406];%RGB
    std = [0.229, 0.224, 0.225];%RGB

    % crop_pct = 0.875;

    % Initialize the preprocessed_images array with zeros
      preprocessed_image = zeros(input_size, input_size, 3, 'single');

    % Add a for loop to process all images one by one
    for i = 1:size(ImageIndex, 1)
%         disp(['Starting preprocessing for image ', num2str(i), '...']);
        
        %% read image
        fileName = strcat(ImageIndexPrefix, num2str(ImageIndex(i),'%08d'),'.JPEG');
        image_RGB = imread(fileName); % H-W-C(RGB)
        
%         disp(['Original image size for image ', num2str(i), ': ']);

        if(size(image_RGB,3)==1) % black white image
            image_RGB = cat(3, image_RGB, image_RGB, image_RGB);
        end

        %% Resize image*64*
        IMAGE_DIM = 256;
        if(size(image_RGB,1)> size(image_RGB,2))
		    image_resized = imresize(image_RGB, [round(IMAGE_DIM*size(image_RGB,1)/size(image_RGB,2)),IMAGE_DIM], 'bilinear');  % resize image_RGB
        else
            image_resized = imresize(image_RGB, [IMAGE_DIM,round(IMAGE_DIM*size(image_RGB,2)/size(image_RGB,1))], 'bilinear');  % resize image_RGB
        end
       
        %% Center crop
        [height, width, ~] = size(image_resized);
        center_x = floor(width / 2);
        center_y = floor(height / 2);
        half_new_w = floor(input_size / 2);
        half_new_h = floor(input_size / 2);

        % Calculate crop rectangle
        crop_rect = [center_x - half_new_w + 1, center_y - half_new_h + 1, input_size - 1, input_size - 1];

        % Crop the image
        cropped_image = imcrop(image_resized, crop_rect);
        
        %% Convert the data to 32-bit floating-point type and divide the pixel values by 255
        convert2float_image = single(cropped_image);
        convert2float_image = convert2float_image/255; 

        %% Normalize the image
        normalized_image = convert2float_image;
        for j = 1:3
            normalized_image(:, :, j) = (normalized_image(:, :, j) - mean(j)) / std(j);
        end

        %% Convert H-W-C to W-H-C to serve matlab and OpenCL
        preprocessed_image(:,:,:,i) = permute(normalized_image, [2, 1, 3]);
    %     disp(['Resized and normalized image size for image ', num2str(i), ': ']);
    %     disp(size(normalized_image)) ;
    %     preprocessed_image = permute(image_BGR, [2, 1, 3]);
    %     preprocessed_image = permute(image_resized, [2, 1, 3]);
    %     preprocessed_image = permute(cropped_image, [2, 1, 3]);
    %     preprocessed_image = permute(convert2float_image, [2, 1, 3]);
    end
end

