function ImageLabel = getLabel(LabelTable, ImageIndex )
	ImageLabel=zeros(size(ImageIndex,1),1);
	for i=1:size(ImageIndex,1)
		ImageLabel(i)=LabelTable(ImageIndex(i));
	end
end
