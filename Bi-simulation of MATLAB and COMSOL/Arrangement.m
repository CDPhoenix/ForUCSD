function dataset = Arrangement(rowspacing)

    %Transfer the rowspacing from cell to double array
    
    rowspacing = cell2mat(rowspacing);
    rowspacing = rowspacing';
    rowspacing = sortrows(rowspacing);
    
    %Generate different heights arrangement
    
    [H1,H2,H3] = ndgrid(3.3:1.55:6.4);
    %[H1,H2,H3] = ndgrid(3.3:1:3.3);
    heights_arrangement = [H1(:),H2(:),H3(:)];
    
    %Repeat both heights arrangement and rowspacing arrangement
    
    heights_arrangement_temp = repmat(heights_arrangement,length(rowspacing),1);
    heights_arrangement_temp = sortrows(heights_arrangement_temp);
    rowspacing_temp = repmat(rowspacing,length(heights_arrangement),1);
    
    %Concat two matrics
    
    dataset = [heights_arrangement_temp,rowspacing_temp];
    
end