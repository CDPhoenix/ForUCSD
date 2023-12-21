%heights1 = {'3.3'};
%heights2 = {'4.85'};
%heights3 = {'3.3'};
%heights1 = cell(1,3);
%heights2 = cell(1,3);
%heights3 = cell(1,3);

%row_spacing = cell(1,3);


%row_spacing = {5.81,6.54,7.26,7.99,8.72,9.45};


 
solidomain = [2,3,4,5,6,7,8,9,10,11];

%isosurfs = [6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,...
%    30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65];

isosurfs = [4,7,8,11,12,15,16,19,20,23,24,27,...
    28,31,32,35,36,39,40,43];

%heatsurfs = [7,13,19,25,31,37,43,49,55,61];
heatsurfs = [5,6,9,10,13,14,17,18,21,22,25,26,29,30,...
    33,34,37,38,41,42];
counting = 1;
%dataset = Arrangement(row_spacing);
%save('dataset.mat','dataset');
dataset_new = load('dataset_new.mat');
dataset = dataset_new.dataset_new;
model = mphload('container.mph');



for i = 1:length(dataset)
    
    data = dataset(i,:);
    
    heights1 = num2str(data(1,1));
    heights2 = num2str(data(1,2));
    heights3 = num2str(data(1,3));
    row_spacing = data(1,4);
    velocity = data(1,5);
    
    model.param.set('Panel_height1', [heights1,'[m]']);
    model.param.set('Panel_height2', [heights2,'[m]']);
    model.param.set('Panel_height3', [heights3,'[m]']);
    
    model.param.set('row_spacing', [num2str(row_spacing),'[m]']);
    model.param.set('U', [num2str(velocity),'[m/s]']);
    model.param.set('Bottom_Height', '1.52[m]');
    model.param.set('Up_height', '1.52[m]');
    
    if strcmp(heights1,heights2) ~= 1&& strcmp(heights1,heights3) == 1
       
       model.geom('geom1').feature('arr1').set('linearsize', '3');
       model.geom('geom1').feature('arr1').set('displ', [row_spacing*4;0]);
       model.geom('geom1').feature('arr2').set('linearsize', '5');
       model.geom('geom1').feature('arr2').set('displ', [row_spacing*2;0]);
       model.geom('geom1').feature('arr3').set('linearsize', '2');
       model.geom('geom1').feature('arr3').set('displ', [row_spacing*4;0]);
       
       if str2double(heights2) > str2double(heights3)
           model.param.set('Bottom_Height', [heights3,'[m]']);
           model.param.set('Up_height', [heights2,'[m]']);
       else
           model.param.set('Bottom_Height', [heights2,'[m]']);
           model.param.set('Up_height', [heights2,'[m]']);
       end
            
    else
            
       model.geom('geom1').feature('arr1').set('linearsize', '4');
       model.geom('geom1').feature('arr1').set('displ', [row_spacing*3;0]);
       model.geom('geom1').feature('arr2').set('linearsize', '3');
       model.geom('geom1').feature('arr2').set('displ', [row_spacing*3;0]);
       model.geom('geom1').feature('arr3').set('linearsize', '3');
       model.geom('geom1').feature('arr3').set('displ', [row_spacing*3;0]);
       
       %if str2double(heights2) > str2double(heights1)
       %    model.param.set('Bottom_Height', [heights1,'[m]']);
       %    model.param.set('Up_height', [heights2,'[m]']);
       %else
       %    model.param.set('Bottom_Height', [heights2,'[m]']);
       %    model.param.set('Up_height', [heights2,'[m]']);
       %end
       
    end
    model.geom('geom1').run
    disp('Geometry Modification Completed');
        
    model.material('mat2').selection().set(solidomain);
    model.physics('ht2').feature('solid2').selection().set(solidomain);
    model.physics('ht2').feature('ins2').selection().set(isosurfs);
    model.physics('ht2').feature('temp1').selection().set(heatsurfs);
    model.physics('ht2').feature('init2').selection().set(solidomain);

    disp('Parameters Modification Completed');
        
    Filename = [heights1,'_',heights2,'_',heights3,'_',num2str(row_spacing),'_',num2str(velocity),'.mph'];
    
    disp(datestr(now))  
    disp(['Caes_',num2str(counting),'_Studying']);
    model.study('std1').run;
    disp(['Case_',num2str(counting),'_Complete']);
    counting = counting + 1;
    
    model.result.dataset('cln1').run()
    model.result.dataset('cln2').run()
    model.result.dataset('cln3').run()
    model.result.dataset('cln4').run()


    mphsave(model,Filename);
    disp('model saving complete')
    disp(datestr(now))
    
    
    
end
