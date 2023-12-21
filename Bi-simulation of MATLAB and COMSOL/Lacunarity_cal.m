classdef Lacunarity_cal
    methods(Static)
        function [Lsc,H] = calculation(model,modelname,export)
            %B = model.geom('geom1').feature('blk2').getDouble('z'); 
            % Ground-to-panel height [m]
            [B1,~,~,~] = mphevaluate(model, 'Panel_height1');
            [B2,~,~,~] = mphevaluate(model, 'Panel_height2');
            [B3,~,~,~] = mphevaluate(model, 'Panel_height3');

            
           
            
            %[B3,~,~,~] = mphevaluate(model, 'Panel_height3');
            angle = 30;     % Panel inclination [deg]
            angle_inrad = pi/6;
            %Lmod = model.geom('geom1').feature('blk2').getDouble('ly')*cos(angle);    % Module length [m] (lower-to-upper edge)
            [Lmod,~,~,~] = mphevaluate(model, 'Panel_length');
            %spacing_array = model.geom('geom1').feature('arr1').getDoubleArray('displ');
            [spacing_array,~,~,~] = mphevaluate(model, 'row_spacing');
            %S = (spacing_array(1,1)^2 + spacing_array(2,1)^2)^0.5;     % Module row-to-row spacing [m]
            S = spacing_array;
            GCR = Lmod/S;       % ground coverage ratio 
            %thick = model.geom('geom1').feature('blk2').getDouble('lz');   % Panel Thickness [m]
            [thick,~,~,~] = mphevaluate(model, 'Panel_thickness');
            %Depth = model.geom('geom1').feature('blk2').getDouble('lx');      % 'Spanwise' module width [m]
            Depth =2.0;%mphevaluate(model,'Panel_spanwise');
            showArray = 0;      % Show/plot Representative array after creation? (1 = true; 0 = false)
            %row_num = model.geom('geom1').feature('arr1').getDouble('linearsize');        % Number of module rows within the array
            row_num = 10;
            resolution = 0.5*thick;%0.001    % Spatial resolution of created matrix [m] (pref.: res <= 0.5*L_{minimumFeatureLength})
            
            %H1 = B1 + Lmod * sin(angle);
            H = B2 + Lmod * sin(angle_inrad);
            %H3 = B3 + Lmod * sin(angle);
            %B1 = B1*100;
            %B2 = B2*100;
            %B3 = B3*100;
            modelname = [modelname,'Array'];
            Lsc = Lacunarity_cal.Lsc_cal(B1, GCR, Lmod, thick, angle, Depth, row_num, showArray, resolution,B2,B3,modelname,export);
            %H = mean([B1,B2,B3]);
            %if H1 == H3 || H1~=H2
            %    H = H2;
            %else
            %    H = mean([H1,H2,H3]);
            %end
            
            %H = H2;
            

        end
        function Lsc = Lsc_cal(B1, GCR, Lmod, thick, angle, Depth, row_num, showArray, resolution,B2,B3,modelname,export)
            ArrLog = Solar_Logical3D(B1, GCR, Lmod, thick, angle, Depth, row_num, showArray, resolution,B2,B3,modelname,export);
             % Define how many boxes for lacunarity.m
            n_p_Now = floor((1/2)*max(size(ArrLog)));
            tic
            
            %obtain lacunarity (L), box sizes
            [L, ~, R, ~, ~, ~] = lacunarity(ArrLog, n_p_Now);
        
            R_m = R.*resolution;        % translate box sizes into physical length
            
            asy = find(islocalmin(L),1,'first');    % Find largest relevant length scale
                
            Lsc = (sum(L(1:asy).*R_m(1:asy)))/length(R_m(1:asy));   % calculate lacunarity length scale as in Smith 2022.
            %Lsc = 53*Lsc;    
        end
    end
end