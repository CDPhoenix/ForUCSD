function [ArrayLog] = Solar_Logical3D(baseheight, GCR, Lmod,thick, angle, Depth, row_num, showArray, resolution, baseheight2,baseheight3,modelname,export)
%% The following function is designed to create a spatial logical matrix
%   representing space occupied by panels within the volume of a given solar array.
%
%     Primary use case is to create a volumetric logical representation of
%       solar panel presence for use in lacunarity calculations
%
%
%     **Note** This function does not include occupancy based on module support.
%          In order to consider support structure, one might instead consider
%           conversion from 3D .stl representation of the array to a logical matrix
%
%
% Inputs:
%
%   - baseheight: height from ground-to-panel lower edge [m]
%   - GCR: Ground Coverage Ratio; Lmod/S  [m/m]
%   - Lmod: Length of entire module [m]
%   - thick: Module Thickness [m]
%   - angle: module inclination angle [degree]
%   - Depth: spanwise distance occupied by modules [m]
%   - row_num: number of rows to consider/create within array
%   - showArray: Whether to create a figure representing the array (1 if
%   yes, 0 if no)
%   - resolution: (optional) Desired resolution of resulting logical array [m] ;
%           **Default: 0.5 times module thickness (resolution = 0.5*thick);
%   - baseheight2: (optional) secondary height from ground to panel edge - for
%   staggered arrays with baseheight alternating rows only.
%       **NOTE** Do not define if array all same height
%
% Output:
%   - ArrayLog: logical 3D matrix (1's and 0's) representing locations
%   where array modules are present
%     **Orientation of array dimensions:
%           (dim1, dim2, dim3) = (x,y,z) = (array-front-to-rear, vertical, spanwise)
%
% See Solar_ReadMe for visual example
%
% Created by Sarah E. Smith, 2022
%   Affiliation (at time of creation): Portland State University
%

%%
if nargin<9
    resolution = thick/2;   % Default resolution of logical array
end

if nargin<10   %If no second baseheight given, all modules same baseheight(i.e. not staggered heights)
    H_pan = baseheight+(Lmod*sind(angle));  % Height of panel upper edge [m]
    Lmod_x = Lmod*cosd(angle);      % Module length projected in x [m]
    S = Lmod/GCR;                   % Spacing between modules (e.g. low edge to low edge) [m]
        Smid = S-Lmod_x;        % Empty space distance between modules (i.e. upper edge to lower edge) [m]
    Ltot_x = row_num*(Lmod_x+Smid);    % Total array length [m]

    %Create empty volume to insert array
    ArrayLog = zeros(round(Ltot_x/resolution), round(H_pan/resolution), round(Depth/resolution));
        [mA, nA, oA] = size(ArrayLog);

    % Create Panel matrices to input into empty ArrayLog:
    Pmat = logical(repmat(imrotate(ones(floor(Lmod/resolution), floor(thick/resolution)), angle), 1, 1, oA));
    [m,n,~] = size(Pmat);%floor(),向下取整

    % Input panel into ArrayLog
    SmidInd = round(Smid/(2*resolution));   % obtain index of the point central between modules
    SInd = round(S/resolution);             % how many indices are present between module front edges
    BInd = round(baseheight/resolution - thick*cosd(angle)/resolution); % index from ground to lower panel edge

    if BInd+n > nA
        ArrayLog = [ArrayLog zeros(mA, BInd+n-nA,oA)];
    end

    Pcount = 0; %Counter to keep track of rows
    for jj = 1:row_num      %For each row needed

        %Place module in appropriate location
        ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd+1:BInd+n,:) = Pmat;
        if jj==1
            % Declare array as a logical if not done already
            ArrayLog = logical(ArrayLog);
        end
        Pcount = Pcount+1;  %increase row counter
    end

else
    %% This will only initiate if dealing with staggered arrays

    H_panLow = baseheight+(Lmod*sind(angle));  %Height of upper panel edge (short panel)
    H_panHi = baseheight2+(Lmod*sind(angle));  %Height of upper panel edge (tall panel)
    H_pan3 = baseheight3 + (Lmod*sind(angle));

    compare = [H_panLow,H_panHi,H_pan3];
    
    H_pan = max(compare);

    %H_pan = mean(compare);

    Lmod_x = Lmod*cosd(angle);      % Module length projected in x [m]
    S = Lmod/GCR;                   % Spacing between modules (e.g. low edge to low edge) [m]
        Smid = S-Lmod_x;        % Empty space distance between modules (i.e. upper edge to lower edge) [m]

    Ltot_x = row_num*(Lmod_x+Smid);    % Total array length [m]

    %Create empty volume to insert array
    %ArrayLog = zeros(round(Ltot_x/resolution), round(H_panHi/resolution), round(Depth/resolution));
        %[mA, nA, oA] = size(ArrayLog);

    ArrayLog = zeros(round(Ltot_x/resolution), round(H_pan/resolution), round(Depth/resolution));
    [mA, nA, oA] = size(ArrayLog);

    % Create Panel matrices:
    Pmat = logical(repmat(imrotate(ones(floor(Lmod/resolution), floor(thick/resolution)), angle), 1, 1, oA));
    [m,n,~] = size(Pmat);

    % Input panel into ArrayLog
    SmidInd = round(Smid/(2*resolution));
    SInd = round(S/resolution);
    BIndLow = floor(baseheight/resolution - thick*cosd(angle)/resolution);
    BIndHi = floor(baseheight2/resolution - thick*cosd(angle)/resolution);
    BInd3 = floor(baseheight3/resolution - thick*cosd(angle)/resolution);

    %compare = [BIndLow,BIndHi,BInd3];
    %BIndHiest = max(compare);


    if BIndLow+n > nA   % If length of array with panels will be greater than previously allocated
        ArrayLog = [ArrayLog zeros(mA, BIndLow+n-nA,oA)];
    end

    Pcount = 0; %Initialize counting panels
    if baseheight == baseheight3 && baseheight ~= baseheight2
        for jj = 1:row_num
            if mod(jj,2)
                %if mod(jj,3)==0
                %    ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd3+1:BInd3+n,:) = Pmat;
                %else
                ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndLow+1:BIndLow+n,:) = Pmat;
                if jj ==1
                     ArrayLog = logical(ArrayLog);
                end
           
            else
                ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndHi+1:BIndHi+n,:) = Pmat;
            end
            Pcount = Pcount+1;
        end
    else
        for jj = 1:row_num
            if mod(jj,3)
                if mod(jj+1,3)==0
                    ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndHi+1:BIndHi+n,:) = Pmat;
                    %ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd3+1:BInd3+n,:) = Pmat;
                else
                    ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndLow+1:BIndLow+n,:) = Pmat;
                    ArrayLog = logical(ArrayLog);
                    %if jj==1
                    %   ArrayLog = logical(ArrayLog);
                    %end
                end
            else
                %ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndHi+1:BIndHi+n,:) = Pmat;
                ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd3+1:BInd3+n,:) = Pmat;
            end
    
            Pcount = Pcount+1;
        end
    end
    %for jj = 1:row_num
    %    if mod(jj,3)
    %        if mod(jj+1,3)==0
    %            ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndHi+1:BIndHi+n,:) = Pmat;
                %ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd3+1:BInd3+n,:) = Pmat;
    %        else
    %            ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndLow+1:BIndLow+n,:) = Pmat;
    %            ArrayLog = logical(ArrayLog);
                %if jj==1
                %   ArrayLog = logical(ArrayLog);
                %end
    %        end
    %    else
            %ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BIndHi+1:BIndHi+n,:) = Pmat;
    %        ArrayLog(SmidInd+(SInd*Pcount):SmidInd+(SInd*Pcount)+m-1,BInd3+1:BInd3+n,:) = Pmat;
    %    end

    %    Pcount = Pcount+1;
    %end
end
%% This section save ArrayLog Matrix
if export == 1
    MatFileName = [modelname,'.mat'];
    save(MatFileName,'ArrayLog');
end
%% This section will plot/show the array just created
[mA, nA, oA] = size(ArrayLog);
xx = 1:mA; yy = 1:nA; zz = 1:oA;
if showArray==1
    X = repmat(xx', 1, nA, oA);
    Y = repmat(yy, mA, 1, oA);
    for ii = 1:length(zz)
        Z(:,:,ii) = repmat(zz(ii), mA, nA, 1);
    end
    figure(); scatter3(X(ArrayLog), Z(ArrayLog), Y(ArrayLog), 40, 'filled')
    xlim([1 mA]); ylim([1 oA]); zlim([1 nA])
    xlabel('x'); ylabel('z'); zlabel('y')
    title(['Rep Array: Resolution = ' num2str(resolution) 'm; GCR = ' num2str(GCR) '; B = ' num2str(baseheight) '; Lmod = ' num2str(Lmod) '; $\alpha = $' num2str(angle)],...
        'Interpreter', 'latex')
end
end
