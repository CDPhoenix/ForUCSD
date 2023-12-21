exper = cell(1,22);
plotgroup = cell(1,22);

exper{1} = 'front_u';
exper{2} = 'front_T';
exper{3} = 'front_muT';
exper{4} = 'front_Prt';
exper{5} = 'front_gradT';
exper{6} = 'back_u';
exper{7} = 'back_T';
exper{8} = 'back_muT';
exper{9} = 'back_Prt';
exper{10} = 'back_gradT';
exper{11} = 'bottom_u';
exper{12} = 'bottom_T';
exper{13} = 'bottom_muT';
exper{14} = 'bottom_Prt';
exper{15} = 'bottom_gradT';
exper{16} = 'up_u';
exper{17} = 'up_T';
exper{18} = 'up_muT';
exper{19} = 'up_Prt';
exper{20} = 'up_gradT';
exper{21} = 'Tcv';
exper{22} = 'Tm';


plotgroup{1} = 'plot1';%'plot1';
plotgroup{2} = 'plot2';%'plot2';
plotgroup{3} = 'plot3';%'plot1';
plotgroup{4} = 'plot4';%'plot2';
plotgroup{5} = 'plot5';%'plot1';
plotgroup{6} = 'plot6';%'plot2';
plotgroup{7} = 'plot7';%'plot1';
plotgroup{8} = 'plot8';%'plot1';
plotgroup{9} = 'plot9';%'plot1';
plotgroup{10} = 'plot10';%'plot1';
plotgroup{11} = 'plot11';%'plot1';
plotgroup{12} = 'plot12';%'plot1';
plotgroup{13} = 'plot13';%'plot1';
plotgroup{14} = 'plot14';%'plot2';
plotgroup{15} = 'plot15';%'plot1';
plotgroup{16} = 'plot16';%'plot1';
plotgroup{17} = 'plot17';%'plot1';
plotgroup{18} = 'plot18';%'plot1';
plotgroup{19} = 'plot19';%'plot1';
plotgroup{20} = 'plot20';%'plot1';
plotgroup{21} = 'data1';%'plot1';
plotgroup{22} = 'plot21';%'plot2';


dataset = load('dataset1.mat');

dataset = dataset.dataset;
%Breakpoint_cases = dataset(176:200,:);
%dataset = dataset(141:160,:);
%dataset = [3,3,3,7.26,3.6];
%dataset = dataset(101:end,:,:,:,:);
Heights = dataset(:,1:3);
Heights = unique(Heights,'rows');
row_spacings = unique(dataset(:,4));
velocitys = unique(dataset(:,end));

datapath = 'D:\PolyU\year3 sem02\URIS\COMSOL Practice\2DCases\2023.09.11\data\';
modelpath = 'D:\PolyU\year3 sem02\URIS\COMSOL Practice\2Dcases\2023.09.11\';

La_cal = Lacunarity_cal;
cal = ParameterCalculator;

export = 0;
simuExport = 0;
k_cond = 0.028;
Kv = 1.57e-5;
T_ref =300.15;

Pr = 0.69;


calMethod = 1; 
ResearchDataX = zeros(length(Heights),length(row_spacings)*length(velocitys));
ResearchDataY = zeros(length(Heights),length(row_spacings)*length(velocitys));
RE_sum = zeros(length(Heights),length(row_spacings)*length(velocitys));
hc_sum = zeros(length(Heights),length(row_spacings)*length(velocitys));
h_EXPECT_sum = zeros(length(Heights),length(row_spacings)*length(velocitys));
%ResearchDataX = zeros(1,length(row_spacings)*length(velocitys));
%ResearchDataY = zeros(1,length(row_spacings)*length(velocitys));
%RE_sum = zeros(1,length(row_spacings)*length(velocitys));

figure(1);

%data = [1.52,3,4.56,6.54,2.3];
for i=1:length(Heights)
    
    %data = dataset((i-1)*1 + 1:i*1,:);
    data = dataset((i-1)*20 + 1:i*20,:);
    heights1 = num2str(data(1,1));
    heights2 = num2str(data(1,2));
    heights3 = num2str(data(1,3));
    row_spacing = data(:,4);
    
    filename_prime = [datapath,heights1,'_',heights2,'_',heights3,'_'];
    modelname_prime = [heights1,'_',heights2,'_',heights3,'_'];
    
    
    for j = 1:length(row_spacings)
        row = num2str(row_spacings(j));
        x_axis = zeros(1,length(velocitys));
        y_axis = zeros(1,length(velocitys));
    
        NU = zeros(1,length(velocitys));
        LSC = zeros(1,length(velocitys));
        RE = zeros(1,length(velocitys));
        hc = zeros(1,length(velocitys));
        h_EXPECT = zeros(1,length(velocitys));
        V = zeros(1,length(velocitys));
        Q = zeros(1,length(velocitys));
        Q_EXPECT = zeros(1,length(velocitys));

        
        for vel=1:length(velocitys)
            %row = num2str(row_spacing(j));
            U = num2str(velocitys(vel));
            %U = '2.3';%扫雷御用
            model = mphload([modelname_prime,row,'_',U,'.mph']);
            model.result('pg11').selection().set([9]);
            model.result('pg11').run();
            model.result.export('data1').set('gridy2', ...
                'range(Panel_height2,0.01,Panel_height2 +Panel_thickness*cos(Attack_angle) + Panel_length*sin(Attack_angle))');
            model.result.export('data1').set('gridx2',...
                'range(Starting_pos+7*row_spacing-Panel_thickness*sin(Attack_angle),0.01,Starting_pos +7*row_spacing+ Panel_length*cos(Attack_angle))');
            modelname = [datapath,modelname_prime,row,'_',U,'_'];
            %model = mphload(modelname);
            temp = 0;
            datafilename3 = [filename_prime,row,'_',U,'_',exper{21},'.csv'];
            datafilename4 = [filename_prime,row,'_',U,'_',exper{22},'.csv'];
            model.result.export(plotgroup{21}).set('filename', datafilename3);
            model.result.export(plotgroup{21}).run;
            model.result.export(plotgroup{22}).set('filename', datafilename4);
            model.result.export(plotgroup{22}).run;
            Tcv = readmatrix(datafilename3);%readmatrix(datafilename3);
            Tcv = Tcv(:,end);
            Tcv(find(Tcv>=320.15))=[];
            Tcv = mean(Tcv);
            Tm = readmatrix(datafilename4);%readmatrix(datafilename4);
            Tm = mean(Tm(:,end));
            %rightnow_data = [data(1,1:3),row_spacings(j),velocitys(vel)];
            %Lia = ismember(rightnow_data,Breakpoint_cases, 'rows');

            for z = 1:4%length(exper)
                datafilename = [filename_prime,row,'_',U,'_',exper{(z-1)*5+1},'.csv'];
                datafilename2 = [filename_prime,row,'_',U,'_',exper{(z-1)*5+2},'.csv'];
                datafilename3 = [filename_prime,row,'_',U,'_',exper{(z-1)*5+3},'.csv'];
                datafilename4 = [filename_prime,row,'_',U,'_',exper{(z-1)*5+4},'.csv'];
                datafilename5 = [filename_prime,row,'_',U,'_',exper{z*5},'.csv'];
                model.result.export(plotgroup{(z-1)*5+1}).set('filename', datafilename);
                model.result.export(plotgroup{(z-1)*5+1}).run;
                model.result.export(plotgroup{(z-1)*5+2}).set('filename', datafilename2);
                model.result.export(plotgroup{(z-1)*5+2}).run;
                model.result.export(plotgroup{(z-1)*5+3}).set('filename', datafilename3);
                model.result.export(plotgroup{(z-1)*5+3}).run;
                model.result.export(plotgroup{(z-1)*5+4}).set('filename', datafilename4);
                model.result.export(plotgroup{(z-1)*5+4}).run;
                model.result.export(plotgroup{z*5}).set('filename', datafilename5);
                model.result.export(plotgroup{z*5}).run;
                u = readmatrix(datafilename);
                T_surf = readmatrix(datafilename2);
                muT = readmatrix(datafilename3);
                Prt = readmatrix(datafilename4);
                gradT = readmatrix(datafilename5);
                
                %u = u.data;
                %T_surf = T_surf.data;
                %muT = muT.data;
                %Prt = Prt.data;
                %gradT = gradT.d9ata;
                if length(unique(u(:,1)))==1
                    u = sortrows(u,2);
                    T_surf = sortrows(T_surf,2);
                    muT = sortrows(muT,2);
                    Prt = sortrows(Prt,2);
                    gradT = sortrows(gradT,2);
                    flag = 2;
                else
                    u = sortrows(u,1);
                    T_surf = sortrows(T_surf,1);
                    muT = sortrows(muT,1);
                    Prt = sortrows(Prt,1);
                    gradT = sortrows(gradT,1);
                    flag = 1;
                end
                
                %T_surf(find(T_surf(:,end)>=320.15),:)=[];
                %muT(find(T_surf(:,end)>=320.15),:)=[];
                %Prt(find(T_surf(:,end)>=320.15),:)=[];
                %gradT(find(T_surf(:,end)>=320.15),:)=[];
                

                standard = min([length(u),length(T_surf),length(muT),length(Prt),length(gradT)]);

                u = u(1:standard,:);
                T_surf = T_surf(1:standard,:);
                muT = muT(1:standard,:);
                Prt = Prt(1:standard,:);
                gradT = gradT(1:standard,:);
                
                %if Lia == 1
                %   u = u(:,end);
                %   T_surf = T_surf(:,end);
                %   muT = muT(:,end);
                %   Prt = Prt(:,end);
                %   gradT = gradT(:,end);
                %   integ1 = u.*(T_surf-Tcv);
                %   integ2 = muT./Prt.*gradT;
                %   temp = trapz(integ1)/length(integ1) - trapz(integ2)/length(integ2) + temp;
                %else
                u = [u(:,flag),u(:,end)];
                T_surf = [T_surf(:,flag),T_surf(:,end)];
                muT = [muT(:,flag),muT(:,end)];
                Prt = [Prt(:,flag),Prt(:,end)];
                gradT = [gradT(:,flag),gradT(:,end)];
                x = u(:,1);
                len = max(x) - min(x);
                integ1 = u(:,end).*(T_surf(:,end)-Tcv);
                integ2 = muT(:,end)./Prt(:,end).*gradT(:,end);
                temp = trapz(x,integ1)/len - trapz(x,integ2)/len + temp;



            end
            temp = abs(temp);
            q = 1005*1.161*temp;
            %q = q(:,end);
            %T = T(:,end);
            [Lsc,H] = La_cal.calculation(model,modelname,export);

            Nu = cal.NusseltNumber_cal(q,Tm,T_ref,H,k_cond,calMethod);
            h = cal.convcof_cal(q,Tm,T_ref,calMethod);
            
            Re = cal.Reynolds_cal(velocitys(vel),Lsc,Kv);
            
            x_axis_data = Re^0.2*Pr^(1/12);
            h_expect = cal.convexp_cal(Re,Pr,k_cond,H);
            %q_expect = h_expect * (mean(T(:,end))-T_ref);
            y_axis_data = Nu;
            
            x_axis(1,vel) = x_axis_data;
            y_axis(1,vel) = y_axis_data;
            
            NU(1,vel) = Nu;
            RE(1,vel) = Re;
            hc(1,vel) = h;
            h_EXPECT(1,vel) = h_expect;
            V(1,vel) = velocitys(vel);
            %LSC(1,vel) = Lsc;
            Q(1,vel) = mean(q(:,end));
            %Q_EXPECT(1,vel) = q_expect;
        
        end
        if simuExport == 1
            save([datapath,modelname_prime,row,'_NU','.mat'],'NU');
            save([datapath,modelname_prime,row,'_RE','.mat'],'RE');
            save([datapath,modelname_prime,row,'_hc','.mat'],'hc');
            save([datapath,modelname_prime,row,'_h_EXPECT','.mat'],'h_EXPECT');
            save([datapath,modelname_prime,row,'_LSC','.mat'],'LSC');
            save([datapath,modelname_prime,row,'_V','.mat'],'V');
            save([datapath,modelname_prime,row,'_Q','.mat'],'Q');
            save([datapath,modelname_prime,row,'_Q_EXPECT','.mat'],'Q_EXPECT');
        end
        ResearchDataX(i,(j-1)*4+1:j*4) = x_axis;
        ResearchDataY(i,(j-1)*4+1:j*4) = y_axis;
        RE_sum(i,(j-1)*4+1:j*4) = RE;
        hc_sum(i,(j-1)*4+1:j*4) = hc;
        h_EXPECT_sum(i,(j-1)*4+1:j*4) = h_EXPECT;
        plot(x_axis,hc,'o');
        hold on

    end    
end


ResearchDataX = reshape(ResearchDataX,[1,length(Heights)*length(row_spacings)*length(velocitys)]);
ResearchDataY = reshape(ResearchDataY,[1,length(Heights)*length(row_spacings)*length(velocitys)]);
RE_sum = reshape(RE_sum,[1,length(Heights)*length(row_spacings)*length(velocitys)]);
L_sum = reshape(L_sum,[1,length(Heights)*length(row_spacings)*length(velocitys)]);

save([datapath,'ResearchDataX.mat'],'ResearchDataX');
save([datapath,'ResearchDataY.mat'],'ResearchDataY');
save([datapath,'RE_sum.mat'],'RE_sum');
params0 = zeros(1,2);
%ResearchDataY = log10(ResearchDataY);

params0(1)=0.1;
%params0(2)=1/5;
%params0(3)=1/12;
params0(2)=1;


%fun = @(params,Re)params(1).*Re.^(1/5).*0.69.^(1/12)+ params(2);
fun = @(params,Re)params(1).*Re+ params(2);
TolFun = 1e-6;
TolX = 1e-6;

options = optimoptions(@lsqcurvefit,'TolFun',TolFun,'TolX',TolX);
params = lsqcurvefit(fun,params0,ResearchDataX,log10(ResearchDataY),[],[],options);

%p = polyfit(ResearchDataX,ResearchDataY,1);
%y_axis_RSQ = params(1).*RE_sum.^(1/5).*0.69.^(1/12) + params(2);%polyval(p,ResearchDataX);
y_axis_RSQ = params(1).*ResearchDataX + params(2);%polyval(p,ResearchDataX);
%y_axis_RSQ = 0.09.*ResearchDataX.^(1/5).*Pr.^(1/12) +1.91;%polyval(p,ResearchDataX);
%y_axis_RSQ =polyval(p,ResearchDataX);
Rsq = 1- sum((log10(ResearchDataY)-y_axis_RSQ).^2)/sum((log10(ResearchDataY)-mean(log10(ResearchDataY))).^2);
x_axis2 = linspace(13,18,100);
y_axis2 = 10.^(params(1).*x_axis2+params(2));
plot(x_axis2,y_axis2);
grid on

min(L_sum)
max(L_sum)