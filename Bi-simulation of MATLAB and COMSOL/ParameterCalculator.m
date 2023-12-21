classdef ParameterCalculator
    
    methods(Static)

        function Nu = NusseltNumber_cal(q,T,T_ref,H,k,calMethod)
            h = ParameterCalculator.convcof_cal(q,T,T_ref,calMethod);
            Nu = h*H/k;
        end

        function Re = Reynolds_cal(U,Lsc,Kv)

            %if size(Kv,2)>1
            %    Kv = Kv(:,end);
            %end
            %Kv = Kv(~isnan(Kv));
            %Kv = mean(Kv(:,1));
            Re = U*Lsc./Kv;
        end

        function h = convcof_cal(q,T,T_ref,calMethod)

            %if size(T,2)>1 || size(q,2)>1
            %    q = q(:,end);
            %    T = T(:,end);
            %end

            %if calMethod == 1

                %T_delta = T - T_ref;
            %    T_delta = mean(T) - T_ref;
            %    h = q./T_delta;
                %len = size(h,1);
                %h = mean(h);%*(1+1/std(h));
            %else
                
                %T_delta = mean(T) - T_ref;
            %    T_delta = mean(T) - T_ref;
                %T_delta = mean(T_delta);
            %    q = mean(abs(q));
                %h = q/T_delta;

            %end
            %h = h/53;
            h = q/(T-T_ref);
            
        end

        function h_expect = convexp_cal(Re,Pr,k,H)

            h_expect = 10.^(Re.^(1/5).*Pr.^(1/12).*0.1884+0.2858).*k./H;

        end

        function RP = PowerRatio(T,T_ref1)
            if size(T,2)>1
                T = T(:,end);
            end
            %sigma = T-T_abs;
            %idx = find(sigma>0);
            %T = T(idx);
            T_delta1 = T - T_ref1;
            %idx1 = find(T_delta1>0);
            %T_delta1 = T_delta1(idx1);
            RP = abs(mean(T_delta1)*(1+1/std(T_delta1)));%*(1+std(T_delta1));
            %RP = abs(mean(T_delta1)/(std(T_delta1)));

        end
    end
end