% close all
clear
% get exported data
AP_Na = importdata('HH-radius-beading-Geo1-Na-conserving2.txt');

% measuring locations
x = [-90;-0.00302;277];

T_Na = zeros(size(AP_Na,1)-3,size(AP_Na,2));


for l=1:size(AP_Na,1)-3   
    T_Na(l,:) = [str2double(AP_Na{l+3,1}) str2double(AP_Na{l+3,2})];
end

aux = find(T_Na(:,1) == 0);
t = T_Na(1:aux(2)-1,1);

V_Na = [T_Na(1:aux(2)-1,2), T_Na(aux(2):aux(3)-1,2), T_Na(aux(3):end,2)];

v_Na = (x(3)-x(2))/(t(V_Na(:,3) == max(V_Na(:,3)))-t(V_Na(:,2) == max(V_Na(:,2))));

figure
set(gcf, 'Position',  [50, 50, 800, 400])
plot(t,V_Na(:,1),'b','linewidth',2)
hold on 
plot(t,V_Na(:,2),'g','linewidth',2)
plot(t,V_Na(:,3),'r','linewidth',2)
set(gca,'FontSize',14)
xlabel('time [ms]')
ylabel('V [mV]')
legend(num2str(x(1)),num2str(x(2)),num2str(x(3)))
title('Periodic Na channels, current 30 {\mu}A/cm^2')
grid on 
axis([0 10 -15 112])

