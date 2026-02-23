function plot_Poincare_disk(P, x, namep, namex, mc,pn)
% each row of P and x denotes a point in Poincare disk of dim=2

      
    m = size(x,1);
    name_hl = ["primate.n.02",...
               "mammal.n.01",...
               "carnivore.n.01",...               
               "dog.n.01",...
               "pug.n.01",...               
               "rodent.n.01",...
               "ungulate.n.01",...
               "monkey.n.01",...
               "cow.n.01",...
               "feline.n.01",...
               "cheetah.n.01",...
               "mouse.n.01",...
               "workhorse.n.02",...
               "carthorse.n.01",...
               "horse.n.01",...
               "warhorse.n.03",...
               "dun.n.01"]; 

    figure(pn)
    n = length(mc);
    for i = 1:n
        mc1 = mc{i,1};      
        mc2 = mc{i,2};
        indx1 = find(namep == string(mc1));
        indx2 = find(namep == string(mc2));
        hold on
        fprintf('%d, indx1:%d, indx2:%d \n', i, indx1, indx2)
        px = [P(indx1(1), 1), P(indx2(1), 1)];
        py = [P(indx1(1), 2), P(indx2(1), 2)];        
%         plt = plot(px, py, 'Color',[0 139 139]/255);
        plt = line(px, py, 'Color',[0 139 139]/255);
        plt.Color(4) = 0.9;
    end


    hold on
    figure(pn)
    n = length(namep);  
%     for i = 1:n
%         hold on
%         scatter(P(i,1), P(i,2), 40, [102/255 139/255 139/255], 'filled')
%         hold on
%         scatter(P(i,1), P(i,2), 42, [0/255 0/255 0/255],'LineWidth',2)
%     end    
    scatter(P(:,1), P(:,2), 40, [102/255 139/255 139/255], ...
            'filled')    
    scatter(P(:,1), P(:,2), 42, [0/255 0/255 0/255],'LineWidth',.8)

    for i = 1:n
        tmp = find(name_hl == namep(i));        
        if ~isempty(tmp) == 1            
            nx = char(namep(i));            
            text(P(i,1)+0.03, P(i,2)+0.008, nx(1:end-5), 'FontSize',15);
        end        
    end

%     figure(3)
%     for i = 1:m
%         hold on
%         scatter(x(i,1), x(i,2), 150, [139/255 101/255 8/255], 'red', 'filled')  
%         hold on
%         scatter(x(i,1), x(i,2), 150, [0/255 0/255 0/255], 'LineWidth',2)  
%         text(x(i,1)+0.001, x(i,2)+0.001, namex(i), 'FontSize',15);
%     end
    
    hold on
        
    ax=gca;
    ax.XAxis.Visible='off';
    ax.YAxis.Visible='off';
end

