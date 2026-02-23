[r1, c1] = findMinIndex(dist1, 1);
[r2, c2] = findMinIndex(dist2, 1);
[r3, c3] = findMinIndex(dist3, 1);

x1_ = infos1{c1}.xs{r1-1};
x2_ = infos2{c2}.xs{r2-1};
x3_ = infos3{c3}.xs{r3-1};

% plot poincare disk

xtrue_ = lor2poin(xtrue);
x1_    = lor2poin(x1_);
x2_    = lor2poin(x2_);
x3_    = lor2poin(x3_);

P = xlsread('poincare_points.xlsx');
np = readcell('names.xlsx');
mc = readcell('edgelist.xlsx');

namep = [];
for i = 1:length(np)
    namep = [namep, string(np{i})];
end

x = [xtrue_';x1_'; x2_'; x3_'];
namex = [" ", "RFedAGS", "RFedAvg", "RFedSVRG"];

figure(4)
plot_Poincare_disk(P, x, namep, namex, mc,4);

hold on
figure(4)
scatter(xtrue_(1), xtrue_(2), 100, [102/255 139/255 139/255], 'filled', 'LineWidth',2);
hold on
h1 = scatter(x1_(1), x1_(2), 150, [139/255 101/255 8/255], "filled", 'h', 'LineWidth',2);
hold on
h2 = scatter(x2_(1), x2_(2), 150, [139/255 101/255 8/255], 'filled', 'diamond','LineWidth',2);
hold on
h3 = scatter(x3_(1), x3_(2), 150, [139/255 101/255 8/255], "filled", 'square' ,'LineWidth',2);
[~, ldg] = legend([h1, h2, h3], "RFedAGS", "RFedAvg", "RFedSVRG",'FontSize',25,'Location','eastoutside','Orientation','vertical');
ldg(4).Children.MarkerSize = 15;
ldg(5).Children.MarkerSize = 15;
ldg(6).Children.MarkerSize = 15;


function [row, col] = findMinIndex(X, dim)
    if nargin == 1
        dim = 1;
    end
    [n, m] = size(X);
    indx = find(X == min(min(X, [], dim)));
    col = ceil(indx / n);
    row = indx - (col - 1) * n;   
end