function myplot(Xs, Ys, options)
% Note that Markers is cell
% the first
set(0,'defaultaxesfontsize',22, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)
    
    [n, p] = size(Ys);
        
    if ~isfield(options, 'Markers')
        Markers = cell(p,1); for i = 1:p, Markers{i} = '-';  end
    else
        Markers = options.Markers;
    end    
    if ~isfield(options, 'number')
        number = 1;
    else
        number = options.number;
    end
    if ~isfield(options, 'Color4')  % Curve transparency
        Color4 = 1;
    else
        Color4 = options.Color4;
    end
    if ~isfield(options, 'Markersize')
        Markersize = 8;  % defualt = 10
    else
        Markersize = options.Markersize;
    end
    if ~isfield(options, 'type')
        type = "semilogy";
    else
        type = options.type;
    end    
    if ~isfield(options, 'MarkerIndices')
        MarkerIndices = ones(p,1);
    else
        MarkerIndices = options.MarkerIndices;
    end     
    if ~isfield(options, 'Fontsize')
        Fontsize = 20;
    else
        Fontsize = options.Fontsize;
    end
    if ~isfield(options, 'LineWidth')
        LineWidth = 2.5;
    else
        LineWidth = options.LineWidth;
    end
    if ~isfield(options, 'filePath')
        filePath = [];
    else
        filePath = options.filePath;
    end       
    if strcmp(type, 'semilogy')
        plt = @semilogy;
    else
        plt = @plot;
    end
    
    figure(number)
    pic = plt(Xs(:,1), Ys(:,1), Markers{1}, 'MarkerSize', Markersize,'LineWidth', LineWidth, 'MarkerIndices',1:MarkerIndices(1):n);
    grid on
    pic.Color(4) = Color4;
    for i = 2:p
        figure(number)
        hold on 
        pic = plt(Xs(:,i), Ys(:,i), Markers{i}, 'MarkerSize', Markersize, 'LineWidth', LineWidth, 'MarkerIndices',1:MarkerIndices(i):n);
        grid on
        pic.Color(4) = Color4;
    end
        
    if isfield(options, 'title')
        title(options.title, 'FontSize', Fontsize, 'Interpreter', 'latex')
    end
    if isfield(options, 'xlabel')
        xlabel(options.xlabel, 'FontSize', Fontsize, 'Interpreter', 'latex') 
    end
    if isfield(options, 'ylabel')
        ylabel(options.ylabel, 'FontSize', Fontsize, 'Interpreter', 'latex')
    end
    if isfield(options, 'legends')
        legend(options.legends, 'FontSize', Fontsize, 'Interpreter', 'latex')
    end

    % save 
    if ~isempty(filePath)
        saveas(gca, filePath) 
    end
end

