function importDir(arg)

addpath(pwd());

% Recursively add Manopt directories to the Matlab path.
cd('RFedAGS');
addpath(genpath(pwd()));
cd('..');
cd('manopt');
addpath(genpath(pwd()));
cd('..');

% Ask user if the path should be saved or not
fprintf('Manopt was added to Matlab''s path.\n');
if nargin == 0
    response = input('Save path for future Matlab sessions? [Y/N] ', 's');
else
    response = 'N';
end
if strcmpi(response, 'Y')
    failed = savepath();
    if ~failed
        fprintf('Path saved: no need to call importmanopt next time.\n');
    else
        fprintf(['Something went wrong.. Perhaps missing permission ' ...
                 'to write on pathdef.m?\nPath not saved: ' ...
                 'please re-call importmanopt next time.\n']);
    end
else
    fprintf('Path not saved: please re-call importmanopt next time.\n');
end
