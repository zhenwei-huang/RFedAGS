% clear;clc

rng(0)

table = dlmread('ratings.dat');
table = table(:,[1,3,5]);

table(:,3) = normalize(table(:,3));

nusers = 6040;
nmovies = 3952;

dataset = cell(nusers,1);
dataset_train = cell(nusers,1);
dataset_test  = cell(nusers,1);
for i = 1:nusers
    set = find(table(:,1) == i);
    tmp = sparse(nmovies,1);
    tmp(table(set,2)) = table(set,3);
    dataset{i}.X = tmp;
    dataset{i}.Omega = sort(table(set,2));
    
    len = length(set);
    per = randperm(len, ceil(0.8*len));
    set_train = set(per);
    set(per) = [];
    set_test  = set;
    
    tmp_train = sparse(nmovies,1);    
    tmp_train(table(set_train,2)) = table(set_train,3);
    dataset_train{i}.X = tmp_train;
    dataset_train{i}.Omega = sort(table(set_train,2));

    tmp_test = sparse(nmovies,1); 
    tmp_test(table(set_test,2)) = table(set_test,3);
    dataset_test{i}.X = tmp_test;
    dataset_test{i}.Omega = sort(table(set_test,2));
end