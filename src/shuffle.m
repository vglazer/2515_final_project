function data = shuffle(data) 
decksize = size(data,2);
for k = decksize:-1:2
    disp(['current card: ' num2str(k)]);
    index = ceil(rand*k);
    data = [data(:,1:decksize ~= index), data(:,index)];
end
