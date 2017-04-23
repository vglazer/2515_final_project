tcr1 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    tcr1(i) = nbayes(lingspam{i}); 
end
tcr9 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    tcr9(i) = nbayes(lingspam{i},0.9); 
end

arctcr1 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    arctcr1(i) = arcnbayes(lingspam{i}); 
end
arctcr9 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    arctcr9(i) = arcnbayes(lingspam{i},0.9); 
end

boosttcr1 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    boosttcr1(i) = boostnbayes(lingspam{i}); 
end
boosttcr9 = zeros(1,length(nfeatures));
for i = 1:length(nfeatures)
    boosttcr9(i) = boostnbayes(lingspam{i},0.9); 
end

clear i;
