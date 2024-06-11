function [ind_train,ind_test]=get_percentage_A(fraction,data)
ones=[];
twos=[];
threes=[];
fours=[];
fives=[];
sixs=[];
sevens=[];
eights=[];
nines=[];
zeros=[];
for i=1:10000
    if data.label(1,i)==1
    ones=[ones,i];
        elseif data.label(1,i)==2
        twos=[twos,i];
            elseif data.label(1,i)==3
            threes=[threes,i];
            elseif data.label(1,i)==4
            fours=[fours,i];
                elseif data.label(1,i)==5
                fives=[fives,i];
                    elseif data.label(1,i)==6
                    sixs=[sixs,i];
                elseif data.label(1,i)==7
                sevens=[sevens,i];
            elseif data.label(1,i)==8
            eights=[eights,i];
        elseif data.label(1,i)==9
        nines=[nines,i];
    elseif data.label(1,i)==0
        zeros=[zeros,i];
    end
end

flag=fraction*1000;
ind=randperm(1000);
    ind_train=[nines(ind(1:flag))';sevens(ind(1:flag))';fours(ind(1:flag))';twos(ind(1:flag))';fives(ind(1:flag))';
        threes(ind(1:flag))';ones(ind(1:flag))';sixs(ind(1:flag))';eights(ind(1:flag))';zeros(ind(1:flag))'];
    ind_test=[nines(ind((flag+1):1000))';sevens(ind((flag+1):1000))';fours(ind((flag+1):1000))';twos(ind((flag+1):1000))';fives(ind((flag+1):1000))';
        threes(ind((flag+1):1000))';ones(ind((flag+1):1000))';sixs(ind((flag+1):1000))';eights(ind((flag+1):1000))';zeros(ind((flag+1):1000))'];
end