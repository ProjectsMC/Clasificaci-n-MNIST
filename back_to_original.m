function [real_group]=back_to_original(labels,cluster_per_class)
[D,N]=size(labels');
real_group=zeros(1,N);
for i=1:N
    j=1;
    for j=1:9
        if labels(i)<=sum(cluster_per_class(1:j))
            real_group(i)=j;
            break;
        else
            j=j+1;
         
        end
    end
     i=i+1;
end
end