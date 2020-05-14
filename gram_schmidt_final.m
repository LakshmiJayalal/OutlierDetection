
%%
function [metric outlier_index U flag]= gram_schmidt_final(x,D)
    [sortedx index]=sort(x);
    Dsorted=D(:,(index));
    
    
    [metric rank_formed U flag] = gram_schmidt(Dsorted);
    
    figure;
    subplot(211)
    plot(metric);
    
    title([' gram schmidt']);
    subplot(212);plot(rank_formed);title(['rank_formed vs r']);
    
    
    %%
    %find tge change point between inlier and oultier
    
    if flag==1
        changept=findchangepts(metric(size(U,2):end),"MinThreshold",0.1*max(metric(size(U,2):end)),"Stat" + ...
            "istic","mean") + size(U,2);
    else
        changept=findchangepts(metric,"MinThreshold",0.1*max(metric),"Stat" + ...
            "istic","mean");
        U=U(:,1:changept-1);
    end
    figure;plot(x);hold on;
    if ~isempty(changept)
        outlier_index=sort(index(changept(1):length(index)));
        scatter(outlier_index,x(outlier_index));
    end
end


%%
%metric vs r
function [metric rank_formed U flag] = gram_schmidt(D)
    %gram schmidt
    D=normc(D);
    U=normc(D(:,1));
    flag =0;
    for r=2:size(D,2)
        g=((eye(size(D,1))-U*U')*D(:,r));
        metric(r)=norm(g);
        if norm(g)>1e-2
            if flag==0
                U=[U g./norm(g)];
            end
        else
            flag=1;
        end
        rank_formed(r)=rank(U);
    end
end