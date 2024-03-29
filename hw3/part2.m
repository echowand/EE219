clear;

%load data
data = importdata('ml-100k/u.data');
userId = data(:, 1);
itemId = data(:, 2);
rating = data(:, 3);

obervation=[userId,itemId,rating];
indices = crossvalind('Kfold',100000,10);
iteration=[50,100,200,500,1000,2000];
for i=1:1:6
    error=zeros(1,10);
    option.iter=iteration(i);
    option.dis=false;
    allabs=zeros(1,10);
    for j = 1:1:10
        test = (indices == j); 
        train = ~test;
        R=NaN*ones(943,1682);
        trainset=obervation(train,:);
        for m=1:1:90000
             curuser=trainset(m,1);
             curitem=trainset(m,2);
             currating=trainset(m,3);
             R(curuser,curitem)=currating;
        end
        [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100,option);
        P=A*Y;
        testset=obervation(test,:);
        currentabs=0;
        for n=1:1:10000
            curuser=testset(n,1);
            curitem=testset(n,2);
            currating=testset(n,3);
            currentabs=currentabs+abs(P(curuser,curitem)-currating);
        end
        allabs(j)=currentabs;
    end
    allabs=allabs/10000;
    disp(['iteration: ' num2str(iteration(i))])
    disp(['average absolute error: ' num2str(mean(allabs))])
    disp(['highest average absolute error: ' num2str(max(allabs))])
    disp(['lowest average absolute error: ' num2str(min(allabs))])
end
